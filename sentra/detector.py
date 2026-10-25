"""SENTRA detector: YOLOv8n run by OpenCV 5 DNN + SENTRA's own IoU tracker.

No PyTorch or Ultralytics at runtime. The model is a plain ONNX file
(models/yolov8n.onnx, made once with scripts/export_onnx.py) and every step
runs in OpenCV 5:

  frame -> letterbox -> cv.dnn.blobFromImage -> cv.dnn net.forward()
        -> decode boxes -> cv.dnn.NMSBoxesBatched -> IoUTracker -> Track list

Turns each video frame into a list of tracked objects (people and vehicles)
with a stable track ID, so later steps can apply rules such as loitering
or fence crossing.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sentra.tracker import IoUTracker

# COCO class IDs we care about
PERSON = 0
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRACKED_CLASSES = [PERSON, *VEHICLES.keys()]

ENGINES = {"auto": cv2.dnn.ENGINE_AUTO, "new": cv2.dnn.ENGINE_NEW, "classic": cv2.dnn.ENGINE_CLASSIC}


@dataclass
class Track:
    track_id: int
    label: str          # "person" or vehicle type
    conf: float
    box: tuple          # (x1, y1, x2, y2) in pixels

    @property
    def is_person(self) -> bool:
        return self.label == "person"

    @property
    def foot(self) -> tuple:
        """Bottom-centre of the box: where the object touches the ground.
        Used to test whether someone is inside a zone."""
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int(y2))


def resolve_weights(weights: str) -> Path:
    """Accept the old models/yolov8n.pt path and use the .onnx next to it."""
    p = Path(weights)
    if p.suffix != ".onnx":
        p = p.with_suffix(".onnx")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run once: python scripts/export_onnx.py")
    return p


class Detector:
    def __init__(self, weights: str = "models/yolov8n.onnx", conf: float = 0.35, imgsz: int = 640,
                 device=None, engine: str = "auto", nms_iou: float = 0.5):
        self.weights = resolve_weights(weights)
        self.net = cv2.dnn.readNetFromONNX(str(self.weights), ENGINES.get(engine, cv2.dnn.ENGINE_AUTO))
        self.conf = conf
        self.low_conf = 0.1                      # second-stage threshold for the tracker
        self.imgsz = imgsz
        self.nms_iou = nms_iou
        self.device = f"OpenCV {cv2.__version__} DNN (CPU)"
        self.tracker = IoUTracker(high_conf=conf, low_conf=self.low_conf)
        self.last_ms = 0.0                       # inference time of the last frame, for evaluation

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, px, py

    def detect(self, frame, min_conf=None) -> list:
        """Run YOLOv8n with OpenCV 5 DNN. Returns list of (box, conf, label), no tracking."""
        min_conf = self.low_conf if min_conf is None else min_conf
        t0 = time.perf_counter()
        img, r, px, py = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()                 # (1, 84, N): 4 box values + 80 class scores
        self.last_ms = (time.perf_counter() - t0) * 1000

        preds = np.squeeze(out, 0).T             # (N, 84)
        scores_all = preds[:, 4:][:, TRACKED_CLASSES]
        best = scores_all.argmax(1)
        scores = scores_all[np.arange(len(best)), best]
        keep = scores >= min_conf
        if not keep.any():
            return []
        preds, scores = preds[keep], scores[keep]
        cls_ids = np.array(TRACKED_CLASSES)[best[keep]]

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2 - px) / r
        y1 = (cy - bh / 2 - py) / r
        boxes_xywh = np.stack([x1, y1, bw / r, bh / r], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes_xywh.tolist(), scores.tolist(), cls_ids.tolist(),
                                      min_conf, self.nms_iou)
        h, w = frame.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw_, bh_ = boxes_xywh[i]
            box = (max(0, x), max(0, y), min(w - 1, x + bw_), min(h - 1, y + bh_))
            label = "person" if cls_ids[i] == PERSON else VEHICLES.get(int(cls_ids[i]), "vehicle")
            dets.append((box, float(scores[i]), label))
        return dets

    def track(self, frame) -> list[Track]:
        """Detect + track on one frame. Call once per frame, in order."""
        tracked = self.tracker.update(self.detect(frame))
        return [Track(tid, label, conf, box) for tid, box, conf, label in tracked]
