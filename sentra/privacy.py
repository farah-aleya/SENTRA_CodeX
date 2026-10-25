"""SENTRA privacy filter: blur faces and heads BEFORE any frame is stored or shown.

Two layers, both pure OpenCV 5:
  1. Face detection with YuNet (cv.FaceDetectorYN, OpenCV 5 DNN) on the whole frame.
     Catches close-up faces, e.g. a doorbell camera, even when no person box exists.
  2. Head region of every tracked person (top part of the person box).
     Catches people seen from behind, at night, or too small for face detection.

SENTRA never recognises or identifies anyone: faces are only located so they
can be pixelated. The raw frame is used for detection in memory and is never
written to disk. Snapshots, 10-second clips, the live console picture and the
annotated video all receive the blurred frame.
"""
from pathlib import Path

import cv2
import numpy as np

DEFAULTS = {
    "blur": True,              # master switch (keep on; turn off only for offline evaluation)
    "faces": True,             # layer 1: YuNet face detection
    "heads": True,             # layer 2: top of every person box
    "head_ratio": 0.22,        # fraction of the person box height treated as head
    "face_score": 0.6,         # YuNet confidence; lower blurs more (privacy errs on the safe side)
    "face_width": 640,         # frame is resized to this width for face detection (speed)
    "pixel_block": 10,         # bigger = coarser pixelation
}


def pixelate(frame, x1, y1, x2, y2, block=10):
    """Irreversibly pixelate a region in place."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


class PrivacyFilter:
    def __init__(self, cfg: dict = None, model="models/face_detection_yunet_2023mar.onnx"):
        self.cfg = {**DEFAULTS, **((cfg or {}).get("privacy") or {})}
        self.face_net = None
        self.faces_blurred = 0
        self.heads_blurred = 0
        if self.cfg["blur"] and self.cfg["faces"]:
            if Path(model).exists():
                self.face_net = cv2.FaceDetectorYN.create(str(model), "", (320, 320),
                                                          self.cfg["face_score"], 0.3, 50)
            else:
                print(f"[privacy] {model} not found: face layer off, head blur still on")

    @property
    def enabled(self):
        return bool(self.cfg["blur"])

    def find_faces(self, frame) -> list:
        """Face boxes (x1, y1, x2, y2) in full-frame pixels. Location only, no identity."""
        if self.face_net is None:
            return []
        h, w = frame.shape[:2]
        s = min(1.0, self.cfg["face_width"] / w)
        small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        self.face_net.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.face_net.detect(small)
        boxes = []
        if faces is not None:
            for x, y, fw, fh in faces[:, :4]:
                pad = 0.25 * max(fw, fh)                    # cover hair and ears too
                boxes.append(((x - pad) / s, (y - pad) / s, (x + fw + pad) / s, (y + fh + pad) / s))
        return boxes

    def apply(self, frame, tracks=()):
        """Blur faces and heads in place. Call right after detection, before drawing or saving."""
        if not self.enabled:
            return frame
        b = self.cfg["pixel_block"]
        for box in self.find_faces(frame):
            pixelate(frame, *box, block=b)
            self.faces_blurred += 1
        if self.cfg["heads"]:
            for t in tracks:
                if not t.is_person:
                    continue
                x1, y1, x2, y2 = t.box
                bw, bh = x2 - x1, y2 - y1
                head_h = max(bh * self.cfg["head_ratio"], min(bw, bh) * 0.35)
                pixelate(frame, x1, y1 - 0.05 * bh, x2, y1 + head_h, block=b)
                self.heads_blurred += 1
        return frame
