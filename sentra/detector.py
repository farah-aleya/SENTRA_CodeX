"""SENTRA detector: YOLOv8 detection + ByteTrack tracking.

Turns each video frame into a list of tracked objects (people and vehicles)
with a stable track ID, so later steps can apply rules such as loitering
or fence crossing.
"""
from dataclasses import dataclass

from ultralytics import YOLO

# COCO class IDs we care about
PERSON = 0
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRACKED_CLASSES = [PERSON, *VEHICLES.keys()]


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


class Detector:
    def __init__(self, weights: str = "models/yolov8n.pt", conf: float = 0.35, imgsz: int = 640, device=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        if device is None:  # use the NVIDIA GPU automatically when available
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device

    def track(self, frame) -> list[Track]:
        """Run detection + ByteTrack on one frame. Call once per frame, in order."""
        results = self.model.track(
            frame,
            persist=True,                 # keep IDs across frames
            tracker="bytetrack.yaml",
            classes=TRACKED_CLASSES,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        tracks: list[Track] = []
        if r.boxes is None or r.boxes.id is None:
            return tracks
        for box, tid, cls, conf in zip(
            r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
            r.boxes.cls.int().tolist(), r.boxes.conf.tolist()
        ):
            label = "person" if cls == PERSON else VEHICLES.get(cls, "vehicle")
            tracks.append(Track(tid, label, conf, tuple(int(v) for v in box)))
        return tracks
