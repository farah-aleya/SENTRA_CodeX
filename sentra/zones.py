"""Zones: areas drawn on a camera view.

Zone types
  restricted : private area behind a fence or wall. A person here = perimeter breach.
  house      : the front of one house (gate, porch, door). Used for loitering and Away Mode.
  road       : street or car park area. Used for vehicle dwell.

Polygons are stored as fractions of the frame (0 to 1), so they still fit if
the video resolution changes.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

ZONE_TYPES = ("restricted", "house", "road")


@dataclass
class Zone:
    id: str
    type: str
    polygon: list                 # [[x, y], ...] normalised 0..1
    name: str = ""
    house: str | None = None      # house number for type == "house"
    _px: np.ndarray | None = field(default=None, repr=False)

    def pixels(self, w: int, h: int) -> np.ndarray:
        if self._px is None or self._px.shape[0] != len(self.polygon):
            self._px = np.array([[int(x * w), int(y * h)] for x, y in self.polygon], dtype=np.int32)
        return self._px

    def contains(self, pt, w: int, h: int) -> bool:
        return cv2.pointPolygonTest(self.pixels(w, h), (float(pt[0]), float(pt[1])), False) >= 0

    @property
    def label(self) -> str:
        return self.name or (f"House {self.house}" if self.house else self.id)


def load_config(path="config/site.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict, path="config/site.yaml"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def camera_zones(cfg: dict, camera_id: str) -> list[Zone]:
    cam = cfg.get("cameras", {}).get(camera_id, {})
    return [Zone(id=z["id"], type=z["type"], polygon=z["polygon"], name=z.get("name", ""), house=z.get("house"))
            for z in cam.get("zones", [])]
