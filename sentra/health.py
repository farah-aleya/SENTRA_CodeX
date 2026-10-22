"""SENTRA camera health monitor.

Catches the problem from the USJ Heights case: cameras that look installed
but are not actually working.

  CAMERA_OFFLINE  no frames received for offline_seconds          HIGH
  CAMERA_BLOCKED  lens covered, sprayed or pointed at a wall      HIGH
  CAMERA_FROZEN   the picture has not changed at all              MEDIUM

Works on tiny greyscale thumbnails, so it costs almost nothing per frame.
"""
import time
from collections import deque

import cv2
import numpy as np

from .rules import Event

HEALTH_INFO = {
    "CAMERA_OFFLINE": ("HIGH", "Camera offline"),
    "CAMERA_BLOCKED": ("HIGH", "Camera blocked"),
    "CAMERA_FROZEN": ("MEDIUM", "Camera frozen"),
}

DEFAULT_HEALTH = {
    "offline_seconds": 5,      # no frames for this long = offline
    "frozen_seconds": 5,       # identical picture for this long = frozen
    "blocked_seconds": 2,      # covered / no detail for this long = blocked
    "baseline_seconds": 3,     # first seconds used to learn what "normal" looks like
    "frozen_diff": 0.15,       # mean pixel change below this = no change at all (real freezes repeat the exact same frame)
    "blocked_detail": 0.3,     # detail drops below 30% of normal ...
    "blocked_contrast": 35,    # ... and contrast is this low = blocked (both smoothed over ~0.5 s)
    "dark_level": 12,          # or the picture is almost black
}


class HealthMonitor:
    def __init__(self, cfg: dict, camera_id: str, camera_name: str):
        self.p = {**DEFAULT_HEALTH, **(cfg.get("health") or {})}
        self.camera_id, self.camera_name = camera_id, camera_name
        self.prev = None
        self.t_start = None
        self.last_frame_t = None
        self._base = []
        self.baseline_detail = None
        self._since = {"CAMERA_FROZEN": None, "CAMERA_BLOCKED": None}
        self.active = set()        # problems currently happening
        self.status = "STARTING"
        self.metrics = {}
        self._recent = deque(maxlen=12)  # (detail, contrast) of recent frames, smoothed to ignore flicker

    # ------------------------------------------------------------------
    def _event(self, rule, t, msg):
        sev, title = HEALTH_INFO[rule]
        return Event(rule, sev, title, msg, self.camera_id, self.camera_name, "", "camera", -1, "camera",
                     round(t, 1), time.strftime("%Y-%m-%d %H:%M:%S"))

    def _set_status(self):
        for r in ("CAMERA_OFFLINE", "CAMERA_BLOCKED", "CAMERA_FROZEN"):
            if r in self.active:
                self.status = r.replace("CAMERA_", "")
                return
        self.status = "OK" if self.baseline_detail else "LEARNING"

    def _check(self, rule, condition, t, need_s, msg, out, restored):
        if condition:
            if self._since[rule] is None:
                self._since[rule] = t
            if rule not in self.active and t - self._since[rule] >= need_s:
                self.active.add(rule)
                out.append(self._event(rule, t, msg))
        else:
            self._since[rule] = None
            if rule in self.active:
                self.active.discard(rule)
                restored.append(HEALTH_INFO[rule][1])

    # ------------------------------------------------------------------
    def update(self, frame, t: float):
        """Call for every frame received. Returns (new_events, restored_problem_names)."""
        out, restored = [], []
        if self.t_start is None:
            self.t_start = t
        if "CAMERA_OFFLINE" in self.active:
            self.active.discard("CAMERA_OFFLINE")
            restored.append("Camera offline")
        self.last_frame_t = t

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        detail = float(cv2.Laplacian(small, cv2.CV_64F).var())
        contrast = float(small.std())
        brightness = float(small.mean())
        # frame-to-frame change on a full-resolution centre crop, so normal sensor
        # noise still counts as "alive" while a frozen feed reads close to zero
        h, w = gray.shape
        crop = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        diff = float(np.abs(crop.astype(np.int16) - self.prev.astype(np.int16)).mean()) if self.prev is not None else 99.0
        self.prev = crop
        self.metrics = dict(detail=round(detail, 1), contrast=round(contrast, 1),
                            brightness=round(brightness, 1), change=round(diff, 2))

        # learn what a healthy picture looks like
        if self.baseline_detail is None:
            self._base.append(detail)
            if t - self.t_start >= self.p["baseline_seconds"]:
                self.baseline_detail = max(float(np.median(self._base)), 1.0)
            self._set_status()
            return out, restored

        p = self.p
        self._recent.append((detail, contrast))
        d_med = float(np.median([d for d, _ in self._recent]))
        c_med = float(np.median([c for _, c in self._recent]))
        blocked = brightness < p["dark_level"] or (
            d_med < p["blocked_detail"] * self.baseline_detail and c_med < p["blocked_contrast"])
        self._check("CAMERA_BLOCKED", blocked, t, p["blocked_seconds"],
                    "Camera view is covered or blocked (no detail in the picture)", out, restored)
        self._check("CAMERA_FROZEN", diff < p["frozen_diff"] and not blocked, t, p["frozen_seconds"],
                    f"Picture has not changed for {p['frozen_seconds']}s (feed may be frozen)", out, restored)
        self._set_status()
        return out, restored

    def no_frame(self, t: float):
        """Call when a frame could not be read. Returns new events."""
        out = []
        if self.last_frame_t is None:
            self.last_frame_t = t
        if "CAMERA_OFFLINE" not in self.active and t - self.last_frame_t >= self.p["offline_seconds"]:
            self.active.add("CAMERA_OFFLINE")
            out.append(self._event("CAMERA_OFFLINE", t, f"No video received for {self.p['offline_seconds']}s"))
        self._set_status()
        if "CAMERA_OFFLINE" not in self.active:
            self.status = "NO SIGNAL"                  # waiting to see if the feed comes back
        return out
