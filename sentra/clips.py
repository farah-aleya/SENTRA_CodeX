"""10-second evidence clips: a few seconds before the alert + a few seconds after.

Frames are kept in a short rolling buffer. When an alert fires, the buffer is
copied and the next seconds are added, then the clip is saved as H.264 MP4
(plays in any browser) in a background thread so detection never pauses.
"""
import threading
from collections import deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


class ClipRecorder:
    def __init__(self, fps: float, pre_seconds: float = 4, post_seconds: float = 6, max_width: int = 960):
        self.fps = max(float(fps), 1.0)
        self.pre = deque(maxlen=int(self.fps * pre_seconds))
        self.post_n = int(self.fps * post_seconds)
        self.max_width = max_width
        self.jobs = []          # [frames, remaining, path, callback]
        self.threads = []       # clip writers still running

    def _small(self, frame):
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)))
        h, w = frame.shape[:2]
        return frame[: h - h % 2, : w - w % 2]          # H.264 needs even sizes

    def push(self, frame):
        """Call once per frame with the annotated frame."""
        f = self._small(frame)
        self.pre.append(f)
        done = []
        for job in self.jobs:
            job[0].append(f)
            job[1] -= 1
            if job[1] <= 0:
                done.append(job)
        for job in done:
            self.jobs.remove(job)
            t = threading.Thread(target=self._write, args=(job[0], job[2], job[3]), daemon=True)
            t.start()
            self.threads.append(t)

    def start(self, path: str, on_done=None):
        """Begin a clip for an alert that fired on the current frame."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.jobs.append([list(self.pre), self.post_n, path, on_done])

    def flush(self):
        """Save unfinished clips (e.g. the video ended). Returns the threads to join."""
        threads = []
        for frames, _, path, cb in self.jobs:
            t = threading.Thread(target=self._write, args=(frames, path, cb)); t.start(); threads.append(t)
        self.jobs.clear()
        return threads + [t for t in self.threads if t.is_alive()]   # include clips still being written

    def _write(self, frames, path, on_done):
        if not frames:
            return
        h, w = frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(path, (w, h), fps=self.fps, codec="libx264",
                                             pix_fmt_out="yuv420p", quality=7, macro_block_size=2,
                                             ffmpeg_log_level="error")
        writer.send(None)
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.send(np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        writer.close()
        if on_done:
            on_done(path)
