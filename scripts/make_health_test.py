"""Make a test clip for the camera health check (Step 3).

Timeline (seconds)
   0-4   normal
   4-11  FROZEN   (same frame repeated)
  11-14  normal
  14-18  BLOCKED  (lens covered: dark, blurry)
  18-22  normal
  22-29  OFFLINE  (simulate with --simulate-offline 22-29)
  29-32  normal

  python scripts/make_health_test.py
  python scripts/run_sentra.py --camera healthtest --config config/demo.yaml --simulate-offline 22-29 --show
"""
import os
from pathlib import Path

import cv2
import numpy as np
import ultralytics

W, H, FPS = 640, 360, 25
img = cv2.imread(str(Path(os.path.dirname(ultralytics.__file__)) / "assets" / "zidane.jpg"))
img = cv2.resize(img, (int(img.shape[1] * (H * 1.2) / img.shape[0]), int(H * 1.2)))
rng = np.random.default_rng(0)
out = Path("samples/health_test.mp4"); out.parent.mkdir(exist_ok=True)
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


def normal(i):
    x = int((img.shape[1] - W) * (0.5 + 0.5 * np.sin(i / 40)))
    f = img[20:20 + H, x:x + W].copy()
    noise = rng.normal(0, 3, f.shape)                       # sensor noise like a real camera
    return np.clip(f + noise, 0, 255).astype(np.uint8)


def covered(i):
    f = cv2.GaussianBlur(normal(i), (0, 0), 25)              # hand over the lens: blurry and dark
    f = (f * 0.25 + np.array([20, 35, 60]) * 0.75).astype(np.uint8)
    return np.clip(f + rng.normal(0, 1.5, f.shape), 0, 255).astype(np.uint8)


frame_no = 0
for sec_from, sec_to, kind in [(0, 4, "n"), (4, 11, "frozen"), (11, 14, "n"), (14, 18, "covered"),
                               (18, 32, "n")]:
    held = None
    for _ in range((sec_to - sec_from) * FPS):
        frame_no += 1
        if kind == "frozen":
            held = held if held is not None else normal(frame_no)
            writer.write(held)
        elif kind == "covered":
            writer.write(covered(frame_no))
        else:
            writer.write(normal(frame_no))
writer.release()
print(f"Saved {out} ({frame_no / FPS:.0f}s)")
