"""Make a short synthetic test clip from Ultralytics' bundled sample images.

Only for checking the pipeline works before you record real footage.
The camera "pans" across the images so people move and tracking can be tested.

  python scripts/make_test_video.py
"""
import os
from pathlib import Path

import cv2
import ultralytics

assets = Path(os.path.dirname(ultralytics.__file__)) / "assets"
out = Path("samples/test_walk.mp4")
out.parent.mkdir(exist_ok=True)

W, H, FPS, SECONDS = 960, 540, 25, 8
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for name in ("bus.jpg", "zidane.jpg"):
    img = cv2.imread(str(assets / name))
    scale = (H * 1.3) / img.shape[0]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    n = FPS * SECONDS // 2
    for i in range(n):
        x = int((img.shape[1] - W) * i / max(n - 1, 1)) if img.shape[1] > W else 0
        y = int((img.shape[0] - H) * 0.5)
        crop = img[y:y + H, x:x + W]
        crop = cv2.copyMakeBorder(crop, 0, H - crop.shape[0], 0, W - crop.shape[1], cv2.BORDER_CONSTANT)
        writer.write(crop)
writer.release()
print(f"Saved {out}")
