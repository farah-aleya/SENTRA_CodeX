"""One-time: convert models/yolov8n.pt to models/yolov8n.onnx for OpenCV 5 DNN.

  pip install ultralytics onnx          (only needed for this export, not at runtime)
  python scripts/export_onnx.py

Then check that OpenCV 5 can load and run it:
  python scripts/export_onnx.py --check
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(weights: Path, imgsz: int):
    from ultralytics import YOLO
    out = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=True)
    print(f"Saved {out}")


def check(onnx_path: Path, imgsz: int):
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from sentra.detector import Detector

    major = int(cv2.__version__.split(".")[0])
    print(f"OpenCV version: {cv2.__version__}")
    if major < 5:
        sys.exit("OpenCV 5 is required: pip install -U 'opencv-python>=5.0'")
    det = Detector(str(onnx_path), imgsz=imgsz)
    frame = np.full((480, 640, 3), 114, dtype=np.uint8)
    for _ in range(3):
        det.detect(frame)
    print(f"Model loaded with OpenCV DNN, inference {det.last_ms:.1f} ms per frame on CPU")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--check", action="store_true", help="only test the existing .onnx with OpenCV 5")
    args = ap.parse_args()
    onnx_path = Path(args.weights).with_suffix(".onnx")
    if not args.check:
        export(Path(args.weights), args.imgsz)
    check(onnx_path, args.imgsz)
