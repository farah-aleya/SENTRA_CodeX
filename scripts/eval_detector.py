"""Evaluation 1: does the OpenCV 5 DNN pipeline match the original PyTorch/Ultralytics pipeline?

Same YOLOv8n weights, two runtimes:
  A) SENTRA:   models/yolov8n.onnx in OpenCV 5 cv.dnn (letterbox, decode, NMSBoxesBatched)  <- what SENTRA runs
  B) baseline: models/yolov8n.pt in Ultralytics + PyTorch (CPU)                              <- the old SENTRA

Two tests
  1. Labelled images (YOLO-format dataset, e.g. COCO128): precision, recall and AP@0.5 for the classes
     SENTRA uses (person, car, motorcycle, bus, truck), for both runtimes.
  2. CCTV frames (samples/*.mp4, no labels): agreement between the two runtimes, box IoU and
     confidence difference, and speed per frame.

  pip install ultralytics                      (baseline only; SENTRA itself does not need it)
  python scripts/eval_detector.py --images datasets/coco128 --videos samples
Writes eval/detector_results.json and prints Markdown tables for the report.
"""
import argparse
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentra.detector import PERSON, TRACKED_CLASSES, VEHICLES, Detector  # noqa: E402

NAMES = {PERSON: "person", **VEHICLES}
GROUP = lambda label: "person" if label == "person" else "vehicle"  # noqa: E731


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Torch:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.m = YOLO(weights)

    def detect(self, frame, min_conf):
        r = self.m.predict(frame, classes=TRACKED_CLASSES, conf=min_conf, imgsz=640, device="cpu", verbose=False)[0]
        return [(tuple(b), float(c), NAMES[int(k)]) for b, c, k in
                zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


def timed(fn, *a):
    t = time.perf_counter()
    out = fn(*a)
    return out, (time.perf_counter() - t) * 1000


# ------------------------------------------------------------------ 1. labelled images
def load_labels(img_path, w, h):
    lab = Path(str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
    out = []
    if lab.exists():
        for line in lab.read_text().split("\n"):
            p = line.split()
            if len(p) >= 5 and int(p[0]) in NAMES:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, NAMES[c]))
    return out


def ap50(dets, n_gt):
    """dets: list of (conf, is_true_positive). VOC-style all-point AP."""
    if n_gt == 0:
        return None
    dets = sorted(dets, key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in dets]); fp = np.cumsum([not d[1] for d in dets])
    rec = tp / n_gt if len(dets) else np.array([0.0])
    prec = tp / np.maximum(tp + fp, 1e-9) if len(dets) else np.array([0.0])
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def eval_images(folder, runtimes, op_conf=0.35):
    imgs = sorted(glob.glob(str(Path(folder) / "images" / "**" / "*.jpg"), recursive=True))
    res = {}
    for name, rt in runtimes.items():
        per = {g: {"dets": [], "n_gt": 0, "tp": 0, "fp": 0} for g in ("person", "vehicle")}
        ms = []
        for p in imgs:
            im = cv2.imread(p)
            h, w = im.shape[:2]
            gts = load_labels(p, w, h)
            dets, t = timed(rt.detect, im, 0.01)
            ms.append(t)
            for g in per:
                gt = [x for x in gts if GROUP(x[4]) == g]
                per[g]["n_gt"] += len(gt)
                used = set()
                for box, conf, label in sorted([d for d in dets if GROUP(d[2]) == g], key=lambda d: -d[1]):
                    best, bj = 0.0, -1
                    for j, x in enumerate(gt):
                        if j not in used and (g == "vehicle" or x[4] == label):
                            v = iou(box, x[:4])
                            if v > best:
                                best, bj = v, j
                    hit = best >= 0.5
                    if hit:
                        used.add(bj)
                    per[g]["dets"].append((conf, hit))
                    if conf >= op_conf:
                        per[g]["tp" if hit else "fp"] += 1
        out = {}
        for g, d in per.items():
            out[g] = {"AP50": round(ap50(d["dets"], d["n_gt"]) or 0, 3),
                      f"precision@{op_conf}": round(d["tp"] / max(d["tp"] + d["fp"], 1), 3),
                      f"recall@{op_conf}": round(d["tp"] / max(d["n_gt"], 1), 3),
                      "ground_truth_objects": d["n_gt"]}
        out["images"] = len(imgs)
        res[name] = out
    return res


# ------------------------------------------------------------------ 2. CCTV frames
def eval_videos(folder, sentra, torch, per_video=30, conf=0.35):
    vids = [v for v in sorted(glob.glob(str(Path(folder) / "*.mp4"))) if not Path(v).name.startswith("test_")]
    rows = []
    for v in vids:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats = {"video": Path(v).name, "frames": 0, "opencv5": 0, "pytorch": 0, "matched": 0, "ious": [], "dconf": [],
                 "ms_cv": [], "ms_pt": []}
        for k in range(per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / per_video))
            ok, f = cap.read()
            if not ok:
                continue
            a, ta = timed(sentra.detect, f, conf)
            b, tb = timed(torch.detect, f, conf)
            stats["frames"] += 1; stats["ms_cv"].append(ta); stats["ms_pt"].append(tb)
            stats["opencv5"] += len(a); stats["pytorch"] += len(b)
            used = set()
            for box, c, lab in a:
                best, bj = 0.0, -1
                for j, (bb, cc, ll) in enumerate(b):
                    if j not in used and GROUP(ll) == GROUP(lab):
                        v_ = iou(box, bb)
                        if v_ > best:
                            best, bj = v_, j
                if best >= 0.5:
                    used.add(bj); stats["matched"] += 1; stats["ious"].append(best); stats["dconf"].append(abs(c - b[bj][1]))
        rows.append(stats)
    tot = {k: sum(r[k] for r in rows) for k in ("frames", "opencv5", "pytorch", "matched")}
    allr = lambda key: [x for r in rows for x in r[key]]  # noqa: E731
    summary = {
        "frames": tot["frames"],
        "boxes_opencv5": tot["opencv5"], "boxes_pytorch": tot["pytorch"], "matched": tot["matched"],
        "agreement_f1": round(2 * tot["matched"] / max(tot["opencv5"] + tot["pytorch"], 1), 3),
        "mean_iou_matched": round(float(np.mean(allr("ious"))), 3) if allr("ious") else None,
        "mean_abs_conf_diff": round(float(np.mean(allr("dconf"))), 3) if allr("dconf") else None,
    }
    per_video = [{"video": r["video"], "frames": r["frames"], "opencv5_boxes": r["opencv5"], "pytorch_boxes": r["pytorch"],
                  "agreement_f1": round(2 * r["matched"] / max(r["opencv5"] + r["pytorch"], 1), 3)} for r in rows]
    return summary, per_video


def sample_frames(folder, per_video):
    frames = []
    for v in [v for v in sorted(glob.glob(str(Path(folder) / "*.mp4"))) if not Path(v).name.startswith("test_")]:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / per_video))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    return frames


def speed(rt, frames, conf=0.35, warmup=3):
    for f in frames[:warmup]:
        rt.detect(f, conf)
    ms = [timed(rt.detect, f, conf)[1] for f in frames]
    return {"frames": len(frames), "median_ms": round(float(np.median(ms)), 1), "p90_ms": round(float(np.percentile(ms, 90)), 1),
            "fps": round(1000 / float(np.median(ms)), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="YOLO-format labelled dataset folder (with images/ and labels/)")
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--per-video", type=int, default=30)
    ap.add_argument("--out", default="eval/detector_results.json")
    args = ap.parse_args()

    sentra = Detector(str(ROOT / "models" / "yolov8n.onnx"))
    # speed is measured first, OpenCV alone, before PyTorch is even imported (its idle threads would slow OpenCV down)
    frames = sample_frames(args.videos, 10) if args.videos else []
    speed_cv = speed(sentra, frames) if frames else None
    torch_rt = Torch(str(ROOT / "models" / "yolov8n.pt"))
    import torch
    speed_pt = speed(torch_rt, frames) if frames else None
    result = {"machine": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "os": platform.platform(),
                          "opencv": cv2.__version__, "torch": torch.__version__},
              "speed_cctv_frames": {"opencv5_dnn": speed_cv, "pytorch": speed_pt,
                                    "note": "end-to-end detect() per frame incl. pre/post-processing, CPU only, runtimes measured separately"}}
    if args.images:
        result["labelled_images"] = eval_images(args.images, {"opencv5_dnn": sentra, "pytorch": torch_rt})
    if args.videos:
        result["cctv_agreement"], result["cctv_per_video"] = eval_videos(args.videos, sentra, torch_rt, args.per_video)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
