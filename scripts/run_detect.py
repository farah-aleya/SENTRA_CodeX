"""STEP 1 - Run SENTRA detection + tracking on a video, webcam or CCTV stream.

Examples
  python scripts/run_detect.py --source samples/test_walk.mp4
  python scripts/run_detect.py --source 0 --show                # laptop webcam
  python scripts/run_detect.py --source rtsp://user:pass@IP:554/stream1 --show

Output: an annotated video in outputs/ and a per-second log in the console.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentra.detector import Detector  # noqa: E402
from sentra.draw import draw_header, draw_tracks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path, webcam index (0) or RTSP URL")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="outputs/step1_detect.mp4")
    ap.add_argument("--show", action="store_true", help="open a live preview window")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    det = Detector(args.weights, conf=args.conf)
    frame_no, t0, seen = 0, time.time(), set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        tracks = det.track(frame)
        seen.update((t.label, t.track_id) for t in tracks)

        draw_tracks(frame, tracks)
        draw_header(frame, f"SENTRA  |  frame {frame_no}  |  people {sum(t.is_person for t in tracks)}"
                           f"  vehicles {sum(not t.is_person for t in tracks)}")
        writer.write(frame)

        if frame_no % int(fps) == 0:
            print(f"[{frame_no / fps:6.1f}s] " + (", ".join(f"{t.label}#{t.track_id}" for t in tracks) or "nothing"))
        if args.show:
            cv2.imshow("SENTRA - Step 1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    dt = time.time() - t0
    people = len({i for l, i in seen if l == "person"})
    vehicles = len({i for l, i in seen if l != "person"})
    print(f"\nDone: {frame_no} frames in {dt:.1f}s ({frame_no / max(dt, 1e-6):.1f} FPS). "
          f"Unique people: {people}, unique vehicles: {vehicles}. Saved: {args.out}")


if __name__ == "__main__":
    main()
