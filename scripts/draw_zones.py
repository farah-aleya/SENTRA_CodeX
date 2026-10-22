"""Draw zones on a camera view and save them to config/site.yaml.

  python scripts/draw_zones.py --camera gate01 --source samples/my_gate_clip.mp4 --name "CAM 01 Front Gate"

In the window
  Left click   add a corner point
  Backspace    undo last point
  Enter        finish this zone (then answer the questions in the terminal)
  c            clear all zones for this camera and start again
  s            save and quit
  Esc          quit without saving
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentra.draw import draw_zones  # noqa: E402
from sentra.zones import ZONE_TYPES, Zone, load_config, save_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--name", default=None, help="display name, e.g. 'CAM 01 Front Gate'")
    ap.add_argument("--frame", type=int, default=0, help="which frame of the video to draw on")
    ap.add_argument("--config", default="config/site.yaml")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, base = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Cannot read a frame from {args.source}")
    h, w = base.shape[:2]

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    cams = cfg.setdefault("cameras", {})
    cam = cams.setdefault(args.camera, {})
    cam["source"] = args.source
    cam["name"] = args.name or cam.get("name", args.camera)
    zones = [Zone(**{k: z[k] for k in ("id", "type", "polygon") if k in z}, name=z.get("name", ""), house=z.get("house"))
             for z in cam.get("zones", [])]
    points = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    win = "SENTRA zones - click corners, Enter = finish zone, s = save"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = base.copy()
        draw_zones(img, zones)
        for i, p in enumerate(points):
            cv2.circle(img, p, 5, (0, 255, 255), -1)
            if i:
                cv2.line(img, points[i - 1], p, (0, 255, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k == 8 and points:                       # Backspace
            points.pop()
        elif k == 13 and len(points) >= 3:          # Enter
            print("\nZone types: restricted (behind fence), house (front of a house), road (street / car park)")
            ztype = ""
            while ztype not in ZONE_TYPES:
                ztype = input("  Zone type: ").strip().lower()
            house = input("  House number (e.g. 12): ").strip() if ztype == "house" else None
            default_id = f"house_{house}" if house else f"{ztype}_{len(zones) + 1}"
            zid = input(f"  Zone id [{default_id}]: ").strip() or default_id
            name = input("  Display name (optional): ").strip()
            zones.append(Zone(zid, ztype, [[round(x / w, 4), round(y / h, 4)] for x, y in points], name, house))
            points.clear()
            print(f"  Added zone '{zid}'. Draw another, or press s to save.")
        elif k == ord("c"):
            zones.clear(); points.clear()
        elif k == ord("s"):
            cam["zones"] = [{k2: v for k2, v in dict(id=z.id, type=z.type, name=z.name, house=z.house,
                                                        polygon=z.polygon).items() if v not in (None, "")}
                            for z in zones]
            save_config(cfg, args.config)
            Path("outputs").mkdir(exist_ok=True)
            preview = base.copy(); draw_zones(preview, zones)
            cv2.imwrite(f"outputs/zones_{args.camera}.jpg", preview)
            print(f"Saved {len(zones)} zones for '{args.camera}' to {args.config}")
            break
        elif k == 27:
            print("Quit without saving")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
