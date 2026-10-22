"""Export real SENTRA output into the static prototype website (web/).

  python scripts/run_sentra.py --camera gate --config config/showcase.yaml   (repeat for each camera)
  python scripts/export_showcase.py --config config/showcase.yaml

Writes:
  web/media/cam_<id>.mp4      annotated camera video (H.264, plays in any browser)
  web/media/clip_<id>.mp4     10-second evidence clip per alert
  web/media/snap_<id>.jpg     snapshot per alert
  web/data/showcase.json      cameras + alerts (with the second each alert fired)
The page replays the videos and raises each alert at the moment SENTRA raised it.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentra import store  # noqa: E402
from sentra.notify import _caption  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()


def transcode(src: Path, dst: Path, width=640, crf=28):
    """H.264 MP4 (Safari, Chrome, Edge) + VP9 WebM fallback (any browser without H.264)."""
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", f"scale={width}:-2",
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dst)], check=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", f"scale={width}:-2",
                    "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "38", "-row-mt", "1", "-deadline", "good",
                    "-cpu-used", "4", str(dst.with_suffix(".webm"))], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/showcase.yaml")
    ap.add_argument("--out", default="web")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out = ROOT / args.out
    media = out / "media"
    if media.exists():
        shutil.rmtree(media)
    media.mkdir(parents=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    con = store.connect(ROOT / "outputs" / "sentra.db")

    cams, events = [], []
    for cid, cam in cfg["cameras"].items():
        annotated = ROOT / "outputs" / f"{cid}_annotated.mp4"
        if not annotated.exists():
            print(f"skip {cid}: run run_sentra.py --camera {cid} first")
            continue
        transcode(annotated, media / f"cam_{cid}.mp4")
        cams.append({"id": cid, "name": cam.get("name", cid), "video": f"media/cam_{cid}.mp4",
                     "zones": len(cam.get("zones") or [])})
        for e in sorted(store.list_events(con, 1000), key=lambda e: e["video_time"]):
            if e["camera_id"] != cid:
                continue
            item = {k: e[k] for k in ("id", "rule", "severity", "title", "message", "camera_id", "camera_name",
                                      "zone_name", "label", "track_id", "video_time", "house")}
            if e.get("clip") and (ROOT / e["clip"]).exists():
                transcode(ROOT / e["clip"], media / f"clip_{e['id']}.mp4", crf=27)
                item["clip"] = f"media/clip_{e['id']}.mp4"
            if e.get("snapshot") and (ROOT / e["snapshot"]).exists():
                shutil.copy(ROOT / e["snapshot"], media / f"snap_{e['id']}.jpg")
                item["snapshot"] = f"media/snap_{e['id']}.jpg"
            item["telegram"] = _caption({**e, "guard_note": ""})
            events.append(item)

    data = {"generated": time.strftime("%Y-%m-%d %H:%M"), "cameras": cams, "alerts": events,
            "rules": cfg.get("rules", {}), "away_mode": cfg.get("away_mode", [])}
    (out / "data" / "showcase.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in media.glob("*")) / 1e6
    print(f"Exported {len(cams)} cameras, {len(events)} alerts, {size:.1f} MB of media to {out}/")


if __name__ == "__main__":
    main()
