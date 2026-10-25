"""Run SENTRA on one camera: detection + zones + rules (Step 2) + camera health (Step 3).

  python scripts/run_sentra.py --camera demo --config config/demo.yaml --show
  python scripts/run_sentra.py --camera healthtest --config config/demo.yaml --simulate-offline 22-29 --show
  python scripts/run_sentra.py --camera gate01 --source samples/my_gate_clip.mp4 --show

  python scripts/run_sentra.py --camera demo --config config/demo.yaml --realtime --loop   (live demo for the console)

Output
  outputs/<camera>_annotated.mp4   video with zones, timers, camera status and alert banners
  outputs/sentra.db                alerts + camera status, read by the guard console (Step 4)
  outputs/clips/*.mp4              10-second evidence clip per alert (4 s before, 6 s after)
  outputs/snapshots/*.jpg          one picture per alert
  outputs/live/<camera>.jpg        latest picture, shown live in the guard console
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentra.detector import Detector  # noqa: E402
from sentra.draw import (draw_alert_banner, draw_header, draw_no_signal, draw_status,  # noqa: E402
                         draw_tracks, draw_zones)
from sentra import store  # noqa: E402
from sentra.clips import ClipRecorder  # noqa: E402
from sentra.health import HealthMonitor  # noqa: E402
from sentra.privacy import PrivacyFilter  # noqa: E402
from sentra.agent import AGENT_RULES, AgentRunner  # noqa: E402
from sentra.rules import RuleEngine  # noqa: E402
from sentra.zones import camera_zones, load_config  # noqa: E402


def open_source(src):
    cap = cv2.VideoCapture(src)
    return cap if cap.isOpened() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="camera id from the config file")
    ap.add_argument("--config", default="config/site.yaml")
    ap.add_argument("--source", help="override the camera source (video path, 0 for webcam, RTSP URL)")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--conf", type=float, default=None, help="detection confidence (default: camera 'conf' or 0.35)")
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (default: auto)")
    ap.add_argument("--simulate-offline", default=None, metavar="START-END",
                    help="pretend the feed is lost between these video seconds, e.g. 22-29")
    ap.add_argument("--show", action="store_true", help="open a live preview window (q to quit)")
    ap.add_argument("--realtime", action="store_true", help="play video files at normal speed (for the live console)")
    ap.add_argument("--loop", action="store_true", help="restart the video file when it ends (for demos)")
    ap.add_argument("--agent", choices=["bedrock", "local", "off"], default=None,
                    help="who triages alerts: bedrock (Claude/Nova on AWS), local (offline policy), off (old behaviour)")
    ap.add_argument("--no-blur", action="store_true", help="turn OFF face/head blur (offline evaluation only, never for demos)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam = cfg.get("cameras", {}).get(args.camera)
    if cam is None:
        sys.exit(f"Camera '{args.camera}' not found in {args.config}")
    cam_name = cam.get("name", args.camera)
    source = args.source or str(cam.get("source", ""))
    src = int(source) if source.isdigit() else source
    live = isinstance(src, int) or str(src).lower().startswith(("rtsp", "http"))
    cap = open_source(src)
    if cap is None:
        sys.exit(f"Cannot open source: {source}")
    offline_window = tuple(float(x) for x in args.simulate_offline.split("-")) if args.simulate_offline else None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path("outputs")
    for d in ("snapshots", "clips", "live"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    db = store.connect()
    clips = ClipRecorder(fps)
    live_path = f"outputs/live/{args.camera}.jpg"
    out_video = out_dir / f"{args.camera}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    events_log = open(out_dir / "events.jsonl", "a", encoding="utf-8")

    zones = camera_zones(cfg, args.camera)
    engine = RuleEngine(cfg, args.camera, zones, (w, h))
    health = HealthMonitor(cfg, args.camera, cam_name)
    det = Detector(args.weights, conf=args.conf or cam.get("conf", 0.35), imgsz=cam.get("imgsz", 640),
                   device=args.device)                # night / small cameras can use a lower conf + bigger imgsz
    if args.no_blur:
        cfg.setdefault("privacy", {})["blur"] = False
    privacy = PrivacyFilter(cfg)
    if args.agent:
        acfg = cfg.setdefault("agent", {}) or {}
        cfg["agent"] = acfg
        acfg["enabled"] = args.agent != "off"
        if args.agent != "off":
            acfg["provider"] = args.agent
    trace_log = open(out_dir / "agent_trace.jsonl", "a", encoding="utf-8")

    def on_decision(eid, ev, d):
        con = store.connect()
        status = store.set_agent(con, eid, d)
        con.close()
        trace_log.write(json.dumps({"id": eid, "rule": ev["rule"], "camera": ev["camera_id"], **d}, default=str) + "\n")
        trace_log.flush()
        steps = [x["name"] for x in d["trace"] if x.get("type") == "tool_call"] or \
                [x["name"] for x in d["trace"] if x.get("type") == "tool_result"]
        print(f"   [agent {eid}] {' > '.join(steps) or '-'} => {d['action']} {d.get('priority', '')} "
              f"({d['decided_by']}, {d.get('model')}, {d['ms']} ms) -> {status}: {d['summary'][:110]}")

    agent = AgentRunner(cfg, det, store.DB_PATH, fps, (w, h), on_decision=on_decision)
    first_seen = {}
    print(f"SENTRA running on '{cam_name}' | {len(zones)} zones | device: {det.device}")
    print(f"Privacy: face + head blur {'ON' if privacy.enabled else 'OFF (evaluation mode)'}")
    if agent.enabled:
        a = agent.agent.cfg
        print(f"Agent: {a['provider']} | model {a['model_id'] if a['provider'] == 'bedrock' else 'local policy'} "
              f"| re-check window {a['recheck_seconds']}s")
    else:
        print("Agent: OFF (alerts go straight to the guard)")
    if engine.away_houses:
        print(f"Away Mode ON for: {', '.join(sorted(engine.away_houses))}")

    state = dict(n=0, banner=None, banner_until=-1.0)
    alerted = set()

    def handle(events, frame, tracks, t, frame_no):
        for ev in events:
            state["n"] += 1
            if ev.track_id >= 0:
                alerted.add(ev.track_id)
            snap = out_dir / "snapshots" / f"{args.camera}_{ev.rule}_{ev.track_id}_{frame_no}.jpg"
            img = frame.copy()
            if tracks is not None:
                draw_tracks(img, tracks, engine.dwell, alerted)
            cv2.imwrite(str(snap), img)
            ev.snapshot = str(snap).replace("\\", "/")
            to_agent = agent.enabled and ev.rule in AGENT_RULES and tracks is not None
            if to_agent:
                ev.status = "REVIEWING"                 # the agent decides if a human must look
            eid = store.add_event(db, ev.to_dict())
            if to_agent:
                tr = next((x for x in tracks if x.track_id == ev.track_id), None)
                zone = next((z for z in zones if z.id == ev.zone_id), None)
                agent.open(eid, ev.to_dict(), zone, {
                    "box": tr.box if tr else None,
                    "dwell": round(engine.dwell.get((ev.track_id, ev.zone_id), 0), 1),
                    "tracked_for": round(t - first_seen.get(ev.track_id, t), 1),
                    "people": sum(x.is_person for x in tracks),
                    "vehicles": sum(not x.is_person for x in tracks),
                    "away": bool(zone and zone.house and str(zone.house) in engine.away_houses),
                    "live": live,
                })
            clip_path = f"outputs/clips/{args.camera}_{eid}.mp4"
            clips.start(clip_path, on_done=lambda p, eid=eid: store.set_clip(store.connect(), eid, p))
            events_log.write(json.dumps({"id": eid, **ev.to_dict()}) + "\n"); events_log.flush()
            print(f"[{t:6.1f}s] {'CASE ' if to_agent else 'ALERT'} {ev.severity:<6} {ev.title}: {ev.message}"
                  + ("  (agent reviewing)" if to_agent else ""))
            state["banner"], state["banner_until"] = ev, t + 4

    frame_no, t0 = 0, time.time()
    last_live, last_away = [0.0], [0.0]

    def emit(img, t):
        """Send a finished frame everywhere: output video, clip buffer, live console, preview."""
        writer.write(img)
        clips.push(img)
        now = time.time()
        if now - last_live[0] >= 0.25:                  # live picture for the guard console (4 per second)
            last_live[0] = now
            tmp = live_path + ".tmp.jpg"
            cv2.imwrite(tmp, cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1]))))
            try:
                Path(tmp).replace(live_path)
            except OSError:
                pass                                    # console was reading it; next frame will update
            store.update_camera(db, args.camera, cam_name, health.status, live_path.replace("outputs/", ""),
                                {"t": round(t, 1), "zones": len(zones), "alerts": state["n"]})
        if args.realtime and not live:                  # pace video files like a real camera
            wait = t0 + t - time.time()
            if wait > 0:
                time.sleep(wait)
        if args.show:
            cv2.imshow("SENTRA", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    while True:
        ok, frame = cap.read()
        if not ok and not live:
            if args.loop:                               # demo mode: play the clip again, time keeps going
                cap.release(); cap = open_source(src)
                ok, frame = cap.read()
            if not ok:
                break                                   # end of a video file
        t = (frame_no + 1) / fps if not live else time.time() - t0

        # ---- feed lost (real for live cameras, simulated for video files) ----
        lost = not ok or (offline_window and offline_window[0] <= t < offline_window[1])
        if lost:
            frame_no += 1 if ok else 0
            black = np.zeros((h, w, 3), np.uint8)
            handle(health.no_frame(t), draw_no_signal(black.copy()), None, t, frame_no)
            draw_no_signal(black)
            draw_header(black, f"SENTRA  |  {cam_name}  |  t={t:5.1f}s  |  alerts {state['n']}")
            draw_status(black, health.status)
            if state["banner"] and t <= state["banner_until"]:
                draw_alert_banner(black, state["banner"])
            if not emit(black, t):
                break
            if not ok and live:                         # try to reconnect a live camera
                time.sleep(1.0); cap.release(); cap = open_source(src) or cap
            continue
        frame_no += 1

        # ---- camera health (Step 3) ----
        new, restored = health.update(frame, t)
        for name in restored:
            print(f"[{t:6.1f}s] RESTORED {name}")
        tracks = det.track(frame)
        for x in tracks:
            first_seen.setdefault(x.track_id, t)
        agent.push(frame, tracks)                       # raw frame, kept in memory for the agent's re-check
        privacy.apply(frame, tracks)                    # blur faces/heads before anything is saved or shown
        handle(new, frame, tracks, t, frame_no)

        # ---- zone rules (Step 2) + Away Mode from Telegram (Step 5) ----
        if time.time() - last_away[0] >= 1.0:
            last_away[0] = time.time()
            before = set(engine.away_houses)
            engine.away_houses = engine.config_away | set(store.away_houses(db))
            if engine.away_houses != before:
                print(f"[{t:6.1f}s] Away Mode now ON for: {', '.join(sorted(engine.away_houses)) or 'none'}")
        handle(engine.update(tracks, t), frame, tracks, t, frame_no)

        draw_zones(frame, zones, engine.away_houses)
        draw_tracks(frame, tracks, engine.dwell, alerted)
        draw_header(frame, f"SENTRA  |  {cam_name}  |  t={t:5.1f}s  |  alerts {state['n']}")
        draw_status(frame, health.status)
        if state["banner"] and t <= state["banner_until"]:
            draw_alert_banner(frame, state["banner"])
        if not emit(frame, t):
            break

    agent.close()                                       # finish open agent cases
    trace_log.close()
    for th in clips.flush():                            # save clips that were still recording
        th.join()
    time.sleep(0.5)
    store.update_camera(db, args.camera, cam_name, "OFFLINE", live_path.replace("outputs/", ""), {"ended": True})
    cap.release(); writer.release(); events_log.close(); cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"\nDone: {frame_no} frames, {frame_no / max(dt, 1e-6):.1f} FPS, {state['n']} alerts. Video: {out_video}")
    print(f"Privacy: {privacy.faces_blurred} face and {privacy.heads_blurred} head regions blurred")
    print("Open the guard console to review alerts:  python scripts/run_console.py")


if __name__ == "__main__":
    main()
