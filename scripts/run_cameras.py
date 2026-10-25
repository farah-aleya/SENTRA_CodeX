"""Start several SENTRA cameras at once (one process each), looped like live CCTV.

  python scripts/run_cameras.py                                  all cameras in config/showcase.yaml
  python scripts/run_cameras.py --cameras gate porch door        only these
  python scripts/run_cameras.py --agent local                    offline agent policy (no AWS)

Ctrl+C stops all of them. Each camera uses CPU, so 3 at once is a good limit on a laptop.
"""
import argparse
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentra.zones import load_config  # noqa: E402


def pipe(proc, name):
    for line in proc.stdout:
        line = line.rstrip()
        if line and ("CASE" in line or "ALERT" in line or "[agent" in line or "RESTORED" in line
                     or "Error" in line or "Traceback" in line or "running on" in line):
            print(f"[{name:>6}] {line.strip()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/showcase.yaml")
    ap.add_argument("--cameras", nargs="*", help="camera ids (default: all in the config)")
    ap.add_argument("--agent", choices=["bedrock", "local", "off"], default=None)
    args = ap.parse_args()

    cams = args.cameras or list((load_config(args.config).get("cameras") or {}).keys())
    procs = []
    for cam in cams:
        cmd = [sys.executable, "scripts/run_sentra.py", "--camera", cam, "--config", args.config, "--realtime", "--loop"]
        if args.agent:
            cmd += ["--agent", args.agent]
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=pipe, args=(p, cam), daemon=True).start()
        procs.append(p)
    print(f"Started {len(procs)} cameras: {', '.join(cams)}   (Ctrl+C to stop all)")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("Stopping cameras...")
        for p in procs:
            p.terminate()
