"""Start the SENTRA guard console.

  python scripts/run_console.py                 this laptop only   (http://127.0.0.1:8000)
  python scripts/run_console.py --lan           phones/laptops on the same Wi-Fi
  python scripts/run_console.py --tunnel        public https link from anywhere (Cloudflare, free)
  python scripts/run_console.py --set-password  set or change the console password

With --lan or --tunnel a password is required (you are asked to create one the first time).
Login: any username (e.g. guard) + your password.

Run the detector in a second terminal so alerts appear live:
  python scripts/run_sentra.py --camera demo --config config/demo.yaml --realtime --loop
"""
import argparse
import getpass
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
AUTH_FILE = ROOT / "config" / "console.yaml"


def set_password():
    from console.app import hash_password
    while True:
        pw = getpass.getpass("Choose a console password (hidden, min 6 characters): ")
        if len(pw) >= 6 and pw == getpass.getpass("Type it again: "):
            break
        print("Passwords did not match or too short. Try again.")
    data = yaml.safe_load(AUTH_FILE.read_text(encoding="utf-8")) if AUTH_FILE.exists() else {}
    data = data or {}
    data["password"] = hash_password(pw)
    AUTH_FILE.write_text(yaml.safe_dump(data), encoding="utf-8")
    print(f"Password saved (hashed) in {AUTH_FILE.relative_to(ROOT)}. Login with any username, e.g. guard.")


def has_password():
    return AUTH_FILE.exists() and (yaml.safe_load(AUTH_FILE.read_text(encoding="utf-8")) or {}).get("password")


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # no data is sent; just picks the Wi-Fi address
        return s.getsockname()[0]
    except OSError:
        return "your-laptop-ip"
    finally:
        s.close()


def find_cloudflared():
    exe = shutil.which("cloudflared")
    for p in [exe, r"C:\Program Files (x86)\cloudflared\cloudflared.exe", r"C:\Program Files\cloudflared\cloudflared.exe",
              str(ROOT / "tools" / "cloudflared.exe")]:
        if p and Path(p).exists():
            return p
    return None


def start_tunnel(port):
    exe = find_cloudflared()
    if not exe:
        sys.exit("cloudflared is not installed. Run this once, then open a NEW terminal:\n"
                 "  winget install --id Cloudflare.cloudflared")
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def watch():
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                url = m.group(0)
                (ROOT / "outputs").mkdir(exist_ok=True)
                (ROOT / "outputs" / "public_url.txt").write_text(url + "\n")
                print("\n" + "=" * 64 + f"\n  PUBLIC LINK:  {url}\n  (works from any phone or laptop; login with your password)\n"
                      + "=" * 64 + "\n", flush=True)
    threading.Thread(target=watch, daemon=True).start()
    return proc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan", action="store_true", help="allow devices on the same Wi-Fi")
    ap.add_argument("--tunnel", action="store_true", help="public https link via Cloudflare")
    ap.add_argument("--set-password", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SENTRA_PORT", 8000)))
    args = ap.parse_args()

    if args.set_password:
        set_password()
        sys.exit(0)
    if (args.lan or args.tunnel) and not has_password():
        print("The console will be reachable from other devices, so it needs a password first.")
        set_password()

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    local = f"http://127.0.0.1:{args.port}"
    print(f"SENTRA guard console on this laptop: {local}   (Ctrl+C to stop)")
    if args.lan:
        print(f"Same Wi-Fi devices: http://{lan_ip()}:{args.port}   (if Windows asks, allow on Private networks)")
    tunnel = start_tunnel(args.port) if args.tunnel else None
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(local)).start()
    try:
        uvicorn.run("console.app:app", host=host, port=args.port, log_level="warning")
    finally:
        if tunnel:
            tunnel.terminate()
