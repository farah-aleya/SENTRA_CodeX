"""Start the SENTRA web app (guards and residents).

  python scripts/run_console.py                 this laptop only   (http://127.0.0.1:8000)
  python scripts/run_console.py --lan           phones/laptops on the same Wi-Fi
  python scripts/run_console.py --tunnel        public https link from anywhere (Cloudflare, free)
  python scripts/run_console.py --host 0.0.0.0 --port 80 --no-browser     (server / EC2)

Everyone signs in with their own account (Create account on the first page).
Guards need the guard invite code: set SENTRA_GUARD_CODE, default "codex-guard".

Run a camera in a second terminal so cases appear live:
  python scripts/run_sentra.py --camera gate --config config/showcase.yaml --realtime --loop
"""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


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
                print("\n" + "=" * 64 + f"\n  PUBLIC LINK:  {url}\n  (works from any phone or laptop; sign in with your account)\n"
                      + "=" * 64 + "\n", flush=True)
    threading.Thread(target=watch, daemon=True).start()
    return proc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan", action="store_true", help="allow devices on the same Wi-Fi")
    ap.add_argument("--tunnel", action="store_true", help="public https link via Cloudflare")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--host", default=None, help="bind address, e.g. 0.0.0.0 on a server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SENTRA_PORT", 8000)))
    args = ap.parse_args()

    from sentra.auth import guard_code
    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    local = f"http://127.0.0.1:{args.port}"
    print(f"SENTRA web app on this machine: {local}   (Ctrl+C to stop)")
    code = guard_code()
    print(f"Guard invite code: {code if code == 'codex-guard' else '(set from SENTRA_GUARD_CODE)'}"
          + ("   <- change it with the SENTRA_GUARD_CODE environment variable before going public"
             if code == "codex-guard" and (args.tunnel or args.lan or args.host) else ""))
    if args.lan:
        print(f"Same Wi-Fi devices: http://{lan_ip()}:{args.port}   (if Windows asks, allow on Private networks)")
    tunnel = start_tunnel(args.port) if args.tunnel else None
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(local)).start()
    try:
        uvicorn.run("console.app:app", host=host, port=args.port, log_level="warning",
                    proxy_headers=True, forwarded_allow_ips="127.0.0.1")     # behind Caddy on EC2: real https scheme
    finally:
        if tunnel:
            tunnel.terminate()
