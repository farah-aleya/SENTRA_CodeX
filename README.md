# SENTRA
**Security Enhancement for Network Township Residential AI**

Turning a township's existing CCTV into a guard that never gets tired.

Team Codex, Universiti Teknologi PETRONAS. Sime Darby Property University Hackathon 2026 (Township of Tomorrow, PEOPLE theme).

## Try it now
| What | Link |
|---|---|
| Prototype website (guard console + camera replay) | https://sentra-codex.vercel.app |
| Resident Telegram bot (online 24/7) | [@SENTRA_Alerts_SDP_bot](https://t.me/SENTRA_Alerts_SDP_bot) |

**Telegram demo in 4 steps**
1. Open [@SENTRA_Alerts_SDP_bot](https://t.me/SENTRA_Alerts_SDP_bot) and press **Start**
2. Send `/register 12` to link your chat to House 12
3. Send `/away_on` to turn on Away Mode
4. Tap **Simulate an intruder at House 12** and receive a verified alert with the real 10-second SENTRA clip

## What SENTRA does
- **Behaviour rules on existing CCTV:** perimeter breach, loitering, vehicle dwell and Away Mode breach
- **Guard verification:** every alert comes with a 10-second clip, and the guard confirms or dismisses it
- **Resident alerts:** only confirmed alerts reach residents (Telegram in the pilot, the Sime Darby Property app in Phase 2)
- **Away Mode:** residents mark their house as away, and any person at that house becomes a high-priority alert
- **Camera health:** detects cameras that go offline, frozen or blocked
- **No facial recognition:** SENTRA only detects "person" or "vehicle" and tracks movement

## Team Codex
| Name | Role |
|---|---|
| Anuar Afiq Bin Arfahairy | Team Leader, Project Management |
| Nur Farah Aleya Binti Mohamad Noor'Azlan | AI and Computer Vision |
| Siti Zafirah Binti Zailan | Product and User Experience |
| Ilhan Zikri Bin Zul Azlin | Research and Business |

## Build progress
- [x] Step 1: Project setup + person/vehicle detection and tracking (YOLOv8 + ByteTrack)
- [x] Step 2: Zones + rule engine (perimeter breach, loitering, vehicle dwell, Away Mode)
- [x] Step 3: Camera health check (offline, frozen, blocked)
- [x] Step 4: Event clips + guard console (confirm / dismiss)
- [x] Step 5: Resident alerts (Telegram) + Away Mode toggle
- [x] Step 6: Showcase footage + public prototype website (`web/`)
- [x] Step 7: Always-online Telegram demo (`web/api/telegram.js`)

## Tech stack
Python, Ultralytics YOLOv8n (COCO pretrained), ByteTrack, OpenCV, PyTorch (CUDA), FastAPI, SQLite, Telegram Bot API, Vercel.

## Setup
**Windows + NVIDIA GPU:** double-click `setup_windows.bat` (one time). It should end with `CUDA available: True`.

**Other systems (Python 3.10+):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Every new terminal on Windows: `.venv\Scripts\activate`
The model file `models/yolov8n.pt` is included. If missing, Ultralytics downloads it on first run.

## Step 1: Detection + tracking
```bash
python scripts/make_test_video.py                              # quick synthetic test clip
python scripts/run_detect.py --source samples/test_walk.mp4    # writes outputs/step1_detect.mp4
python scripts/run_detect.py --source 0 --show                 # live laptop webcam, press q to quit
python scripts/run_detect.py --source your_video.mp4 --show    # your own recorded footage
```
Each person and vehicle gets a box and a stable ID (`person #3`). The dot at the bottom of the box is the "foot point" used in Step 2 to check if someone is inside a zone.

## Step 2: Zones + rules
| Rule | Fires when | Severity |
|---|---|---|
| Perimeter breach | a person is inside a `restricted` zone (e.g. climbed over the fence) | HIGH |
| Away Mode breach | a person is at a `house` zone whose house number is in `away_mode` | HIGH |
| Loitering | a person stays in one `house` zone longer than `loiter_seconds` | MEDIUM |
| Vehicle dwell | a vehicle stays in a `road` zone longer than `vehicle_seconds` | MEDIUM |

```bash
# 1. Test on the demo clip (all four rules fire)
python scripts/run_sentra.py --camera demo --config config/demo.yaml --show

# 2. Draw zones on your own footage, then run it
python scripts/draw_zones.py --camera gate01 --source samples/my_gate_clip.mp4 --name "CAM 01 Front Gate"
python scripts/run_sentra.py --camera gate01 --show
```
Outputs: `outputs/<camera>_annotated.mp4`, `outputs/events.jsonl`, `outputs/snapshots/`.
Thresholds and Away Mode are in `config/site.yaml`. Lower `loiter_seconds` and `vehicle_seconds` for short demo clips.

## Step 3: Camera health
| Check | Fires when | Severity |
|---|---|---|
| Camera offline | no video received for `offline_seconds` | HIGH |
| Camera blocked | lens covered or sprayed: picture loses its detail, or goes almost black | HIGH |
| Camera frozen | picture does not change at all for `frozen_seconds` | MEDIUM |

The first `baseline_seconds` are used to learn what the normal picture looks like. Each check sends a "RESTORED" message when the camera recovers. The status badge (top right) shows `CAMERA OK / FROZEN / BLOCKED / NO SIGNAL / OFFLINE`.

```bash
python scripts/make_health_test.py
python scripts/run_sentra.py --camera healthtest --config config/demo.yaml --simulate-offline 22-29 --show
```
Expected: frozen at about 9s, blocked at 16s, offline at 27s, each followed by RESTORED.
Live cameras reconnect automatically when the stream drops.

## Step 4: Guard console
Every alert is saved with a snapshot and a **10-second clip** (4 s before, 6 s after) in `outputs/clips/`.
The guard console is a web page that shows live camera tiles, the alert queue and the clip,
with **Confirm** and **Dismiss** buttons. It also tracks the pilot KPIs: false alarm rate and average response time.

```bash
# Terminal 1: the console (opens http://127.0.0.1:8000)
python scripts/run_console.py

# Terminal 2: a camera, played at normal speed and looped so it looks live
python scripts/run_sentra.py --camera demo --config config/demo.yaml --realtime --loop --show
```
Or double-click `demo_live.bat` to start both. Shortcuts in the console: `C` confirm, `D` dismiss, arrow keys move in the queue.

## Step 5: Resident alerts on Telegram
Only alerts **confirmed by the guard** reach residents, with the 10-second clip.
Residents control Away Mode from Telegram, and SENTRA picks it up within a second.

| Command | What it does |
|---|---|
| `/register 12` | link this chat to House 12 |
| `/away_on` or `/away_on 3` | Away Mode on (optionally for 3 days) |
| `/away_off` | Away Mode off |
| `/status` | house, Away Mode and recent alerts |
| `/manager` | (team) receive camera health alerts in this chat |

Who receives a confirmed alert: house alerts go to that house's residents, street and vehicle alerts go to all residents, camera health alerts go to `/manager` chats.

```bash
python scripts/setup_telegram.py     # one time: paste the token from @BotFather (saved locally, git-ignored)
python scripts/run_bot.py            # keep running so residents can use the commands
```
`demo_live.bat` starts the console, the bot and a demo camera together.

## Opening the console from other devices
| Command | Who can open it |
|---|---|
| `python scripts/run_console.py` | this laptop only (`http://127.0.0.1:8000`) |
| `python scripts/run_console.py --lan` | devices on the same Wi-Fi (`http://<laptop-ip>:8000`) |
| `python scripts/run_console.py --tunnel` | anyone with the link (`https://xxxx.trycloudflare.com`, free Cloudflare tunnel) |

`--lan` and `--tunnel` require a password (asked the first time, stored hashed in `config/console.yaml`, git-ignored).
Log in with any username, e.g. `guard`. Change it with `python scripts/run_console.py --set-password`.
For `--tunnel`, install Cloudflare's tool once on Windows: `winget install --id Cloudflare.cloudflared`, then open a new terminal.
`demo_public.bat` starts the public console, the bot and a demo camera together. The link changes every time it starts.

## Prototype website (`web/`)
A static page that replays **real SENTRA output** (annotated camera videos, alerts at their original times, 10-second clips) with an interactive guard console and a Telegram preview. It is hosted free on Vercel.

```bash
# 1. run SENTRA on each showcase camera
for cam in gate porch door street lobby; do python scripts/run_sentra.py --camera $cam --config config/showcase.yaml; done
# 2. export videos, clips and alerts into web/
python scripts/export_showcase.py --config config/showcase.yaml
# 3. preview locally
cd web && python -m http.server 8090
```
Deploy: Vercel > Add New Project > import this repo > Root Directory `web` > Framework `Other` > Deploy.

## Always-online Telegram demo (`web/api/telegram.js`)
The resident bot [@SENTRA_Alerts_SDP_bot](https://t.me/SENTRA_Alerts_SDP_bot) runs as a Vercel function, so anyone can try the resident side of SENTRA without the on-site system running.
It replies to `/start`, `/register`, `/away_on`, `/away_off`, `/status` and `/demo_alert`. After `/away_on`, the **Simulate an intruder** button sends a verified alert with a real SENTRA Away Mode clip.
It does not store any data. Every user gets the same demo house (House 12).

Setup (one time):
1. Vercel > Settings > Environment Variables: add `TELEGRAM_BOT_TOKEN` (the bot token from @BotFather) and `TELEGRAM_SECRET` (any random word), then redeploy
2. Open `https://<your-site>/api/telegram?setup=<TELEGRAM_SECRET>` once in a browser

The same bot is used by the on-site system (`scripts/run_bot.py`). Only one can receive messages at a time: starting `run_bot.py` pauses the online demo, and opening the setup link again turns it back on.

## Privacy
- SENTRA does not use facial recognition. It only detects the class "person" or "vehicle" and tracks movement.
- Clips are only shown to the guard on duty, and only guard-confirmed alerts reach residents.
- Bot tokens and the console password are stored locally or in Vercel settings, and are never committed to this repo.

## Data
- Demo scenes are recorded by our team acting out scenarios (synthetic data).
- Free stock clips from Mixkit, Pexels and Pixabay are used under their free licences.
- Public research datasets (UCF-Crime, VIRAT) are used for testing only. They are not redistributed in this repo.
- The showcase in `web/` uses short, publicly available CCTV clips from online news reports and CCTV product demonstrations, processed by SENTRA to demonstrate detection. All rights remain with their owners; source videos are not stored in this repo.
- No data from real township residents is used.

## Disclaimer
SENTRA is a student hackathon prototype by Team Codex. It is not an official Sime Darby Property product.
