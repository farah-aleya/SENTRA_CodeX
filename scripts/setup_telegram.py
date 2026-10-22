"""STEP 5 - Connect SENTRA to your Telegram bot (one time).

  python scripts/setup_telegram.py

Paste the token from @BotFather when asked (it is hidden while you type/paste).
It is saved only on this laptop in config/telegram.yaml, which is git-ignored.
"""
import getpass
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "telegram.yaml"

COMMANDS = [
    ("start", "How to use SENTRA"),
    ("register", "Link this chat to your house, e.g. /register 12"),
    ("away_on", "Turn on Away Mode, e.g. /away_on 3 (days)"),
    ("away_off", "Turn off Away Mode"),
    ("status", "Your house, Away Mode and recent alerts"),
    ("unregister", "Stop receiving alerts"),
]


def main():
    token = getpass.getpass("Paste your bot token from @BotFather (hidden), then press Enter: ").strip()
    if ":" not in token:
        sys.exit("That does not look like a bot token (it should contain a ':'). Try again.")
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15).json()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not reach Telegram: {e}")
    if not r.get("ok"):
        sys.exit(f"Telegram rejected the token: {r.get('description')}")
    bot = r["result"]

    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    settings = settings or {}
    settings["bot_token"] = token
    settings["bot_username"] = bot["username"]
    settings.setdefault("management_chat_ids", [])
    CONFIG.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    # menu of commands shown in Telegram
    requests.post(f"https://api.telegram.org/bot{token}/setMyCommands",
                  json={"commands": [{"command": c, "description": d} for c, d in COMMANDS]}, timeout=15)
    requests.post(f"https://api.telegram.org/bot{token}/setMyDescription", timeout=15, json={
        "description": "SENTRA sends you verified security alerts from your township's CCTV. "
                       "Only alerts confirmed by the guard reach you. No facial recognition."})
    requests.post(f"https://api.telegram.org/bot{token}/setMyShortDescription", timeout=15,
                  json={"short_description": "Verified township security alerts from SENTRA."})

    print(f"\nConnected to @{bot['username']}. Token saved to {CONFIG.relative_to(ROOT)} (not shared, git-ignored).")
    print("Next: python scripts/run_bot.py   then message your bot /start and /register 12")


if __name__ == "__main__":
    main()
