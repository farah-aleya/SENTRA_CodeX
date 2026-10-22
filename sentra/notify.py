"""SENTRA resident alerts via Telegram (Step 5).

Only alerts CONFIRMED by the guard are sent. Who gets them:
  - alert linked to a house (loitering, Away Mode breach, a house's yard)  -> residents of that house
  - other alerts (street, vehicle)                                          -> all registered residents
  - camera health alerts                                                    -> management chats (or everyone if none set)

The bot token lives only on your laptop in config/telegram.yaml (never commit it).
"""
import time
from pathlib import Path

import requests
import yaml

from . import store

CONFIG = Path(__file__).resolve().parents[1] / "config" / "telegram.yaml"
API = "https://api.telegram.org/bot{token}/{method}"
EMOJI = {"HIGH": "\U0001F6A8", "MEDIUM": "⚠️"}


def load_settings() -> dict:
    if not CONFIG.exists():
        return {}
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def call(method: str, token: str = None, files=None, http_timeout=30, **params):
    token = token or load_settings().get("bot_token")
    if not token:
        raise RuntimeError("No bot token. Run: python scripts/setup_telegram.py")
    r = requests.post(API.format(token=token, method=method), data=params, files=files, timeout=http_timeout)
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data.get('description')}")
    return data["result"]


def _recipients(con, event, settings):
    if event["rule"].startswith("CAMERA_"):
        mgmt = [int(c) for c in settings.get("management_chat_ids") or []]
        return mgmt or [r["chat_id"] for r in store.residents(con)]
    if event.get("house"):
        return [r["chat_id"] for r in store.residents(con, event["house"])]
    return [r["chat_id"] for r in store.residents(con)]


def _caption(event):
    when = time.strftime("%I:%M %p", time.localtime(event["created"]))
    lines = [f"{EMOJI.get(event['severity'], '')} SENTRA VERIFIED ALERT",
             f"{event['title']}: {event['message']}",
             f"{event['camera_name']}, {when}",
             "Confirmed by the guard on duty."]
    if event.get("guard_note"):
        lines.append(f"Guard note: {event['guard_note']}")
    if event["rule"] == "AWAY_BREACH":
        lines.append("Your house is in Away Mode. Reply /away_off when you are back home.")
    elif not event["rule"].startswith("CAMERA_"):
        lines.append("Stay safe. Call the guardhouse or 999 if you are in danger.")
    return "\n".join(lines)


def notify_resident(event: dict, con=None) -> bool:
    """Send a confirmed alert. Returns True if at least one message was delivered."""
    settings = load_settings()
    if not settings.get("bot_token"):
        print("[notify] Telegram not set up yet. Run: python scripts/setup_telegram.py")
        return False
    con = con or store.connect()
    chats = _recipients(con, event, settings)
    if not chats:
        print(f"[notify] No registered resident for this alert (house '{event.get('house') or '-'}').")
        return False
    root = Path(__file__).resolve().parents[1]
    caption = _caption(event)
    sent = 0
    for chat in chats:
        try:
            clip = root / event["clip"] if event.get("clip") else None
            snap = root / event["snapshot"] if event.get("snapshot") else None
            if clip and clip.exists():
                with open(clip, "rb") as f:
                    call("sendVideo", files={"video": f}, chat_id=chat, caption=caption, supports_streaming="true")
            elif snap and snap.exists():
                with open(snap, "rb") as f:
                    call("sendPhoto", files={"photo": f}, chat_id=chat, caption=caption)
            else:
                call("sendMessage", chat_id=chat, text=caption)
            sent += 1
        except Exception as e:                       # noqa: BLE001
            print(f"[notify] Could not send to {chat}: {e}")
    print(f"[notify] Alert '{event['title']}' sent to {sent} chat(s).")
    return sent > 0
