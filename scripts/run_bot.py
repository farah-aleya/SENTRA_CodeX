"""STEP 5 - Run the SENTRA Telegram bot (resident side).

  python scripts/run_bot.py

Residents message the bot:
  /start               how it works
  /register 12         link this chat to House 12
  /away_on [days]      Away Mode on (optionally for N days)
  /away_off            Away Mode off
  /status              house, Away Mode and recent alerts
  /unregister          stop alerts
  /manager             (team only) receive camera health alerts in this chat
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentra import store  # noqa: E402
from sentra.notify import CONFIG, call, load_settings  # noqa: E402

HELP = ("Welcome to SENTRA, your township's AI night guard.\n\n"
        "1. Link your house:  /register 12\n"
        "2. Going away? Turn on Away Mode:  /away_on  or  /away_on 3  (for 3 days)\n"
        "3. Back home:  /away_off\n"
        "4. Check your status:  /status\n\n"
        "You only receive alerts after a guard has checked the video and confirmed it. "
        "SENTRA does not use facial recognition.")


def reply(chat_id, text):
    call("sendMessage", chat_id=chat_id, text=text)


def handle(con, msg):
    chat_id = msg["chat"]["id"]
    name = msg["from"].get("first_name", "Resident")
    text = (msg.get("text") or "").strip()
    if not text.startswith("/"):
        return reply(chat_id, "Send /start to see what I can do.")
    cmd, *args = text.split()
    cmd = cmd.split("@")[0].lower()
    me = store.resident(con, chat_id)

    if cmd == "/start":
        return reply(chat_id, HELP)
    if cmd == "/register":
        if not args:
            return reply(chat_id, "Tell me your house number, e.g. /register 12")
        house = args[0].lstrip("#").upper()
        store.register_resident(con, chat_id, house, name)
        return reply(chat_id, f"Done, {name}. This chat is linked to House {house}. "
                              f"You will get verified alerts for your house and community alerts.")
    if cmd == "/unregister":
        store.unregister_resident(con, chat_id)
        return reply(chat_id, "You will no longer receive SENTRA alerts. Send /register to join again.")
    if cmd == "/manager":
        s = load_settings()
        ids = set(int(x) for x in s.get("management_chat_ids") or [])
        ids.add(chat_id)
        s["management_chat_ids"] = sorted(ids)
        CONFIG.write_text(yaml.safe_dump(s, sort_keys=False), encoding="utf-8")
        return reply(chat_id, "This chat will receive camera health alerts (offline, blocked, frozen).")
    if not me:
        return reply(chat_id, "Please link your house first, e.g. /register 12")

    house = me["house"]
    if cmd == "/away_on":
        until = ""
        if args and args[0].isdigit():
            until = (date.today() + timedelta(days=int(args[0]))).isoformat()
        store.set_away(con, house, True, until, name)
        return reply(chat_id, f"Away Mode is ON for House {house}" + (f" until {until}." if until else ".")
                     + " Anyone detected at your house will be flagged as HIGH priority to the guard right away.")
    if cmd == "/away_off":
        store.set_away(con, house, False)
        return reply(chat_id, f"Welcome home. Away Mode is OFF for House {house}.")
    if cmd == "/status":
        away = store.away_houses(con)
        state = ("ON" + (f" until {away[house]}" if away.get(house) else "")) if house in away else "OFF"
        recent = [e for e in store.list_events(con, 200) if e.get("house") == house][:3]
        lines = [f"House {house}", f"Away Mode: {state}", "", "Recent alerts at your house:"]
        lines += [f"- {e['wall_time']}  {e['title']} ({e['status'].lower()})" for e in recent] or ["- none"]
        return reply(chat_id, "\n".join(lines))
    return reply(chat_id, "I don't know that command. Send /start for help.")


def main():
    s = load_settings()
    if not s.get("bot_token"):
        sys.exit("Telegram is not set up yet. Run: python scripts/setup_telegram.py")
    con = store.connect(ROOT / "outputs" / "sentra.db")
    print(f"SENTRA bot @{s.get('bot_username', '?')} is running. Message it /start. (Ctrl+C to stop)")
    offset = None
    while True:
        try:
            params = {"timeout": 25, **({"offset": offset} if offset else {})}   # long polling
            updates = call("getUpdates", http_timeout=35, **params)
        except Exception as e:  # noqa: BLE001
            print(f"[bot] {e}; retrying in 5 s"); time.sleep(5); continue
        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("edited_message")
            if not msg:
                continue
            print(f"[bot] {msg['chat']['id']}: {msg.get('text', '')}")
            try:
                handle(con, msg)
            except Exception as e:  # noqa: BLE001
                print(f"[bot] error: {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
