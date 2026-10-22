"""SENTRA event store (SQLite).

Shared between the detector (writes alerts and camera status) and the guard
console (reads alerts, sets CONFIRMED / DISMISSED). One small file, no server.
"""
import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path("outputs/sentra.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    created REAL,
    rule TEXT, severity TEXT, title TEXT, message TEXT,
    camera_id TEXT, camera_name TEXT, zone_id TEXT, zone_name TEXT,
    track_id INTEGER, label TEXT, video_time REAL, wall_time TEXT,
    snapshot TEXT, clip TEXT,
    status TEXT DEFAULT 'PENDING',
    handled_at TEXT, guard_note TEXT, notified INTEGER DEFAULT 0,
    house TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS residents (
    chat_id INTEGER PRIMARY KEY, house TEXT, name TEXT, registered TEXT
);
CREATE TABLE IF NOT EXISTS away_mode (
    house TEXT PRIMARY KEY, since TEXT, until TEXT, set_by TEXT
);
CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    name TEXT, status TEXT, frame TEXT, updated REAL, info TEXT
);
"""


def connect(path=DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
    if "house" not in cols:                       # upgrade a Step 4 database
        con.execute("ALTER TABLE events ADD COLUMN house TEXT DEFAULT ''"); con.commit()
    return con


def add_event(con, ev: dict) -> str:
    eid = uuid.uuid4().hex[:8]
    cols = ["rule", "severity", "title", "message", "camera_id", "camera_name", "zone_id", "zone_name",
            "track_id", "label", "video_time", "wall_time", "snapshot", "clip", "house"]
    con.execute(f"INSERT INTO events (id, created, {', '.join(cols)}) VALUES (?, ?, {', '.join('?' * len(cols))})",
                [eid, time.time(), *[ev.get(c) for c in cols]])
    con.commit()
    return eid


def set_clip(con, eid: str, clip: str):
    con.execute("UPDATE events SET clip=? WHERE id=?", (clip, eid))
    con.commit()


def set_status(con, eid: str, status: str, note: str = ""):
    con.execute("UPDATE events SET status=?, handled_at=?, guard_note=? WHERE id=?",
                (status, time.strftime("%Y-%m-%d %H:%M:%S"), note, eid))
    con.commit()


def list_events(con, limit=100):
    return [dict(r) for r in con.execute("SELECT * FROM events ORDER BY created DESC LIMIT ?", (limit,))]


def get_event(con, eid):
    r = con.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
    return dict(r) if r else None


def update_camera(con, camera_id, name, status, frame, info=None):
    con.execute("INSERT INTO cameras (camera_id, name, status, frame, updated, info) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(camera_id) DO UPDATE SET name=excluded.name, status=excluded.status, "
                "frame=excluded.frame, updated=excluded.updated, info=excluded.info",
                (camera_id, name, status, frame, time.time(), json.dumps(info or {})))
    con.commit()


def list_cameras(con):
    return [dict(r) for r in con.execute("SELECT * FROM cameras ORDER BY camera_id")]


# ---------------- residents and Away Mode (Step 5) ----------------
def register_resident(con, chat_id: int, house: str, name: str):
    con.execute("INSERT INTO residents (chat_id, house, name, registered) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(chat_id) DO UPDATE SET house=excluded.house, name=excluded.name",
                (chat_id, str(house), name, time.strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()


def unregister_resident(con, chat_id: int):
    con.execute("DELETE FROM residents WHERE chat_id=?", (chat_id,)); con.commit()


def resident(con, chat_id: int):
    r = con.execute("SELECT * FROM residents WHERE chat_id=?", (chat_id,)).fetchone()
    return dict(r) if r else None


def residents(con, house=None):
    if house is None:
        return [dict(r) for r in con.execute("SELECT * FROM residents")]
    return [dict(r) for r in con.execute("SELECT * FROM residents WHERE house=?", (str(house),))]


def set_away(con, house: str, on: bool, until: str = "", set_by: str = ""):
    if on:
        con.execute("INSERT INTO away_mode (house, since, until, set_by) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(house) DO UPDATE SET since=excluded.since, until=excluded.until, set_by=excluded.set_by",
                    (str(house), time.strftime("%Y-%m-%d %H:%M:%S"), until, set_by))
    else:
        con.execute("DELETE FROM away_mode WHERE house=?", (str(house),))
    con.commit()


def away_houses(con) -> dict:
    """Houses in Away Mode -> until date ('' = until turned off). Expired entries are removed."""
    today = time.strftime("%Y-%m-%d")
    out = {}
    for r in con.execute("SELECT house, until FROM away_mode").fetchall():
        if r["until"] and r["until"] < today:
            con.execute("DELETE FROM away_mode WHERE house=?", (r["house"],))
        else:
            out[r["house"]] = r["until"]
    con.commit()
    return out


def clear_all(con):
    con.execute("DELETE FROM events"); con.execute("DELETE FROM cameras"); con.commit()
