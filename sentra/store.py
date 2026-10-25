"""SENTRA event store (SQLite).

Shared between the detector (writes cases and camera status), the agent (writes its
decision) and the web app (accounts, guard decisions, Away Mode). One small file, no server.
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
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL, name TEXT, role TEXT NOT NULL, house TEXT DEFAULT '',
    pw_salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created TEXT
);
CREATE TABLE IF NOT EXISTS away_mode (
    house TEXT PRIMARY KEY, since TEXT, until TEXT, set_by TEXT
);
CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    name TEXT, status TEXT, frame TEXT, updated REAL, info TEXT
);
"""


# agent fields: decision (REQUEST_APPROVAL / LOG_ONLY), priority, summary, who decided, model, time, full trace
AGENT_COLS = ["agent_action", "agent_priority", "agent_summary", "agent_decided_by", "agent_model",
              "agent_ms", "agent_trace"]
# event status: REVIEWING (agent working) -> PENDING (waiting for a human) or LOGGED (no alert)
#               PENDING -> CONFIRMED / DISMISSED by the guard


def connect(path=DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
    if "house" not in cols:                       # upgrade a Step 4 database
        con.execute("ALTER TABLE events ADD COLUMN house TEXT DEFAULT ''"); con.commit()
    for col in AGENT_COLS + ["handled_by"]:       # upgrade for the agent (OpenCV competition)
        if col not in cols:
            con.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT DEFAULT ''")
    if "updated" not in cols:                     # last change time, used by the live stream
        con.execute("ALTER TABLE events ADD COLUMN updated REAL DEFAULT 0")
    con.commit()
    return con


def add_event(con, ev: dict) -> str:
    eid = uuid.uuid4().hex[:8]
    cols = ["rule", "severity", "title", "message", "camera_id", "camera_name", "zone_id", "zone_name",
            "track_id", "label", "video_time", "wall_time", "snapshot", "clip", "house", "status"]
    now = time.time()
    con.execute(f"INSERT INTO events (id, created, updated, {', '.join(cols)}) "
                f"VALUES (?, ?, ?, {', '.join('?' * len(cols))})",
                [eid, now, now, *[ev.get(c) for c in cols]])
    con.commit()
    return eid


def set_agent(con, eid: str, d: dict):
    """Save the agent decision and move the event to PENDING (needs a human) or LOGGED."""
    status = "PENDING" if d.get("action") == "REQUEST_APPROVAL" else "LOGGED"
    con.execute("UPDATE events SET status=?, agent_action=?, agent_priority=?, agent_summary=?, agent_decided_by=?, "
                "agent_model=?, agent_ms=?, agent_trace=?, updated=? WHERE id=? AND status='REVIEWING'",
                (status, d.get("action", ""), d.get("priority", ""), d.get("summary", ""), d.get("decided_by", ""),
                 d.get("model") or "", str(d.get("ms", "")), json.dumps(d.get("trace", []), default=str),
                 time.time(), eid))
    con.commit()
    return status


def set_clip(con, eid: str, clip: str):
    con.execute("UPDATE events SET clip=?, updated=? WHERE id=?", (clip, time.time(), eid))
    con.commit()


def set_status(con, eid: str, status: str, note: str = "", by: str = ""):
    con.execute("UPDATE events SET status=?, handled_at=?, guard_note=?, handled_by=?, updated=? WHERE id=?",
                (status, time.strftime("%Y-%m-%d %H:%M:%S"), note, by, time.time(), eid))
    con.commit()


def list_events(con, limit=100):
    return [dict(r) for r in con.execute("SELECT * FROM events ORDER BY created DESC LIMIT ?", (limit,))]


def events_updated_since(con, ts: float, limit=200):
    return [dict(r) for r in con.execute("SELECT * FROM events WHERE updated>? ORDER BY updated LIMIT ?", (ts, limit))]


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


# ---------------- accounts (web app) ----------------
def add_user(con, username, name, role, house, pw_salt, pw_hash):
    cur = con.execute("INSERT INTO users (username, name, role, house, pw_salt, pw_hash, created) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (username.lower(), name, role, str(house or ""), pw_salt, pw_hash,
                       time.strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    return cur.lastrowid


def user_by_name(con, username):
    r = con.execute("SELECT * FROM users WHERE username=?", ((username or "").lower(),)).fetchone()
    return dict(r) if r else None


def user_by_id(con, uid):
    r = con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(r) if r else None


def count_users(con, role=None, house=None):
    q, args = "SELECT COUNT(*) FROM users WHERE 1=1", []
    if role:
        q += " AND role=?"; args.append(role)
    if house is not None:
        q += " AND house=?"; args.append(str(house))
    return con.execute(q, args).fetchone()[0]


# ---------------- Away Mode ----------------
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
