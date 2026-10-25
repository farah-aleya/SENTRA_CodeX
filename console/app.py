"""SENTRA guard console (web).

  python scripts/run_console.py        then open http://127.0.0.1:8000

The guard sees live camera tiles and the alert queue, plays each alert's
10-second clip, and presses Confirm or Dismiss. Only confirmed alerts go on
to the resident (Step 5).
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import base64
import hashlib
import hmac

import yaml
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentra import store  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
app = FastAPI(title="SENTRA Guard Console")
app.mount("/media", StaticFiles(directory=str(OUT)), name="media")

# ---------------- password (needed when the console is opened from other devices) ----------------
AUTH_FILE = ROOT / "config" / "console.yaml"


def hash_password(pw: str, salt: bytes = None) -> dict:
    salt = salt or hashlib.sha256(str(time.time()).encode()).digest()[:16]
    return {"salt": salt.hex(), "hash": hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000).hex()}


_cache = {"mtime": None, "auth": None}
_OK_TOKENS = set()                       # remember good logins so each request is not re-hashed


def _auth_settings():
    """Read the password hash, re-reading only when config/console.yaml changes."""
    mtime = AUTH_FILE.stat().st_mtime if AUTH_FILE.exists() else None
    if mtime != _cache["mtime"]:
        _cache["mtime"] = mtime
        _cache["auth"] = (yaml.safe_load(AUTH_FILE.read_text(encoding="utf-8")) or {}).get("password") if mtime else None
        _OK_TOKENS.clear()
    return _cache["auth"]


@app.middleware("http")
async def require_password(request: Request, call_next):
    _AUTH = _auth_settings()
    if not _AUTH:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header in _OK_TOKENS:
        return await call_next(request)
    if header.startswith("Basic "):
        try:
            _, pw = base64.b64decode(header[6:]).decode().split(":", 1)
            check = hash_password(pw, bytes.fromhex(_AUTH["salt"]))["hash"]
            if hmac.compare_digest(check, _AUTH["hash"]):
                _OK_TOKENS.add(header)
                return await call_next(request)
        except Exception:                # noqa: BLE001
            pass
    return Response("SENTRA guard console: login required", status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="SENTRA Guard Console"'})
db = store.connect(OUT / "sentra.db")

try:                                   # Step 5 adds resident notifications
    from sentra.notify import notify_resident
except Exception:                      # noqa: BLE001
    def notify_resident(event, con=None):
        print(f"[console] (Step 5 not installed) would notify resident about: {event['message']}")
        return False


def _media(path):
    if not path:
        return None
    p = str(path).replace("\\", "/")
    return "/media/" + (p[len("outputs/"):] if p.startswith("outputs/") else p)


def _shape(e):
    e = dict(e)
    e["snapshot_url"] = _media(e.get("snapshot"))
    e["clip_url"] = _media(e.get("clip"))
    if e.get("handled_at"):
        handled = datetime.strptime(e["handled_at"], "%Y-%m-%d %H:%M:%S").timestamp()
        e["response_seconds"] = round(handled - e["created"], 1)
    return e


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/events")
def events(limit: int = 100):
    return [_shape(e) for e in store.list_events(db, limit)]


@app.get("/api/cameras")
def cameras():
    out = []
    for c in store.list_cameras(db):
        c["frame_url"] = _media(c["frame"])
        c["stale"] = time.time() - (c["updated"] or 0) > 5     # no update for 5 s = feed stopped
        out.append(c)
    return out


@app.get("/api/stats")
def stats():
    ev = [_shape(e) for e in store.list_events(db, 1000)]
    handled = [e for e in ev if e["status"] in ("CONFIRMED", "DISMISSED")]
    dismissed = [e for e in handled if e["status"] == "DISMISSED"]
    resp = [e["response_seconds"] for e in handled if "response_seconds" in e]
    return {
        "pending": sum(e["status"] == "PENDING" for e in ev),
        "confirmed": sum(e["status"] == "CONFIRMED" for e in ev),
        "dismissed": len(dismissed),
        "reviewing": sum(e["status"] == "REVIEWING" for e in ev),
        "logged_by_agent": sum(e["status"] == "LOGGED" for e in ev),
        "false_alarm_rate": round(100 * len(dismissed) / len(handled)) if handled else None,
        "avg_response_seconds": round(sum(resp) / len(resp), 1) if resp else None,
        "away_houses": sorted(store.away_houses(db)),
        "residents": len(store.residents(db)),
    }


def _handle(eid, status, note):
    e = store.get_event(db, eid)
    if not e:
        raise HTTPException(404, "Alert not found")
    if e["status"] != "PENDING":
        return _shape(e)
    store.set_status(db, eid, status, note)
    e = store.get_event(db, eid)
    if status == "CONFIRMED":
        sent = notify_resident(e, db)
        if sent:
            db.execute("UPDATE events SET notified=1 WHERE id=?", (eid,)); db.commit()
            e["notified"] = 1
    return _shape(e)


@app.post("/api/events/{eid}/confirm")
def confirm(eid: str, body: dict = Body(default={})):
    return _handle(eid, "CONFIRMED", body.get("note", ""))


@app.post("/api/events/{eid}/dismiss")
def dismiss(eid: str, body: dict = Body(default={})):
    return _handle(eid, "DISMISSED", body.get("note", ""))


@app.post("/api/reset")
def reset():
    store.clear_all(db)
    return {"ok": True}
