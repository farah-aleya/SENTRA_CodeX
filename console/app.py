"""SENTRA web app: one site for guards and residents.

  python scripts/run_console.py        then open http://127.0.0.1:8000

Guards
  - see live camera tiles
  - get a pop-up the moment the agent sends a case for review (with clip, agent summary
    and the agent's step-by-step trace), and press Approve alarm or Dismiss
Residents
  - get a pop-up when a guard confirms an alert at their house (or a street-wide alert)
  - turn Away Mode on or off for their house

Live updates use Server-Sent Events (/api/stream). Clips and snapshots are only served to
signed-in users who are allowed to see that alert.
"""
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentra import auth, store  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
COOKIE = "sentra_session"
app = FastAPI(title="SENTRA")
db = store.connect(OUT / "sentra.db")


# ---------------- accounts ----------------
def current_user(request: Request):
    uid = auth.read_token(request.cookies.get(COOKIE, ""))
    user = store.user_by_id(db, uid) if uid else None
    if not user:
        raise HTTPException(401, "Please sign in")
    return user


def guard_only(user=Depends(current_user)):
    if user["role"] != "guard":
        raise HTTPException(403, "Guards only")
    return user


def _public(u):
    return {k: u[k] for k in ("id", "username", "name", "role", "house")}


def _login(resp: Response, request: Request, user):
    resp.set_cookie(COOKIE, auth.make_token(user["id"]), max_age=auth.SESSION_HOURS * 3600, httponly=True,
                    samesite="lax", secure=request.url.scheme == "https")


@app.post("/api/register")
def register(request: Request, response: Response, body: dict = Body(...)):
    username = (body.get("username") or "").strip()
    err = auth.validate_registration(username, body.get("password"), body.get("role"),
                                     body.get("house"), body.get("guard_code"))
    if err:
        raise HTTPException(400, err)
    if store.user_by_name(db, username):
        raise HTTPException(400, "That username is taken.")
    salt, h = auth.hash_password(body["password"])
    house = str(body.get("house") or "").strip() if body["role"] == "resident" else ""
    uid = store.add_user(db, username, (body.get("name") or username).strip()[:60], body["role"], house, salt, h)
    user = store.user_by_id(db, uid)
    _login(response, request, user)
    return _public(user)


@app.post("/api/login")
def login(request: Request, response: Response, body: dict = Body(...)):
    user = store.user_by_name(db, body.get("username"))
    if not user or not auth.check_password(body.get("password") or "", user["pw_salt"], user["pw_hash"]):
        time.sleep(0.5)                                   # slow down password guessing
        raise HTTPException(401, "Wrong username or password.")
    _login(response, request, user)
    return _public(user)


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(current_user)):
    out = _public(user)
    if user["role"] == "resident":
        out["away"] = user["house"] in store.away_houses(db)
        out["away_until"] = store.away_houses(db).get(user["house"], "")
    return out


# ---------------- who may see what ----------------
def visible(e: dict, user) -> bool:
    if user["role"] == "guard":
        return True
    # residents: only guard-confirmed alerts for their house, or street-wide alerts (no house)
    if e["status"] != "CONFIRMED" or e["rule"].startswith("CAMERA_"):
        return False
    return not e.get("house") or str(e["house"]) == str(user["house"])


def _media(path):
    if not path:
        return None
    p = str(path).replace("\\", "/")
    return "/media/" + (p[len("outputs/"):] if p.startswith("outputs/") else p)


def _shape(e, user):
    e = dict(e)
    e["snapshot_url"] = _media(e.get("snapshot"))
    e["clip_url"] = _media(e.get("clip"))
    if e.get("handled_at"):
        handled = datetime.strptime(e["handled_at"], "%Y-%m-%d %H:%M:%S").timestamp()
        e["response_seconds"] = round(handled - e["created"], 1)
    trace = json.loads(e.pop("agent_trace") or "[]") if e.get("agent_trace") else []
    if user["role"] == "guard":
        e["agent_steps"] = [s for s in trace if s.get("type") in ("tool_call", "tool_result", "guardrail", "decision")]
    else:                                                     # residents never see internals or notes
        for k in ("guard_note", "handled_by", "agent_decided_by", "agent_model", "agent_ms", "track_id"):
            e.pop(k, None)
    return e


@app.get("/media/{path:path}")
def media(path: str, user=Depends(current_user)):
    f = (OUT / path).resolve()
    if OUT.resolve() not in f.parents or not f.is_file():
        raise HTTPException(404)
    rel = "outputs/" + path
    if user["role"] != "guard":
        row = db.execute("SELECT * FROM events WHERE clip=? OR snapshot=?", (rel, rel)).fetchone()
        if not row or not visible(dict(row), user):
            raise HTTPException(403)
    return FileResponse(f, headers={"Cache-Control": "private, max-age=60"})


# ---------------- data ----------------
@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/events")
def events(limit: int = 200, user=Depends(current_user)):
    return [_shape(e, user) for e in store.list_events(db, limit) if visible(e, user)]


@app.get("/api/cameras")
def cameras(user=Depends(guard_only)):
    out = []
    for c in store.list_cameras(db):
        info = json.loads(c.get("info") or "{}")
        age = time.time() - (c["updated"] or 0)
        if info.get("ended") or age > 600:          # stopped on purpose, or not seen for 10 min: hide the tile
            continue
        c["frame_url"] = _media(c["frame"]) + f"?t={int((c['updated'] or 0) * 4)}" if c["frame"] else None
        c["stale"] = time.time() - (c["updated"] or 0) > 5     # no update for 5 s = feed stopped
        out.append(c)
    return out


@app.get("/api/stats")
def stats(user=Depends(guard_only)):
    ev = store.list_events(db, 2000)
    handled = [e for e in ev if e["status"] in ("CONFIRMED", "DISMISSED")]
    resp = []
    for e in handled:
        if e.get("handled_at"):
            resp.append(datetime.strptime(e["handled_at"], "%Y-%m-%d %H:%M:%S").timestamp() - e["created"])
    cases = [e for e in ev if e.get("agent_action")]
    return {
        "pending": sum(e["status"] == "PENDING" for e in ev),
        "reviewing": sum(e["status"] == "REVIEWING" for e in ev),
        "confirmed": sum(e["status"] == "CONFIRMED" for e in ev),
        "dismissed": sum(e["status"] == "DISMISSED" for e in ev),
        "logged_by_agent": sum(e["status"] == "LOGGED" for e in ev),
        "agent_cases": len(cases),
        "false_alarm_rate": round(100 * sum(e["status"] == "DISMISSED" for e in handled) / len(handled))
        if handled else None,
        "avg_response_seconds": round(sum(resp) / len(resp), 1) if resp else None,
        "away_houses": sorted(store.away_houses(db)),
        "residents": store.count_users(db, "resident"),
    }


def _decide(eid, status, note, user):
    e = store.get_event(db, eid)
    if not e:
        raise HTTPException(404, "Alert not found")
    if e["status"] == "PENDING":
        store.set_status(db, eid, status, (note or "")[:300], user["username"])
    return _shape(store.get_event(db, eid), user)


@app.post("/api/events/{eid}/confirm")
def confirm(eid: str, body: dict = Body(default={}), user=Depends(guard_only)):
    return _decide(eid, "CONFIRMED", body.get("note"), user)


@app.post("/api/events/{eid}/dismiss")
def dismiss(eid: str, body: dict = Body(default={}), user=Depends(guard_only)):
    return _decide(eid, "DISMISSED", body.get("note"), user)


@app.post("/api/away")
def away(body: dict = Body(...), user=Depends(current_user)):
    if user["role"] != "resident":
        raise HTTPException(403, "Only residents set Away Mode")
    on = bool(body.get("on"))
    until = ""
    days = int(body.get("days") or 0)
    if on and days > 0:
        until = time.strftime("%Y-%m-%d", time.localtime(time.time() + days * 86400))
    store.set_away(db, user["house"], on, until, user["username"])
    return {"house": user["house"], "away": on, "until": until}


@app.post("/api/reset")
def reset(user=Depends(guard_only)):
    store.clear_all(db)
    return {"ok": True}


# ---------------- live stream (Server-Sent Events) ----------------
@app.get("/api/stream")
async def stream(request: Request, user=Depends(current_user)):
    async def gen():
        con = store.connect(OUT / "sentra.db")
        since = time.time()
        yield "retry: 3000\n\n"
        last_ping = time.time()
        try:
            while not await request.is_disconnected():
                for e in store.events_updated_since(con, since):
                    since = max(since, e["updated"])
                    if visible(e, user):
                        yield f"event: update\ndata: {json.dumps(_shape(e, user), default=str)}\n\n"
                if time.time() - last_ping > 15:
                    last_ping = time.time()
                    yield ": ping\n\n"                         # keeps proxies from closing the stream
                await asyncio.sleep(1.0)
        finally:
            con.close()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
