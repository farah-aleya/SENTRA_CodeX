"""SENTRA agent: a perception -> decision -> action loop on top of the rule engine.

When a rule fires (perimeter breach, Away Mode breach, loitering, vehicle dwell)
SENTRA does not alert straight away. It opens a *case* and hands it to an agent
running on Amazon Bedrock (Claude Haiku 4.5, Amazon Nova as backup). The agent
gathers evidence with OpenCV 5 tools, then must take exactly one action:

  Perception tools (OpenCV 5)
    recheck_region       crop + upscale + CLAHE the area around the person over the
                         next seconds, re-run cv.dnn detection, measure motion with
                         Farneback optical flow, and test whether they move toward the zone
  Context tools
    get_track_history    dwell time, distance moved, people in the scene
    get_zone_context     zone type, Away Mode, night/day, how guards handled past
                         alerts in this zone (CONFIRMED / DISMISSED / LOGGED)
  Actions
    request_human_approval   send to the guard pop-up (nothing escalates without a human)
    log_only                 keep the record, no alert

Safety rules enforced in code, not left to the model:
  * A HIGH alert (perimeter or Away Mode breach) can only be logged when the OpenCV
    re-check found a person in under 30% of frames. Otherwise it goes to a human.
  * If Bedrock is unreachable or the model misbehaves, a local policy decides,
    using the same OpenCV tools, so the system never goes silent.
  * The model only ever receives numbers and labels. No image, face or clip
    leaves the site.

Every step (tool call, tool result, decision, timing) is saved as a trace.
"""
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

DEFAULTS = {
    "enabled": True,
    "provider": "bedrock",                                    # bedrock | local
    "region": "ap-southeast-2",
    "model_id": "au.anthropic.claude-haiku-4-5-20251001-v1:0",
    "fallback_model_id": "apac.amazon.nova-lite-v1:0",
    "recheck_seconds": 3.0,                                   # frames collected after the event
    "max_turns": 6,
    "low_person_ratio": 0.3,                                  # guardrail threshold for HIGH alerts
}
AGENT_RULES = {"PERIMETER_BREACH", "AWAY_BREACH", "LOITERING", "VEHICLE_DWELL"}

SYSTEM_PROMPT = """You are the SENTRA security triage agent for a residential township.
A rule on a CCTV camera has fired and opened a case. Decide whether a human guard must review it.

How to work:
1. Gather evidence with the tools first. Always call recheck_region at least once; it re-examines
   the video with OpenCV. Call get_track_history and get_zone_context when they could change your decision.
   If the first re-check is weak (low_light is true, low confidence, or few frames confirmed), call
   recheck_region again with enhance=true and window_seconds=3.
2. Then take exactly one action: request_human_approval or log_only.

Guidance:
- Perimeter breach and Away Mode breach are serious. Only log them if the re-check shows there is no real person.
- Loitering by a person who is still, at night, near an entry point, or at an Away Mode house needs a human.
- Brief, normal activity (a passer-by, a delivery that leaves, a resident moving normally in daytime) can be logged.
- If guards dismissed many similar alerts in this zone, weigh that, but never ignore strong evidence.
- Keep summaries short, factual and based on the tool results. Never guess identity or appearance."""

TOOLS = [
    {"toolSpec": {
        "name": "recheck_region",
        "description": "Re-examine the area around the tracked object with OpenCV 5 over the frames just after the event: "
                       "crop and upscale the region, optionally enhance contrast (CLAHE, for night or IR video), re-run the "
                       "person/vehicle detector, and measure motion with optical flow. Returns how many frames confirmed the "
                       "object, detector confidence, motion level and whether it moves toward the zone.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "window_seconds": {"type": "number", "description": "Seconds of video to check after the event, 0.5 to 3."},
            "enhance": {"type": "boolean", "description": "Apply CLAHE contrast enhancement (use for dark or IR video)."}},
            "required": []}}}},
    {"toolSpec": {
        "name": "get_track_history",
        "description": "How long the object has been in the zone, how far it moved, and how many other people are in the scene.",
        "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}}}},
    {"toolSpec": {
        "name": "get_zone_context",
        "description": "Zone type, house number, whether Away Mode is on, local time (night or day), and how guards "
                       "handled past alerts in this zone in the last 24 hours.",
        "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}}}},
    {"toolSpec": {
        "name": "request_human_approval",
        "description": "FINAL ACTION. Send the case to the guard as a pop-up with the clip. The guard approves or dismisses.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "priority": {"type": "string", "enum": ["HIGH", "MEDIUM"]},
            "summary": {"type": "string", "description": "One factual sentence for the guard, under 200 characters."},
            "evidence": {"type": "array", "items": {"type": "string"}, "description": "Key facts from the tools."}},
            "required": ["priority", "summary", "evidence"]}}}},
    {"toolSpec": {
        "name": "log_only",
        "description": "FINAL ACTION. Keep the event in the log without alerting anyone.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "One factual sentence explaining why no alert is needed."}},
            "required": ["reason"]}}}},
]
FINAL_TOOLS = {"request_human_approval", "log_only"}


# ---------------------------------------------------------------------------------------------
# Case: everything the agent may look at for one event (kept in memory only, never written to disk)
# ---------------------------------------------------------------------------------------------
class Case:
    def __init__(self, eid, event: dict, pre_frames, fps, n_post, zone, frame_size, scene):
        self.eid = eid
        self.event = event
        self.fps = fps
        self.frames = list(pre_frames)          # [(raw_frame, {track_id: (box, label, conf)})]
        self.n_pre = len(self.frames)
        self.remaining = n_post
        self.zone = zone
        self.w, self.h = frame_size
        self.scene = scene                      # snapshot of rule-engine state at event time
        self.created = time.time()

    def add(self, frame, boxes):
        self.frames.append((frame, boxes))
        self.remaining -= 1
        return self.remaining <= 0


def _expand(box, w, h, scale=2.0, min_side=160):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side_w = max((x2 - x1) * scale, min_side)
    side_h = max((y2 - y1) * scale, min_side)
    return (int(max(0, cx - side_w / 2)), int(max(0, cy - side_h / 2)),
            int(min(w, cx + side_w / 2)), int(min(h, cy + side_h / 2)))


class Tools:
    """The agent's tools. recheck_region is pure OpenCV 5; the others read SENTRA state."""

    def __init__(self, case: Case, detector, db_path, rules_cfg):
        self.case = case
        self.detector = detector
        self.db_path = db_path
        self.rules = rules_cfg
        self.last_recheck = None

    # ----- perception -----
    def recheck_region(self, window_seconds=2.0, enhance=False):
        c = self.case
        tid = c.event.get("track_id", -1)
        window_seconds = float(min(max(window_seconds or 2.0, 0.5), 3.0))
        post = c.frames[c.n_pre:][: max(1, int(c.fps * window_seconds))] or c.frames[-1:]
        step = max(1, len(post) // 10)                      # about 10 frames are re-examined
        sample = post[::step]
        last_box = c.event.get("_box")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        checked = confirmed = 0
        confs, flows, prev_gray, light = [], [], None, []
        foot_start = foot_end = None
        want = "person" if c.event.get("label") == "person" else None
        for frame, boxes in sample:
            if tid in boxes:
                last_box = boxes[tid][0]
            if last_box is None:
                continue
            x1, y1, x2, y2 = _expand(last_box, c.w, c.h)
            crop = frame[y1:y2, x1:x2]
            light.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            if crop.size == 0:
                continue
            s = 640 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            if enhance:
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            dets = self.detector.detect(crop, min_conf=0.25)
            hits = [d for d in dets if (want is None and d[2] != "person") or d[2] == want]
            checked += 1
            if hits:
                confirmed += 1
                confs.append(max(d[1] for d in hits))
            gray = cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flows.append(float(np.linalg.norm(flow, axis=2).mean()))
            prev_gray = gray
            foot = ((last_box[0] + last_box[2]) / 2, last_box[3])
            foot_start = foot_start or foot
            foot_end = foot
        toward = None
        moved = 0.0
        if foot_start and foot_end and c.zone is not None:
            pts = c.zone.pixels(c.w, c.h).astype(np.float32)
            zc = pts.mean(0)
            d0 = np.linalg.norm(np.array(foot_start) - zc)
            d1 = np.linalg.norm(np.array(foot_end) - zc)
            toward = bool(d1 < d0 - 0.01 * c.h)
            moved = float(np.linalg.norm(np.array(foot_end) - np.array(foot_start)) / c.h)
        motion = float(np.mean(flows)) if flows else 0.0
        res = {
            "frames_checked": checked,
            "frames_confirmed": confirmed,
            "person_ratio" if want else "object_ratio": round(confirmed / checked, 2) if checked else 0.0,
            "max_confidence": round(max(confs), 2) if confs else 0.0,
            "mean_confidence": round(float(np.mean(confs)), 2) if confs else 0.0,
            "motion_level": "still" if motion < 0.3 else ("moving" if motion < 2.0 else "fast"),
            "moved_fraction_of_frame": round(moved, 3),
            "moving_toward_zone": toward,
            "low_light": bool(light and np.mean(light) < 70),
            "enhanced": bool(enhance),
            "window_seconds": window_seconds,
        }
        self.last_recheck = res
        return res

    # ----- context -----
    def get_track_history(self):
        s = self.case.scene
        return {
            "object": self.case.event.get("label"),
            "rule": self.case.event.get("rule"),
            "severity": self.case.event.get("severity"),
            "seconds_in_zone": s.get("dwell"),
            "seconds_tracked_total": s.get("tracked_for"),
            "people_in_scene": s.get("people"),
            "vehicles_in_scene": s.get("vehicles"),
        }

    def get_zone_context(self):
        from sentra import store
        z = self.case.zone
        con = store.connect(self.db_path)
        since = time.time() - 86400
        rows = con.execute("SELECT status, COUNT(*) n FROM events WHERE zone_id=? AND camera_id=? AND created>? "
                           "AND id<>? GROUP BY status",
                           (self.case.event.get("zone_id"), self.case.event.get("camera_id"), since, self.case.eid)
                           ).fetchall()
        con.close()
        hour = datetime.now().hour
        return {
            "zone_name": z.label if z else self.case.event.get("zone_name"),
            "zone_type": z.type if z else None,
            "house": str(z.house) if z and z.house else None,
            "away_mode_on": bool(self.case.scene.get("away")),
            "local_hour": hour,
            "is_night": hour >= 19 or hour < 7,
            "past_24h_alerts_here": {r["status"]: r["n"] for r in rows},
        }

    def run(self, name, args):
        if name == "recheck_region":
            return self.recheck_region(**{k: v for k, v in (args or {}).items() if k in ("window_seconds", "enhance")})
        if name == "get_track_history":
            return self.get_track_history()
        if name == "get_zone_context":
            return self.get_zone_context()
        raise ValueError(f"unknown tool {name}")


# ---------------------------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------------------------
class SentraAgent:
    def __init__(self, cfg: dict, detector, db_path, client=None):
        self.cfg = {**DEFAULTS, **((cfg or {}).get("agent") or {})}
        self.rules_cfg = (cfg or {}).get("rules") or {}
        self.detector = detector
        self.db_path = db_path
        self._client = client
        self._client_error = None

    @property
    def client(self):
        if self._client is None and self._client_error is None and self.cfg["provider"] == "bedrock":
            try:
                import boto3
                from botocore.config import Config
                self._client = boto3.client("bedrock-runtime", region_name=self.cfg["region"],
                                            config=Config(read_timeout=20, connect_timeout=5,
                                                          retries={"max_attempts": 2}))
            except Exception as e:                                         # noqa: BLE001
                self._client_error = f"{type(e).__name__}: {e}"
        return self._client

    def _user_brief(self, case):
        e = case.event
        return (f"New case {case.eid}. Rule: {e['rule']} ({e['severity']}). Camera: {e['camera_name']}. "
                f"Zone: {e['zone_name']}. Object: {e['label']} #{e['track_id']}. Message: {e['message']}. "
                "Investigate with the tools, then take one final action.")

    def decide(self, case: Case) -> dict:
        tools = Tools(case, self.detector, self.db_path, self.rules_cfg)
        trace, t0 = [], time.time()
        decision = None
        model_used = None
        if self.cfg["provider"] == "bedrock" and self.client is not None:
            for model_id in [self.cfg["model_id"], self.cfg["fallback_model_id"]]:
                if not model_id:
                    continue
                try:
                    decision = self._bedrock_loop(model_id, case, tools, trace)
                    model_used = model_id
                    if decision:
                        break
                except Exception as e:                                     # noqa: BLE001
                    trace.append({"type": "error", "model": model_id, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        elif self._client_error:
            trace.append({"type": "error", "error": self._client_error})
        if decision is None:
            decision = self._local_policy(tools, trace)
            model_used = "local-policy"
        decision = self._guardrail(case, tools, decision, trace)
        decision.update({"model": model_used, "ms": int((time.time() - t0) * 1000), "trace": trace})
        return decision

    def _bedrock_loop(self, model_id, case, tools, trace):
        messages = [{"role": "user", "content": [{"text": self._user_brief(case)}]}]
        for turn in range(self.cfg["max_turns"]):
            t = time.time()
            resp = self.client.converse(
                modelId=model_id, messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                toolConfig={"tools": TOOLS, "toolChoice": {"any": {}}},
                inferenceConfig={"maxTokens": 600, "temperature": 0},
            )
            msg = resp["output"]["message"]
            messages.append(msg)
            uses = [c["toolUse"] for c in msg["content"] if "toolUse" in c]
            for c in msg["content"]:
                if c.get("text"):
                    trace.append({"type": "model_text", "model": model_id, "text": c["text"][:400]})
            if not uses:
                trace.append({"type": "error", "model": model_id, "error": "model returned no tool call"})
                return None
            results = []
            for u in uses:
                name, args = u["name"], u.get("input") or {}
                trace.append({"type": "tool_call", "turn": turn, "model": model_id, "name": name, "input": args,
                              "model_ms": int((time.time() - t) * 1000)})
                if name in FINAL_TOOLS:
                    if tools.last_recheck is None:              # the agent must look before it decides
                        out = {"error": "Call recheck_region before taking a final action."}
                        trace.append({"type": "tool_result", "name": name, "output": out})
                        results.append({"toolResult": {"toolUseId": u["toolUseId"], "content": [{"json": out}],
                                                       "status": "error"}})
                        continue
                    return self._final(name, args, source="agent")
                ts = time.time()
                try:
                    out = tools.run(name, args)
                    status = "success"
                except Exception as e:                                 # noqa: BLE001
                    out, status = {"error": str(e)[:200]}, "error"
                trace.append({"type": "tool_result", "name": name, "output": out,
                              "tool_ms": int((time.time() - ts) * 1000)})
                results.append({"toolResult": {"toolUseId": u["toolUseId"], "content": [{"json": out}],
                                               "status": status}})
            messages.append({"role": "user", "content": results})
        trace.append({"type": "error", "model": model_id, "error": "no final action within max_turns"})
        return None

    @staticmethod
    def _final(name, args, source):
        if name == "request_human_approval":
            return {"action": "REQUEST_APPROVAL", "priority": args.get("priority", "MEDIUM"),
                    "summary": str(args.get("summary", ""))[:300], "evidence": args.get("evidence", []),
                    "decided_by": source}
        return {"action": "LOG_ONLY", "priority": "", "summary": str(args.get("reason", ""))[:300],
                "evidence": [], "decided_by": source}

    def _local_policy(self, tools, trace):
        """Used when Bedrock is not available. Same OpenCV tools, fixed rules."""
        ev = tools.case.event
        r = tools.recheck_region(2.0, enhance=False)
        trace.append({"type": "tool_result", "name": "recheck_region", "output": r, "source": "local-policy"})
        if r["low_light"] or (r["frames_confirmed"] and r.get("mean_confidence", 0) < 0.4):
            r = tools.recheck_region(3.0, enhance=True)
            trace.append({"type": "tool_result", "name": "recheck_region", "output": r, "source": "local-policy"})
        ratio = r.get("person_ratio", r.get("object_ratio", 0))
        if ratio < self.cfg["low_person_ratio"]:
            out = self._final("log_only", {"reason": f"Re-check confirmed the {ev['label']} in only "
                                                     f"{r['frames_confirmed']}/{r['frames_checked']} frames."}, "local-policy")
        else:
            out = self._final("request_human_approval", {
                "priority": "HIGH" if ev["severity"] == "HIGH" else "MEDIUM",
                "summary": f"{ev['message']}. Confirmed in {r['frames_confirmed']}/{r['frames_checked']} frames, "
                           f"{r['motion_level']}.",
                "evidence": [f"{k}={v}" for k, v in r.items()]}, "local-policy")
        trace.append({"type": "decision", **{k: out[k] for k in ("action", "priority", "summary", "decided_by")}})
        return out

    def _guardrail(self, case, tools, decision, trace):
        """Code-level safety: a HIGH alert is only logged when OpenCV found no real person."""
        if decision["action"] == "LOG_ONLY" and case.event["severity"] == "HIGH":
            r = tools.last_recheck or tools.recheck_region(2.0, enhance=True)
            ratio = r.get("person_ratio", r.get("object_ratio", 0))
            if ratio >= self.cfg["low_person_ratio"]:
                trace.append({"type": "guardrail", "rule": "HIGH alert with person confirmed cannot be log_only",
                              "person_ratio": ratio, "original": decision["summary"]})
                decision = {"action": "REQUEST_APPROVAL", "priority": "HIGH",
                            "summary": f"{case.event['message']} (guardrail: person confirmed in {int(ratio * 100)}% "
                                       "of re-checked frames)",
                            "evidence": [f"agent wanted log_only: {decision['summary']}"],
                            "decided_by": decision["decided_by"] + "+guardrail"}
        if decision.get("decided_by") == "agent":
            trace.append({"type": "decision", **{k: decision[k] for k in ("action", "priority", "summary", "decided_by")}})
        return decision


# ---------------------------------------------------------------------------------------------
# Plumbing used by scripts/run_sentra.py
# ---------------------------------------------------------------------------------------------
class AgentRunner:
    """Collects frames for open cases and runs the agent in a background thread."""

    def __init__(self, cfg, detector, db_path, fps, frame_size, on_decision=None, client=None):
        # the agent gets its OWN OpenCV DNN network: cv.dnn nets must not be shared between threads
        from sentra.detector import Detector
        own = Detector(str(detector.weights), conf=detector.conf, imgsz=640)
        self.agent = SentraAgent(cfg, own, db_path, client=client)
        self.enabled = bool(self.agent.cfg["enabled"])
        self.fps = fps
        self.frame_size = frame_size
        self.pre = deque(maxlen=max(1, int(fps * 0.5)))
        self.n_post = max(1, int(fps * self.agent.cfg["recheck_seconds"]))
        self.open_cases: list[Case] = []
        self.q = queue.Queue()
        self.on_decision = on_decision
        self.max_w = 960
        self._t = threading.Thread(target=self._work, daemon=True)
        self._t.start()

    def _small(self, frame, boxes):
        h, w = frame.shape[:2]
        if w <= self.max_w:
            return frame.copy(), boxes
        s = self.max_w / w
        f = cv2.resize(frame, (self.max_w, int(h * s)))
        return f, {k: (tuple(int(v * s) for v in b), lab, c) for k, (b, lab, c) in boxes.items()}

    @property
    def size(self):
        w, h = self.frame_size
        return (self.max_w, int(h * self.max_w / w)) if w > self.max_w else (w, h)

    def push(self, raw_frame, tracks):
        """Call every frame with the RAW (unblurred) frame. Frames stay in memory only."""
        if not self.enabled:
            return
        boxes = {t.track_id: (t.box, t.label, t.conf) for t in tracks}
        item = self._small(raw_frame, boxes)
        self.pre.append(item)
        for c in list(self.open_cases):
            if c.add(*item):
                self.open_cases.remove(c)
                self.q.put(c)

    def open(self, eid, event: dict, zone, scene):
        w, h = self.frame_size
        s = self.size[0] / w
        ev = dict(event)
        box = scene.get("box")
        ev["_box"] = tuple(int(v * s) for v in box) if box else None
        case = Case(eid, ev, self.pre, self.fps, self.n_post, zone, self.size, scene)
        self.open_cases.append(case)

    def _work(self):
        while True:
            case = self.q.get()
            if case is None:
                break
            try:
                d = self.agent.decide(case)
            except Exception as e:                                         # noqa: BLE001
                d = {"action": "REQUEST_APPROVAL", "priority": case.event.get("severity", "MEDIUM"),
                     "summary": f"{case.event.get('message')} (agent error, sent to guard)", "evidence": [],
                     "decided_by": "error-fallback", "model": None, "ms": 0,
                     "trace": [{"type": "error", "error": f"{type(e).__name__}: {e}"}]}
            case.frames = []                                               # free memory
            if self.on_decision:
                self.on_decision(case.eid, case.event, d)

    def close(self, timeout=120):
        for c in self.open_cases:                                          # video ended: decide with what we have
            self.q.put(c)
        self.open_cases = []
        self.q.put(None)
        self._t.join(timeout)
