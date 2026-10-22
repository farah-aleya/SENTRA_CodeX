"""SENTRA rule engine.

Turns tracked objects + zones into security events. Rules are specific
behaviours, not "any motion", to keep false alarms low.

  PERIMETER_BREACH  person inside a restricted zone (e.g. climbed the fence)   HIGH
  AWAY_BREACH       person at a house whose owner set Away Mode                HIGH
  LOITERING         person stays at one house longer than loiter_seconds       MEDIUM
  VEHICLE_DWELL     vehicle stays in a road zone longer than vehicle_seconds   MEDIUM

Time is measured in video seconds, so recorded clips behave like live CCTV.
"""
import time
from dataclasses import asdict, dataclass

RULE_INFO = {
    "PERIMETER_BREACH": ("HIGH", "Perimeter breach"),
    "AWAY_BREACH": ("HIGH", "Away Mode breach"),
    "LOITERING": ("MEDIUM", "Loitering"),
    "VEHICLE_DWELL": ("MEDIUM", "Vehicle dwell"),
}

DEFAULT_RULES = {
    "active_hours": "always",      # "always" or [start_hour, end_hour], e.g. [22, 6] for night only
    "breach_seconds": 1.0,         # how long a person must be in a restricted zone
    "away_seconds": 1.0,           # how long a person must be at an Away house
    "loiter_seconds": 120,         # 2 minutes in the real system
    "vehicle_seconds": 1800,       # 30 minutes in the real system
    "grace_seconds": 3.0,          # short gaps (person hidden for a moment) do not reset timers
    "cooldown_seconds": 60,        # same rule in the same zone alerts at most once per minute (no flooding)
}


@dataclass
class Event:
    rule: str
    severity: str
    title: str
    message: str
    camera_id: str
    camera_name: str
    zone_id: str
    zone_name: str
    track_id: int
    label: str
    video_time: float
    wall_time: str
    snapshot: str = ""
    clip: str = ""
    status: str = "PENDING"        # PENDING -> CONFIRMED / DISMISSED by the guard
    house: str = ""                # house number linked to the zone, used to notify that resident

    def to_dict(self):
        return asdict(self)


class RuleEngine:
    def __init__(self, cfg: dict, camera_id: str, zones, frame_size):
        self.rules = {**DEFAULT_RULES, **(cfg.get("rules") or {})}
        cam = cfg.get("cameras", {}).get(camera_id, {})
        self.camera_id = camera_id
        self.camera_name = cam.get("name", camera_id)
        self.zones = zones
        self.w, self.h = frame_size
        self.config_away = set(str(h) for h in (cfg.get("away_mode") or []))
        self.away_houses = set(self.config_away)     # refreshed from Telegram Away Mode (Step 5)
        self._in_zone = {}      # (track_id, zone_id) -> [enter_t, last_t]
        self._fired = set()     # (rule, track_id, zone_id) already alerted
        self._last_zone_alert = {}  # (rule, zone_id) -> time of last alert, for the cooldown
        self.dwell = {}         # (track_id, zone_id) -> seconds, for drawing timers

    # ------------------------------------------------------------------
    def _active_now(self) -> bool:
        ah = self.rules["active_hours"]
        if ah == "always":
            return True
        start, end = ah
        hr = time.localtime().tm_hour
        return start <= hr < end if start < end else (hr >= start or hr < end)

    def _event(self, rule, zone, track, t, extra=""):
        sev, title = RULE_INFO[rule]
        mins, secs = divmod(int(self.dwell.get((track.track_id, zone.id), 0)), 60)
        dur = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        msg = {
            "PERIMETER_BREACH": f"Person entered restricted area '{zone.label}'",
            "AWAY_BREACH": f"Person at {zone.label}, which is in Away Mode",
            "LOITERING": f"Person loitering at {zone.label} for {dur}",
            "VEHICLE_DWELL": f"{track.label.capitalize()} staying in '{zone.label}' for {dur}",
        }[rule] + extra
        return Event(rule, sev, title, msg, self.camera_id, self.camera_name, zone.id, zone.label,
                     track.track_id, track.label, round(t, 1), time.strftime("%Y-%m-%d %H:%M:%S"),
                     house=str(zone.house or ""))

    def _fire(self, rule, zone, track, t, out):
        key = (rule, track.track_id, zone.id)
        if key in self._fired:
            return
        self._fired.add(key)
        last = self._last_zone_alert.get((rule, zone.id))
        if last is not None and t - last < self.rules["cooldown_seconds"]:
            return                                   # already alerted this zone recently
        self._last_zone_alert[(rule, zone.id)] = t
        out.append(self._event(rule, zone, track, t))

    # ------------------------------------------------------------------
    def update(self, tracks, t: float) -> list[Event]:
        """Call once per frame with the current tracks and video time (seconds)."""
        r = self.rules
        out: list[Event] = []
        active = self._active_now()
        self.dwell = {}

        for tr in tracks:
            for z in self.zones:
                # people are judged by their feet, vehicles by their box centre
                pt = tr.foot if tr.is_person else ((tr.box[0] + tr.box[2]) // 2, (tr.box[1] + tr.box[3]) // 2)
                if not z.contains(pt, self.w, self.h):
                    continue
                key = (tr.track_id, z.id)
                st = self._in_zone.get(key)
                if st is None or t - st[1] > r["grace_seconds"]:
                    st = [t, t]
                st[1] = t
                self._in_zone[key] = st
                dwell = t - st[0]
                self.dwell[key] = dwell

                if tr.is_person:
                    if z.type == "restricted" and active and dwell >= r["breach_seconds"]:
                        self._fire("PERIMETER_BREACH", z, tr, t, out)
                    elif z.type == "house":
                        if z.house and str(z.house) in self.away_houses and dwell >= r["away_seconds"]:
                            self._fire("AWAY_BREACH", z, tr, t, out)       # Away Mode works at any hour
                        elif active and dwell >= r["loiter_seconds"]:
                            self._fire("LOITERING", z, tr, t, out)
                elif z.type == "road" and active and dwell >= r["vehicle_seconds"]:
                    self._fire("VEHICLE_DWELL", z, tr, t, out)

        # forget tracks not seen for a while
        stale = [k for k, (_, last) in self._in_zone.items() if t - last > 30]
        for k in stale:
            del self._in_zone[k]
        return out
