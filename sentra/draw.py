"""Drawing helpers for SENTRA overlays."""
import cv2

RED = (42, 24, 215)      # BGR for SENTRA red
ORANGE = (32, 130, 245)
GREEN = (80, 175, 76)
WHITE = (255, 255, 255)
NAVY = (65, 42, 27)
ZONE_COLORS = {"restricted": RED, "house": (200, 160, 40), "road": (150, 150, 150)}
SEV_COLORS = {"HIGH": RED, "MEDIUM": ORANGE}


def _label(frame, text, x, y, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(y, th + 8)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def draw_zones(frame, zones, away_houses=()):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for z in zones:
        color = RED if (z.house and str(z.house) in away_houses) else ZONE_COLORS.get(z.type, WHITE)
        cv2.fillPoly(overlay, [z.pixels(w, h)], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for z in zones:
        color = RED if (z.house and str(z.house) in away_houses) else ZONE_COLORS.get(z.type, WHITE)
        pts = z.pixels(w, h)
        cv2.polylines(frame, [pts], True, color, 2)
        tag = z.label + ("  [AWAY]" if z.house and str(z.house) in away_houses else "")
        _label(frame, tag, int(pts[:, 0].min()), int(pts[:, 1].min()) + 2, color, 0.45)
    return frame


def draw_tracks(frame, tracks, dwell=None, alerted_ids=()):
    dwell = dwell or {}
    for t in tracks:
        color = RED if t.track_id in alerted_ids else (GREEN if t.is_person else ORANGE)
        x1, y1, x2, y2 = t.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if t.track_id in alerted_ids else 2)
        secs = max([v for (tid, _), v in dwell.items() if tid == t.track_id], default=0)
        label = f"{t.label} #{t.track_id}" + (f"  {int(secs)}s" if secs >= 1 else "")
        _label(frame, label, x1, y1, color)
        cv2.circle(frame, t.foot, 4, color, -1)
    return frame


def draw_header(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), NAVY, -1)
    cv2.putText(frame, text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return frame


def draw_alert_banner(frame, event):
    h, w = frame.shape[:2]
    color = SEV_COLORS.get(event.severity, RED)
    cv2.rectangle(frame, (0, h - 44), (w, h), color, -1)
    text = f"ALERT [{event.severity}] {event.title}: {event.message}"
    scale = 0.6
    while scale > 0.35 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 24:
        scale -= 0.05                                  # shrink long messages to fit small videos
    cv2.putText(frame, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2 if scale >= 0.5 else 1, cv2.LINE_AA)
    return frame


STATUS_COLORS = {"OK": GREEN, "LEARNING": (150, 150, 150), "STARTING": (150, 150, 150),
                 "FROZEN": ORANGE, "NO SIGNAL": ORANGE, "BLOCKED": RED, "OFFLINE": RED}


def draw_status(frame, status):
    """Camera health badge in the top-right corner."""
    text = f"CAMERA {status}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    w = frame.shape[1]
    cv2.rectangle(frame, (w - tw - 22, 4), (w - 6, 24), STATUS_COLORS.get(status, RED), -1)
    cv2.putText(frame, text, (w - tw - 14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame


def draw_no_signal(frame):
    h, w = frame.shape[:2]
    frame[:] = (25, 25, 25)
    cv2.putText(frame, "NO SIGNAL", (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 3, cv2.LINE_AA)
    return frame
