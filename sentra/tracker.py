"""SENTRA tracker: a small ByteTrack-style IoU tracker (no PyTorch, no Ultralytics).

Gives every person / vehicle a stable ID across frames so the rule engine can
measure dwell time and loitering.

How it works (two-stage matching, like ByteTrack):
  1. Predict where each track is now (constant velocity).
  2. Match high-confidence detections to tracks by IoU.
  3. Match the leftover low-confidence detections to the leftover tracks
     (recovers people who are partly hidden, so their ID does not change).
  4. New tracks start from unmatched high-confidence detections and are shown
     once they have been seen in `min_hits` frames.
  5. Tracks not seen for `max_age` frames are removed.
People are never matched to vehicles and vice versa.
"""
import numpy as np

try:  # optimal assignment if `lap` is installed (it is in requirements.txt)
    import lap
except ImportError:  # pragma: no cover
    lap = None


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), boxes as x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def assign(cost: np.ndarray, max_cost: float):
    """Return (matches, unmatched_rows, unmatched_cols) for a cost matrix."""
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    if lap is not None:
        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=max_cost)
        matches = [(r, c) for r, c in enumerate(x) if c >= 0]
        return matches, [r for r in range(rows) if x[r] < 0], [c for c in range(cols) if y[c] < 0]
    # greedy fallback
    matches, used_r, used_c = [], set(), set()
    for idx in np.argsort(cost, axis=None):
        r, c = divmod(int(idx), cols)
        if cost[r, c] > max_cost:
            break
        if r not in used_r and c not in used_c:
            matches.append((r, c))
            used_r.add(r)
            used_c.add(c)
    return matches, [r for r in range(rows) if r not in used_r], [c for c in range(cols) if c not in used_c]


class _Track:
    def __init__(self, tid, box, label, conf):
        self.id = tid
        self.box = np.asarray(box, dtype=np.float32)
        self.vel = np.zeros(4, dtype=np.float32)
        self.label = label
        self.conf = conf
        self.hits = 1
        self.misses = 0

    def predict(self) -> np.ndarray:
        return self.box + self.vel

    def update(self, box, conf):
        box = np.asarray(box, dtype=np.float32)
        self.vel = 0.6 * self.vel + 0.4 * (box - self.box)   # smoothed velocity
        self.box = box
        self.conf = conf
        self.hits += 1
        self.misses = 0


class IoUTracker:
    def __init__(self, high_conf=0.35, low_conf=0.1, match_iou=0.2, max_age=30, min_hits=2):
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.max_cost = 1.0 - match_iou
        self.max_age = max_age          # frames a lost track is kept (about 1 s at 30 fps)
        self.min_hits = min_hits
        self.tracks: list[_Track] = []
        self._next_id = 1
        self.frame_no = 0

    @staticmethod
    def _group(label):
        return "person" if label == "person" else "vehicle"

    def _cost(self, tracks, dets):
        boxes_t = np.array([t.predict() for t in tracks], dtype=np.float32).reshape(-1, 4)
        boxes_d = np.array([d[0] for d in dets], dtype=np.float32).reshape(-1, 4)
        cost = 1.0 - iou_matrix(boxes_t, boxes_d)
        for i, t in enumerate(tracks):          # never match a person to a car
            for j, d in enumerate(dets):
                if self._group(t.label) != self._group(d[2]):
                    cost[i, j] = 1.0
        return cost

    def update(self, detections):
        """detections: list of (box, conf, label). Returns list of (track_id, box, conf, label)."""
        self.frame_no += 1
        high = [d for d in detections if d[1] >= self.high_conf]
        low = [d for d in detections if self.low_conf <= d[1] < self.high_conf]

        # stage 1: all tracks vs high-confidence detections
        matches, um_t, um_d = assign(self._cost(self.tracks, high), self.max_cost)
        updated = set()
        for ti, di in matches:
            self.tracks[ti].update(high[di][0], high[di][1])
            updated.add(ti)

        # stage 2: leftover tracks vs low-confidence detections
        left = [self.tracks[i] for i in um_t]
        m2, um_t2, _ = assign(self._cost(left, low), self.max_cost)
        for ti, di in m2:
            left[ti].update(low[di][0], low[di][1])
            updated.add(self.tracks.index(left[ti]))

        # unmatched tracks age; unmatched high detections become new tracks
        for i, t in enumerate(self.tracks):
            if i not in updated:
                t.misses += 1
                t.box = t.predict()
        for di in um_d:
            box, conf, label = high[di]
            self.tracks.append(_Track(self._next_id, box, label, conf))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        out = []
        for t in self.tracks:
            if t.misses == 0 and (t.hits >= self.min_hits or self.frame_no <= self.min_hits):
                out.append((t.id, tuple(int(v) for v in t.box), float(t.conf), t.label))
        return out
