from dataclasses import dataclass, field
from typing import Iterable, Dict

@dataclass
class AlertPolicy:
    window_sec: float
    cooldown_sec: float = 0.0  # if > 0, min time between any two alerts (reduces duplicate "same person" when track_id changes)
    last_sent: float = field(default=-1e9)
    pending_ids: set[int] = field(default_factory=set)
    pending_best: float = 0.0
    last_by_id: Dict[int, float] = field(default_factory=dict)  # id -> last alert time

    def _min_interval_sec(self) -> float:
        return max(self.window_sec, self.cooldown_sec) if self.cooldown_sec > 0 else self.window_sec

    def add(self, ids: Iterable[int], best_conf: float, now: float, rearm_sec: float, frame_img_path: str = None):
        added_any = False
        for i in ids:
            last = self.last_by_id.get(i, -1e9)
            if (now - last) >= rearm_sec:
                self.pending_ids.add(i)
                added_any = True
        if added_any:
            self.pending_best = max(self.pending_best, best_conf)
            if frame_img_path and not getattr(self, "pending_img_path", None):
                self.pending_img_path = frame_img_path

    def due(self, now: float) -> bool:
        return bool(self.pending_ids) and (now - self.last_sent) >= self._min_interval_sec()

    def flush(self, now: float) -> tuple[int, float, str]:
        n, b = len(self.pending_ids), self.pending_best
        img = getattr(self, "pending_img_path", None)
        for i in list(self.pending_ids):
            self.last_by_id[i] = now
        self.pending_ids.clear()
        self.pending_best = 0.0
        self.pending_img_path = None
        self.last_sent = now
        return n, b, img
