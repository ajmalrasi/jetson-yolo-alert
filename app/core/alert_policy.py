from dataclasses import dataclass, field
from typing import Iterable

@dataclass
class AlertPolicy:
    window_sec: float
    last_sent: float = field(default=-1e9)
    pending_ids: set[int] = field(default_factory=set)
    pending_best: float = 0.0

    def add(self, ids: Iterable[int], best_conf: float):
        if ids:
            self.pending_ids.update(ids)
            self.pending_best = max(self.pending_best, best_conf)

    def due(self, now: float) -> bool:
        return bool(self.pending_ids) and (now - self.last_sent) >= self.window_sec

    def flush(self, now: float) -> tuple[int, float]:
        n, b = len(self.pending_ids), self.pending_best
        self.pending_ids.clear(); self.pending_best = 0.0; self.last_sent = now
        return n, b
