from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional


@dataclass
class AlertPolicy:
    window_sec: float
    cooldown_sec: float = 0.0  # if > 0, min time between any two alerts
    last_sent: float = field(default=-1e9)
    pending_ids: set[int] = field(default_factory=set)
    pending_best: float = 0.0
    pending_img_path: str | None = None
    # Snapshot metadata used to keep message and image consistent.
    pending_snapshot_count: int = 0
    pending_snapshot_best: float = 0.0
    pending_snapshot_classes: set[str] = field(default_factory=set)
    last_by_id: Dict[int, float] = field(default_factory=dict)  # id -> last alert time

    def _min_interval_sec(self) -> float:
        return max(self.window_sec, self.cooldown_sec) if self.cooldown_sec > 0 else self.window_sec

    def add(
        self,
        ids: Iterable[int],
        best_conf: float,
        now: float,
        rearm_sec: float,
        frame_img_path: str | None = None,
        frame_count: int = 0,
        frame_best_conf: float = 0.0,
        frame_class_names: Iterable[str] | None = None,
    ):
        added_any = False
        for i in ids:
            last = self.last_by_id.get(i, -1e9)
            if (now - last) >= rearm_sec:
                self.pending_ids.add(i)
                added_any = True

        if added_any:
            self.pending_best = max(self.pending_best, best_conf)
            # Keep latest snapshot metadata so the sent message matches the sent image.
            self.pending_img_path = frame_img_path
            self.pending_snapshot_count = int(frame_count)
            self.pending_snapshot_best = float(frame_best_conf)
            self.pending_snapshot_classes = set(frame_class_names or ())

    def due(self, now: float) -> bool:
        return bool(self.pending_ids) and (now - self.last_sent) >= self._min_interval_sec()

    def flush(self, now: float) -> tuple[int, float, Optional[str], set[str]]:
        img = self.pending_img_path
        if img:
            n, b = self.pending_snapshot_count, self.pending_snapshot_best
            classes = set(self.pending_snapshot_classes)
        else:
            n, b = len(self.pending_ids), self.pending_best
            classes = set()

        for i in list(self.pending_ids):
            self.last_by_id[i] = now

        self.pending_ids.clear()
        self.pending_best = 0.0
        self.pending_img_path = None
        self.pending_snapshot_count = 0
        self.pending_snapshot_best = 0.0
        self.pending_snapshot_classes.clear()
        self.last_sent = now
        return n, b, img, classes
