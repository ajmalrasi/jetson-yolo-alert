import time as _time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional


@dataclass
class AlertPolicy:
    window_sec: float
    cooldown_sec: float = 0.0  # if > 0, min time between any two alerts
    last_sent: float = field(default=0.0)
    pending_ids: set[int] = field(default_factory=set)
    pending_best: float = 0.0
    pending_img_path: str | None = None
    pending_snapshot_count: int = 0
    pending_snapshot_best: float = 0.0
    pending_snapshot_classes: set[str] = field(default_factory=set)
    pending_snapshot_context_classes: set[str] = field(default_factory=set)
    last_by_id: Dict[int, float] = field(default_factory=dict)  # id -> last alert time
    _wall_last_sent: float = field(default=0.0, repr=False)

    def __post_init__(self):
        # Seed with wall-clock so the first alert after a restart still
        # respects cooldown (prevents restart-spam).
        if self._wall_last_sent == 0.0:
            self._wall_last_sent = _time.time()

    def _min_interval_sec(self) -> float:
        return max(self.window_sec, self.cooldown_sec) if self.cooldown_sec > 0 else self.window_sec

    def _global_cooldown_ok(self, now: float) -> bool:
        """True when enough time has passed since the last alert.

        Uses both the pipeline clock *and* wall-clock so that service
        restarts (which reset the pipeline clock) don't bypass cooldown.
        """
        pipeline_ok = (now - self.last_sent) >= self._min_interval_sec()
        wall_ok = (_time.time() - self._wall_last_sent) >= self._min_interval_sec()
        return pipeline_ok and wall_ok

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
        frame_context_class_names: Iterable[str] | None = None,
    ):
        added_any = False
        for i in ids:
            last = self.last_by_id.get(i, -1e9)
            if (now - last) >= rearm_sec:
                self.pending_ids.add(i)
                added_any = True

        if added_any:
            self.pending_best = max(self.pending_best, best_conf)
            self.pending_img_path = frame_img_path
            self.pending_snapshot_count = int(frame_count)
            self.pending_snapshot_best = float(frame_best_conf)
            self.pending_snapshot_classes = set(frame_class_names or ())
            self.pending_snapshot_context_classes = set(frame_context_class_names or ())

    def due(self, now: float) -> bool:
        return bool(self.pending_ids) and self._global_cooldown_ok(now)

    def flush(self, now: float) -> tuple[int, float, Optional[str], set[str], set[str]]:
        img = self.pending_img_path
        if img:
            n, b = self.pending_snapshot_count, self.pending_snapshot_best
            classes = set(self.pending_snapshot_classes)
            context_classes = set(self.pending_snapshot_context_classes)
        else:
            n, b = len(self.pending_ids), self.pending_best
            classes = set()
            context_classes = set()

        for i in list(self.pending_ids):
            self.last_by_id[i] = now

        self.pending_ids.clear()
        self.pending_best = 0.0
        self.pending_img_path = None
        self.pending_snapshot_count = 0
        self.pending_snapshot_best = 0.0
        self.pending_snapshot_classes.clear()
        self.pending_snapshot_context_classes.clear()
        self.last_sent = now
        self._wall_last_sent = _time.time()
        return n, b, img, classes, context_classes
