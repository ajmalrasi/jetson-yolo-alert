from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Iterable, Sequence, Optional, Tuple, runtime_checkable, Any

# ---------- Basic types ----------
@dataclass
class Detection:
    xyxy: Tuple[int, int, int, int]  # x1,y1,x2,y2
    conf: float
    cls_id: int
    track_id: Optional[int] = None

@dataclass
class Frame:
    image: Any  # np.ndarray
    t: float
    index: int
    w: int
    h: int

# ---------- Ports / Interfaces ----------
@runtime_checkable
class Camera(Protocol):
    def open(self) -> None: ...
    def read(self) -> Optional[Frame]: ...
    def close(self) -> None: ...

@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Frame) -> Sequence[Detection]: ...

@runtime_checkable
class ITracker(Protocol):
    def update(self, frame: Frame, dets: Sequence[Detection]) -> Sequence[Detection]: ...

@runtime_checkable
class AlertSink(Protocol):
    def send(self, text: str, image_path: Optional[str] = None) -> None: ...

@runtime_checkable
class EventBus(Protocol):
    def publish(self, topic: str, event: object) -> None: ...

@runtime_checkable
class Telemetry(Protocol):
    def incr(self, name: str, value: int = 1, **tags): ...
    def gauge(self, name: str, value: float, **tags): ...
    def time_ms(self, name: str, value: float, **tags): ...
