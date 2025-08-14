from dataclasses import dataclass
from typing import Sequence, Optional
from .ports import Detection

@dataclass
class FrameTick:
    frame_index: int
    t: float

@dataclass
class PersonDetected:
    track_ids: Sequence[int]
    best_conf: float

@dataclass
class PersonLost:
    last_seen_t: float

@dataclass
class AlertIssued:
    count: int
    best_conf: float
    image_path: Optional[str]
