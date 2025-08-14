from dataclasses import dataclass
from typing import Sequence
from .ports import Detection
from .state import PresenceState

@dataclass
class PresencePolicy:
    min_frames: int
    min_persist_sec: float

    def update(self, state: PresenceState, t: float, trigger_dets: Sequence[Detection]) -> tuple[PresenceState, bool, bool]:
        """Returns: (state, became_present, became_idle)"""
        became_present = False
        became_idle = False
        if trigger_dets:
            prev = state.present
            state.on_seen(t)
            if not prev and state.frames >= self.min_frames and state.present_duration(t) >= self.min_persist_sec:
                became_present = True
        else:
            if state.present:
                state.on_idle(t)
        return state, became_present, became_idle
