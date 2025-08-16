from dataclasses import dataclass
from typing import Sequence
from .ports import Detection
from .state import PresenceState

@dataclass
class PresencePolicy:
    min_frames: int
    min_persist_sec: float

    def update(self, state: PresenceState, t: float, trigger_dets: Sequence[Detection]) -> tuple[PresenceState, bool, bool]:
        """Returns: (state, became_present, became_idle)
        Arm only when thresholds are met; until then stay idle and accumulate."""
        became_present = False
        became_idle = False
        if trigger_dets:
            if not state.present:
                # warm-up: accumulate frames/time without marking present yet
                if state.frames == 0:
                    state.first_t = t
                state.last_t = t
                state.frames += 1
                if state.frames >= self.min_frames and (t - state.first_t) >= self.min_persist_sec:
                    state.present = True
                    became_present = True
            else:
                # already present; keep freshness
                state.last_t = t
                state.frames += 1
        else:
            if state.present:
                state.on_idle(t)
                became_idle = True
            # reset warm-up counters when nothing is seen
            state.frames = 0
            state.first_t = 0.0
            state.last_t = t
        return state, became_present, became_idle
