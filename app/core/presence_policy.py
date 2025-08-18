from dataclasses import dataclass
from typing import Sequence
from .ports import Detection
from .state import PresenceState

@dataclass
class PresencePolicy:
    min_frames: int
    min_persist_sec: float

    def update(self, state: PresenceState, t: float, trigger_dets: Sequence[Detection]) -> tuple[PresenceState, bool, bool]:
        """Returns: (state, became_present, became_idle)."""
        became_present = False
        became_idle = False

        if trigger_dets:
            if not state.present:
                # warming up toward arm
                if state.frames == 0:
                    state.first_t = t
                state.frames += 1
                state.last_t = t
                if state.frames >= self.min_frames and (t - state.first_t) >= self.min_persist_sec:
                    # arm now
                    state.on_seen(t)
                    became_present = True
            else:
                # already present; keep it fresh
                state.on_seen(t)
        else:
            # nothing visible
            if state.present:
                state.on_idle(t)
                became_idle = True
            state.frames = 0
            state.first_t = 0.0
            state.last_t = t

        return state, became_present, became_idle