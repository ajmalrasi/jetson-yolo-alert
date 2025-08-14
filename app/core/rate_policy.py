from dataclasses import dataclass
from .state import PresenceState

@dataclass
class RateTarget:
    fps: float
    vid_stride: int

@dataclass
class RatePolicy:
    base_fps: float
    high_fps: float
    boost_arm_frames: int
    boost_min_sec: float
    cooldown_sec: float
    base_stride: int

    def decide(self, state: PresenceState, now: float) -> RateTarget:
        if state.present:
            if state.frames >= self.boost_arm_frames and state.present_duration(now) >= self.boost_min_sec:
                return RateTarget(self.high_fps, 1)
            return RateTarget(max(self.base_fps, min(self.high_fps or 9999, 10)), 2)
        # cooling / idle
        if state.time_since_last_present(now) < self.cooldown_sec:
            return RateTarget(max(self.base_fps, min(self.high_fps or 9999, 5)), 2)
        return RateTarget(self.base_fps, self.base_stride)
