from dataclasses import dataclass

@dataclass
class PresenceState:
    present: bool = False
    first_t: float = 0.0
    last_t: float = 0.0
    frames: int = 0

    def on_seen(self, t: float):
        if not self.present:
            self.present = True
            self.first_t = t
            self.frames = 0
        self.last_t = t
        self.frames += 1

    def on_idle(self, t: float):
        if self.present:
            # transition to idle; keep last_t as the last seen time
            self.present = False

    def present_duration(self, t: float) -> float:
        if not self.present: return 0.0
        return max(0.0, (self.last_t if self.last_t else t) - self.first_t)

    def time_since_last_present(self, t: float) -> float:
        return max(0.0, t - (self.last_t or t))
