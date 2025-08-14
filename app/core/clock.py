import time
from typing import Protocol

class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...

class SystemClock:
    def now(self) -> float: return time.time()
    def sleep(self, seconds: float) -> None:
        if seconds > 0: time.sleep(seconds)