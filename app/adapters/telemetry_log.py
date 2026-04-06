import logging
import os
from ..core.ports import Telemetry

log = logging.getLogger("telemetry")

# Configure this logger only (do not call logging.basicConfig — it hides INFO on root).
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [telemetry] %(message)s")
    )
    log.addHandler(_h)
log.setLevel(os.getenv("TELEMETRY_LOG_LEVEL", "INFO").upper())
log.propagate = False


class LogTelemetry(Telemetry):
    def incr(self, name: str, value: int = 1, **tags):
        log.info("metric incr %s=%s %s", name, value, tags)

    def gauge(self, name: str, value: float, **tags):
        log.info("metric gauge %s=%s %s", name, value, tags)

    def time_ms(self, name: str, value: float, **tags):
        log.info("metric time_ms %s=%.3f %s", name, value, tags)
