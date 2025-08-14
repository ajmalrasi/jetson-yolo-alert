import logging
from ..core.ports import Telemetry

log = logging.getLogger("telemetry")
logging.basicConfig(level=logging.INFO)

class LogTelemetry(Telemetry):
    def incr(self, name: str, value: int = 1, **tags):
        log.info("metric incr %s=%s %s", name, value, tags)

    def gauge(self, name: str, value: float, **tags):
        log.info("metric gauge %s=%s %s", name, value, tags)

    def time_ms(self, name: str, value: float, **tags):
        log.info("metric timer %s=%sms %s", name, value, tags)
