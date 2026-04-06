"""Select telemetry backend: TELEMETRY_BACKEND=log (default) or otlp."""
import logging
import os

from ..core.ports import Telemetry

_log = logging.getLogger("telemetry")


def get_telemetry() -> Telemetry:
    backend = os.getenv("TELEMETRY_BACKEND", "log").strip().lower()
    if backend in ("otlp", "otel", "opentelemetry"):
        try:
            from .telemetry_otlp import OtlpTelemetry

            return OtlpTelemetry()
        except ImportError as e:
            _log.warning(
                "OpenTelemetry OTLP packages missing; install requirements.txt. "
                "Falling back to log: %s",
                e,
            )
    from .telemetry_log import LogTelemetry

    return LogTelemetry()
