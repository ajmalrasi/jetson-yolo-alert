"""
OpenTelemetry metrics exported via OTLP (HTTP/protobuf by default).

Set OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://localhost:4318) and optionally
OTEL_SERVICE_NAME. Grafana: use Prometheus/Mimir with an OTLP-receiving
collector, or Grafana Cloud OTLP endpoint.

See docs/metrics.md.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Iterable, Optional

from opentelemetry import metrics
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from ..core.ports import Telemetry

try:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
except ImportError:  # pragma: no cover
    OTLPMetricExporter = None  # type: ignore


class OtlpTelemetry(Telemetry):
    """Maps incr/gauge/time_ms to OTEL Counter, Histogram, and ObservableGauge."""

    _singleton_lock = threading.Lock()
    _instance: Optional["OtlpTelemetry"] = None

    def __new__(cls) -> "OtlpTelemetry":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._otlp_initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_otlp_initialized", False):
            return
        if OTLPMetricExporter is None:
            raise ImportError("opentelemetry-exporter-otlp-proto-http is required")

        self._prefix = os.getenv("OTEL_METRICS_PREFIX", "jetson_yolo").strip().rstrip(
            "."
        )
        self._gauge_values: Dict[str, float] = {}

        interval_ms = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "5000"))
        resource = Resource.create(
            {
                "service.name": os.getenv("OTEL_SERVICE_NAME", "jetson-yolo-alert"),
                "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "default"),
            }
        )
        exporter = OTLPMetricExporter()
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=interval_ms,
        )

        with OtlpTelemetry._singleton_lock:
            provider = MeterProvider(
                resource=resource,
                metric_readers=[reader],
            )
            try:
                metrics.set_meter_provider(provider)
            except Exception as e:
                if "Overriding" not in str(e):
                    raise

        self._meter = metrics.get_meter(self._prefix or "jetson_yolo", "1.0.0")

        self._counter = self._meter.create_counter(
            f"{self._prefix}.events" if self._prefix else "events",
            unit="1",
            description="Pipeline counter events",
        )
        self._duration = self._meter.create_histogram(
            f"{self._prefix}.duration" if self._prefix else "duration",
            unit="s",
            description="Pipeline step durations",
        )

        def gauge_cb(options: Any) -> Iterable[Observation]:
            for name, val in list(self._gauge_values.items()):
                yield Observation(float(val), {"metric": name})

        self._meter.create_observable_gauge(
            f"{self._prefix}.gauge" if self._prefix else "gauge",
            callbacks=[gauge_cb],
            description="Pipeline gauge metrics",
        )

        self._otlp_initialized = True

    def incr(self, name: str, value: int = 1, **tags: Any) -> None:
        attrs = {"event": str(name)}
        attrs.update({k: str(v) for k, v in tags.items() if k != "msg"})
        self._counter.add(value, attrs)

    def gauge(self, name: str, value: float, **tags: Any) -> None:
        # ObservableGauge exports all entries; key by name only (tags ignored v1)
        self._gauge_values[str(name)] = float(value)

    def time_ms(self, name: str, value: float, **tags: Any) -> None:
        sec = max(0.0, float(value)) / 1000.0
        attrs = {"step": str(name)}
        attrs.update({k: str(v) for k, v in tags.items() if k != "msg"})
        self._duration.record(sec, attrs)
