# Pipeline metrics (telemetry)

Quick overview: [README.md](../README.md) (sections **Live Preview** and **Telemetry and Grafana**). This page lists metric names and OTLP/Grafana setup in detail.

## Log backend (default)

Set `TELEMETRY_BACKEND=log` (or omit). Metrics are printed on the `telemetry` logger at `INFO` (override with `TELEMETRY_LOG_LEVEL`).

Per-frame timings (when work runs):

| `time_ms` name     | Meaning                                      |
|--------------------|----------------------------------------------|
| `rate_sleep_ms`    | Time spent sleeping for FPS cap (RateStep)   |
| `read_ms`          | `VideoCapture.read()` / camera               |
| `detect_ms`        | `detector.detect()` (YOLO / TensorRT)        |
| `track_ms`         | External `tracker.update()` (if configured)  |
| `alert_ms`         | Alert step (snapshots, DB, Telegram)         |
| `pipeline_loop_ms` | Full pipeline iteration                      |

Counters/gauges include `frames`, `detect_errors`, `present`, `fps_target`, `vid_stride`, etc.

## OTLP (OpenTelemetry) — Grafana / Mimir / Alloy

1. Run an **OTLP** endpoint. The app **pushes** metrics (no in-process `/metrics` scrape).

   **Local collector + Grafana (Compose profile `observability`):**

   ```bash
   docker compose --profile observability up -d otel-collector grafana
   ```

   - Collector receives OTLP **HTTP** on **4318** (see [`otel/otel-collector.yaml`](otel/otel-collector.yaml)).
   - It exposes a **Prometheus** scrape endpoint on **8889** (for Grafana’s Prometheus datasource).

2. In `.env` (or Compose `environment`):

   ```bash
   TELEMETRY_BACKEND=otlp
   OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
   OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
   ```

   Standard OpenTelemetry env vars are honored (see [OTEL resource config](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/)).

3. **Grafana:** add a **Prometheus** data source with URL `http://localhost:8889` (same host as collector when using `network_mode: host`). Explore metrics with names like `jetson_yolo_duration_seconds` (histogram, **seconds**) and `jetson_yolo_events_total` (counter with label `event`).

4. **Grafana Cloud / managed OTLP:** set `OTEL_EXPORTER_OTLP_ENDPOINT` to your vendor URL and add auth headers via `OTEL_EXPORTER_OTLP_HEADERS` if required.

## Environment variables

| Variable                         | Default              | Description                                      |
|----------------------------------|----------------------|--------------------------------------------------|
| `TELEMETRY_BACKEND`              | `log`                | `log` or `otlp` (aliases: `otel`, `opentelemetry`) |
| `TELEMETRY_LOG_LEVEL`           | `INFO`               | Log level for the `telemetry` logger only      |
| `OTEL_EXPORTER_OTLP_ENDPOINT`   | (SDK default)        | OTLP receiver URL, e.g. `http://127.0.0.1:4318`  |
| `OTEL_EXPORTER_OTLP_PROTOCOL`   | often `http/protobuf`| Must match the collector                         |
| `OTEL_SERVICE_NAME`              | `jetson-yolo-alert`  | `service.name` resource attribute                |
| `OTEL_METRICS_PREFIX`             | `jetson_yolo`        | Prefix for instrument names (dots in OTLP names) |
| `OTEL_METRIC_EXPORT_INTERVAL_MS` | `5000`               | Export batch interval                            |

Rebuild the Docker image after changing `requirements.txt` so OpenTelemetry packages are installed.
