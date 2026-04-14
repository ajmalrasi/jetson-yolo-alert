# Docker Compose Services

## Services Overview

| Service | Purpose | GPU | Auto-start |
|---------|---------|-----|------------|
| `jetson-yolo` | Base image / interactive shell | Yes | Started by `alert` via depends_on |
| `exporter` | One-shot: converts .pt to TensorRT .engine | Yes | Manual |
| `alert` | Detection pipeline + Telegram alerts + frame capture | Yes | Yes |
| `ask-telegram` | Telegram bot (`/ask`, `/describe`, video uploads) | No | Recommended |
| `preview` | Live MJPEG stream / local display | Yes | Optional |
| `otel-collector` | Receives OTLP metrics, exposes Prometheus scrape | No | Optional (profile: observability) |
| `grafana` | Dashboards | No | Optional (profile: observability) |

## Setup Order

### 1. Build the Docker image

```bash
docker compose build
```

### 2. Export TensorRT engine

Required once per device (or when changing YOLO_MODEL / JetPack version):

```bash
docker compose run --rm exporter
```

### 3. Start the pipeline

Detection + alerts + Telegram bot:

```bash
docker compose up -d alert ask-telegram
```

Alerts only (no Telegram bot):

```bash
docker compose up -d alert
```

### 4. (Optional) Preview

For headless Jetson, set `PREVIEW_STREAM_PORT=8080` in `.env`, then:

```bash
docker compose build preview && docker compose up preview
```

Open `http://<JETSON_IP>:8080/` in a browser. Raw MJPEG at `http://<JETSON_IP>:8080/stream`.

**Detector-only mode** (`PREVIEW_DETECTOR_ONLY=1`): full-speed preview with track IDs, no alert/DB/snapshot side effects. Does not stop the `alert` service.

**Reducing lag**: set `CAP_PROP_BUFFERSIZE=1` and optionally `CAMERA_GRAB_FLUSH=8`. For RTSP, tune `RTSP_LATENCY_MS` and `USE_GSTREAMER`.

### 5. (Optional) Telemetry and Grafana

```bash
# Set in .env:
# TELEMETRY_BACKEND=otlp
# OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318

docker compose --profile observability up -d otel-collector grafana
```

Point Grafana's Prometheus datasource at `http://localhost:8889`. See [metrics.md](metrics.md) for details.

## Alert Logic

The pipeline registers a detection alert when:

1. Object class is in `TRIGGER_CLASSES`
2. Confidence >= `CONF_THRESH`
3. Object persists for at least `MIN_PERSIST_SEC` and `MIN_FRAMES`
4. Alert is rate-limited by `RATE_WINDOW_SEC` (e.g., max 1 message per 30s)
5. Per-object re-trigger is throttled by `REARM_SEC`

When triggered: saves an annotated snapshot, writes to SQLite, sends photo + caption to Telegram.

## Frame Capture (for `/describe`)

When the `alert` service detects trigger classes, the `FrameCaptureStep` saves frames at `CAPTURE_ACTIVE_FPS` (default 2 fps) to `FRAMES_DIR`. Saves nothing when idle. Stays active for `CAPTURE_COOLDOWN_SEC` (default 10s) after the last detection to capture exits.

Frames are indexed in SQLite (`frame_index.db`) with timestamps and detected classes. Auto-cleanup removes frames older than `FRAMES_RETENTION_DAYS` (default 30 days).

Storage: ~216 MB per hour of activity, zero when idle.

## Development

```bash
# Interactive shell in the container
docker compose run --rm jetson-yolo bash

# Run tests (host)
./scripts/test_local.sh

# CLI Q&A
python -m app.tools.ask "How many people today?"

# Telegram bot standalone
python -m app.app.ask_telegram
```

The `ask-telegram` service mounts `./app` for live code edits without rebuilding the image.

## Migration from v1

If upgrading from the previous version:

**Remove** from `.env`:
```
LLM_PROVIDER
OPENAI_BASE_URL
TG_QA_POLL_TIMEOUT_SEC
TG_QA_IDLE_SLEEP_SEC
```

**Update** `LLM_MODEL` to include provider prefix:
```bash
# Before
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# After
LLM_MODEL=openai/gpt-4o-mini
```
