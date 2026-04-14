# Configuration

All variables are set in `.env` (loaded by Docker Compose). See `.env.example` for a ready-to-copy template.

## Source / IO

| Variable | Default | Description |
|----------|---------|-------------|
| `SRC` | `0` | Camera source: device index (`0`), RTSP URL, HTTP URL, or file path |
| `YOLO_MODEL` | `yolov8m.pt` | PyTorch model file (used by exporter) |
| `YOLO_ENGINE` | `yolov8m.engine` | TensorRT engine file |
| `IMG_SIZE` | `640` | Inference image size |
| `CONF_THRESH` | `0.60` | Detection confidence threshold |
| `VID_STRIDE` | `1` | Process every Nth frame |
| `SAVE_DIR` | `/workspace/work/alerts` | Alert snapshot directory |
| `DRAW` | `1` | Enable bounding box drawing on snapshots |
| `SAVE_RAW_FRAMES` | `0` | Save clean (un-annotated) frames for fine-tuning |
| `RAW_FRAMES_DIR` | `{SAVE_DIR}/raw_frames` | Directory for raw frames |

## Detection / Tracking / Alerts

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIGGER_CLASSES` | `person,dog,cat` | Classes that trigger alerts and frame capture |
| `DRAW_CLASSES` | `person,car,dog,cat,motorcycle` | Classes to draw bounding boxes for |
| `MIN_FRAMES` | `3` | Frames before registering presence |
| `MIN_PERSIST_SEC` | `1.0` | Seconds before registering presence |
| `REARM_SEC` | `20` | Re-trigger cooldown per tracked object |
| `RATE_WINDOW_SEC` | `30` | Min time between Telegram alerts |
| `ALERT_COOLDOWN_SEC` | `150` | Min seconds between any two alerts (0 = use RATE_WINDOW_SEC only) |
| `TRACKER` | `botsort.yaml` | Tracker config (botsort.yaml or bytetrack.yaml) |
| `TRACKER_ON` | `1` | Enable object tracking |

## Adaptive Frame Rate

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_FPS` | `5` | Idle FPS (no detections) |
| `HIGH_FPS` | `0` | FPS during detection boost (0 = uncapped) |
| `BOOST_ARM_FRAMES` | `3` | Frames before boost triggers |
| `BOOST_MIN_SEC` | `1.0` | Min presence time before boost |
| `COOLDOWN_SEC` | `5.0` | Seconds to maintain high FPS after object leaves |

## Camera Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GSTREAMER` | `0` | Use GStreamer backend for RTSP (falls back to FFmpeg) |
| `RTSP_LATENCY_MS` | `200` | RTSP jitter buffer latency |
| `CAP_PROP_BUFFERSIZE` | `1` | OpenCV buffer size (lower = less lag) |
| `CAMERA_GRAB_FLUSH` | `0` | Discard N queued grabs before decode (reduces lag) |

## Telegram

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_TOKEN` | -- | Bot token (also accepts `TG_BOT`) |
| `TELEGRAM_CHAT_ID` | -- | Chat ID for alerts (also accepts `TG_CHAT`) |
| `TG_QA_ALLOWED_CHAT_ID` | -- | Restrict bot commands to this chat (defaults to TELEGRAM_CHAT_ID) |

## LLM Q&A (`/ask`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `none` | Provider/model for `/ask` text Q&A |
| `ALERT_DB_PATH` | `{SAVE_DIR}/alert_history.db` | SQLite alert database path |
| `GROQ_API_KEY` | -- | For `groq/` models |
| `OPENAI_API_KEY` | -- | For `openai/` models |
| `XAI_API_KEY` | -- | For `xai/` models |
| `QA_DEBUG` | `0` | Enable QA trace logging to `{SAVE_DIR}/qa_trace.log` |

Supported `LLM_MODEL` values:

| Provider | Model string | Key needed |
|----------|-------------|------------|
| Groq (free tier) | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Groq (faster) | `groq/llama-3.1-8b-instant` | `GROQ_API_KEY` |
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| xAI | `xai/grok-2-latest` | `XAI_API_KEY` |
| Ollama (local) | `ollama/llama3` | `OLLAMA_API_BASE` |

## Video Understanding (`/describe`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL` | `none` | Vision-Language Model for `/describe` |
| `VLM_MAX_FRAMES` | `15` | Max frames per VLM call |
| `VLM_MAX_WIDTH` | `512` | Resize width for VLM frames (saves tokens) |

Supported `VLM_MODEL` values:

| Provider | Model string | Key needed |
|----------|-------------|------------|
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| OpenAI (better) | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Groq | `groq/llama-4-scout-17b-16e-instruct` | `GROQ_API_KEY` |
| Gemini | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |

## Frame Capture (for `/describe`)

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMES_DIR` | `/workspace/work/frames` | Directory for captured frames |
| `FRAMES_RETENTION_DAYS` | `30` | Auto-delete frames older than this |
| `CAPTURE_ACTIVE_FPS` | `2` | Frames saved per second during active detections |
| `CAPTURE_COOLDOWN_SEC` | `10` | Seconds to keep saving after last detection |

Frames are only saved when YOLO detects trigger classes. Zero disk usage when idle.

## Preview (MJPEG Stream)

| Variable | Default | Description |
|----------|---------|-------------|
| `PREVIEW_STREAM_PORT` | -- | HTTP port for MJPEG stream (e.g. `8080`) |
| `PREVIEW_STREAM_BIND` | `0.0.0.0` | Listen address |
| `PREVIEW_STREAM_MAX_WIDTH` | `1280` | Max stream width |
| `PREVIEW_STREAM_QUALITY` | `82` | JPEG quality |
| `PREVIEW_STREAM_FPS` | `25` | Max MJPEG frame rate |
| `PREVIEW_USE_DISPLAY` | auto | `0` to disable local window even if DISPLAY is set |
| `PREVIEW_DETECTOR_ONLY` | `0` | `1` = full-speed preview, no alerts/DB/snapshots |

## Telemetry

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEMETRY_BACKEND` | `log` | `log` (stdout) or `otlp` (OpenTelemetry) |
| `TELEMETRY_LOG_LEVEL` | `INFO` | Log level for telemetry logger |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | -- | OTLP receiver URL (e.g. `http://127.0.0.1:4318`) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Must match your collector |
| `OTEL_SERVICE_NAME` | `jetson-yolo-alert` | Service name for OTLP resource |

See [metrics.md](metrics.md) for metric names and Grafana setup.
