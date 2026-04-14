# Jetson YOLO Alert + Video Understanding

Real-time object detection on NVIDIA Jetson with Telegram alerts and on-demand video understanding via cloud Vision-Language Models.

## What It Does

- **Detects objects** (people, animals, vehicles) using YOLOv8 + TensorRT on Jetson
- **Sends Telegram alerts** with annotated snapshots when trigger classes appear
- **Captures frames** during detections at 2 fps, indexed in SQLite for fast retrieval
- **Describes what happened** on demand -- ask natural language questions about any past time period and get a narrative from a cloud VLM
- **Answers alert history questions** via `/ask` (Text-to-SQL over the alert database)

## Quick Start

```bash
# 1. Copy and edit config
cp .env.example .env
# Edit .env: set SRC, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, API keys

# 2. Build
docker compose build

# 3. Export TensorRT engine (once per device)
docker compose run --rm exporter

# 4. Run
docker compose up -d alert ask-telegram
```

## Telegram Commands

| Command | What it does |
|---------|-------------|
| `/describe last night` | VLM describes what the camera saw last night |
| `/describe last 30 minutes` | Describes recent activity |
| `/describe yesterday afternoon` | Any natural language time reference works |
| Send a video file | Bot analyzes and describes the video |
| `/ask how many people today?` | SQL-based query over alert history |
| `/ask any dogs this week?` | Works for any alert-history question |

## How It Works

```
Camera --> YOLOv8/TensorRT --> Detection?
                                  |
                    yes: save frame at 2fps + alert pipeline
                    no:  skip (save nothing)
                                  |
                          Alert pipeline:
                          - Track objects (BoT-SORT/ByteTrack)
                          - Confirm presence (N frames + M seconds)
                          - Rate-limit and send Telegram alert
                          - Write to SQLite alert DB
                          - Save annotated snapshot
                                  |
                          Frame store:
                          - JPEG to disk (work/frames/YYYY-MM-DD/HH/)
                          - Metadata indexed in SQLite (frame_index.db)
                                  |
User asks /describe  -->  Load frames for time range
                          Sample ~15 representative frames
                          Send to cloud VLM (GPT-4o, Groq, Gemini)
                          Return timestamped narrative
```

## Configuration

All config is via environment variables in `.env`. Key settings:

| Variable | Default | What |
|----------|---------|------|
| `SRC` | `0` | Camera source (device index, RTSP URL, or file path) |
| `YOLO_ENGINE` | `yolov8m.engine` | TensorRT engine file |
| `TRIGGER_CLASSES` | `person,dog,cat` | Classes that trigger alerts + frame capture |
| `TELEGRAM_TOKEN` | -- | Telegram bot token |
| `TELEGRAM_CHAT_ID` | -- | Chat ID for alerts |
| `LLM_MODEL` | `none` | Text LLM for `/ask` (e.g. `groq/llama-3.3-70b-versatile`) |
| `VLM_MODEL` | `none` | Vision LLM for `/describe` (e.g. `openai/gpt-4o-mini`) |

See [docs/configuration.md](docs/configuration.md) for all variables.

## Docker Compose Services

| Service | Purpose | GPU |
|---------|---------|-----|
| `alert` | Detection pipeline + Telegram alerts + frame capture | Yes |
| `ask-telegram` | Telegram bot (`/ask` + `/describe` + video uploads) | No |
| `exporter` | One-shot: converts .pt to .engine | Yes |
| `preview` | Live MJPEG stream / local display | Yes |

See [docs/services.md](docs/services.md) for detailed setup and run order.

## Documentation

| Doc | Content |
|-----|---------|
| [docs/configuration.md](docs/configuration.md) | All environment variables |
| [docs/services.md](docs/services.md) | Docker Compose services, run order, preview, telemetry |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture, pipeline steps, design decisions |
| [docs/metrics.md](docs/metrics.md) | Telemetry metrics and Grafana setup |
| [docs/AGENTS.md](docs/AGENTS.md) | QA trace debugging for `/ask` |

## Development

```bash
# Interactive shell
docker compose run --rm jetson-yolo bash

# Run tests
./scripts/test_local.sh

# CLI Q&A
python -m app.tools.ask "How many people today?"
```

## License

MIT
