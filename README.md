# Jetson YOLO Alert + Video Understanding

Real-time object detection on NVIDIA Jetson with Telegram alerts and on-demand video understanding via cloud Vision-Language Models.

## What It Does

- **Detects objects** (people, animals, vehicles) using YOLOv8 + TensorRT on Jetson
- **Sends Telegram alerts** with annotated snapshots when trigger classes appear
- **Captures frames** during detections at 2 fps, indexed for fast retrieval
- **Describes what happened** on demand -- ask about any past time period and get a narrative from a cloud VLM
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

- `/describe last night` -- VLM describes what the camera saw
- `/describe last 30 minutes` -- recent activity
- `/describe yesterday afternoon` -- any natural language time reference
- Send a video file -- bot analyzes and describes the video
- `/ask how many people today?` -- SQL-based alert history query
- `/ask any dogs this week?` -- works for any alert-history question

## How It Works

```mermaid
flowchart LR
    cam["📷 Camera\nUSB / RTSP"]:::input

    subgraph jetson [🟢 Jetson - always running]
        yolo["🧠 YOLOv8\nTensorRT"]:::detect
        track["🔍 Track + Confirm\npresence"]:::detect
        alert["🔔 Telegram Alert\n+ snapshot"]:::alertNode
        alertdb[("📊 Alert DB\nSQLite")]:::store
        frames[("🗂️ Frame Store\ndisk + SQLite")]:::store
    end

    subgraph cloud [☁️ Cloud - on demand]
        vlm["✨ Vision LLM\nGPT-4o / Groq / Gemini"]:::vlmNode
    end

    tg["💬 Telegram"]:::tgNode

    cam --> yolo
    yolo -->|"detection"| track --> alert --> tg
    alert --> alertdb
    yolo -->|"2 fps"| frames
    tg -->|"/describe\nlast night"| frames
    frames -->|"best ~5 frames\n+ event timeline"| vlm
    vlm -->|"narrative"| tg
    tg -->|"/ask how\nmany people?"| alertdb

    classDef input fill:#4a9eff,stroke:#2d7dd2,color:#fff
    classDef detect fill:#34d399,stroke:#059669,color:#fff
    classDef alertNode fill:#fb923c,stroke:#ea580c,color:#fff
    classDef store fill:#a78bfa,stroke:#7c3aed,color:#fff
    classDef vlmNode fill:#f472b6,stroke:#db2777,color:#fff
    classDef tgNode fill:#38bdf8,stroke:#0284c7,color:#fff
```

## Configuration

All config is in `.env`. The essentials: `SRC` (camera), `YOLO_ENGINE`, `TRIGGER_CLASSES`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `LLM_MODEL` (for `/ask`), `VLM_MODEL` (for `/describe`).

Full reference: [docs/configuration.md](docs/configuration.md)

## Docs

- [docs/configuration.md](docs/configuration.md) -- all environment variables
- [docs/services.md](docs/services.md) -- Docker Compose services, setup, preview, telemetry
- [docs/DESIGN.md](docs/DESIGN.md) -- architecture and design decisions
- [docs/metrics.md](docs/metrics.md) -- telemetry metrics and Grafana
- [docs/AGENTS.md](docs/AGENTS.md) -- QA trace debugging

## License

MIT
