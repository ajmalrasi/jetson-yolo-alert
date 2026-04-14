# Design

## Overview

A real-time object detection system on NVIDIA Jetson that sends Telegram alerts, captures frames during activity, and supports on-demand video understanding via cloud VLMs and natural language Q&A over alert history.

```
Camera --> YOLO/TensorRT --> Tracker --> Presence --> Alert --> Telegram
                               |                       |
                        FrameCaptureStep          SQLite alerts
                        (2fps on detection)            |
                               |                  LLM Q&A <-- /ask
                         Frame Store
                        (disk + SQLite)
                               |
                          VLM <-- /describe
```

## Architecture

### Ports and Adapters (Hexagonal)

Core logic has zero dependency on frameworks. All I/O goes through interfaces in `app/core/ports.py`:

```
app/core/ports.py        -- Interfaces (Camera, Detector, ITracker, AlertSink, EventBus, Telemetry)
app/core/pipeline.py     -- Pipeline engine (depends only on ports)
app/adapters/            -- Implementations (camera_cv2, detector_ultra, alerts_telegram, vlm_litellm, etc.)
```

Swapping a component (camera, detector, alert channel, VLM provider) means writing a new adapter.

### Detection Pipeline

Linear chain of steps, each receiving and returning a shared `Ctx` object:

```
RateStep --> ReadStep --> DetectStep --> FrameCaptureStep --> TriggerFilterStep --> PresenceStep --> AlertStep --> TelemetryStep
```

| Step | What it does |
|------|-------------|
| **RateStep** | Adaptive FPS -- slows when idle, speeds up on detection |
| **ReadStep** | Grabs a frame from the camera |
| **DetectStep** | Runs YOLO + optional tracker (BoT-SORT/ByteTrack) |
| **FrameCaptureStep** | Saves frames at 2fps when trigger classes detected; nothing when idle; 10s cooldown |
| **TriggerFilterStep** | Filters detections to configured trigger classes |
| **PresenceStep** | State machine: requires N frames + M seconds before confirming presence |
| **AlertStep** | Rate-limited alerts -- saves annotated snapshot (shared annotation module), writes to SQLite, sends to Telegram. Cooldown enforced by both pipeline clock and wall-clock to survive restarts |
| **TelemetryStep** | Gauges and timing metrics |

### Frame Capture and Storage

`FrameCaptureStep` piggybacks on existing YOLO detections to save frames to disk:

- **Idle**: save nothing (zero disk usage)
- **Active** (YOLO detects trigger classes): save at `CAPTURE_ACTIVE_FPS` (default 2 fps)
- **Cooldown**: keep saving for `CAPTURE_COOLDOWN_SEC` (default 10s) after last detection

`FrameStore` manages disk layout and SQLite index:

```
work/frames/
  2026-04-14/
    08/
      22-13-456.jpg
      22-14-012.jpg
      ...
  frame_index.db    -- ts, path, has_detection, detection_classes, detection_count, best_conf
```

### Video Understanding (VLM)

On-demand analysis of stored frames via cloud Vision-Language Models:

```
User: /describe what happened last night?
                |
        Text LLM parses time range ("last night" --> UTC boundaries)
                |
        FrameStore.query_range() loads matching frame records
                |
        Temporal clustering (gap > 60s = new event)
                |
        Pick best frame per cluster (highest confidence),
        top 5 clusters by (has_det, conf, count)
                |
        Build text timeline of ALL detection events
        (even those without an image slot)
                |
        Frames resized to 512px, base64 encoded
                |
        litellm.completion() sends images + timeline to VLM
                |
        Timestamped narrative returned to user
```

VLM adapter (`app/adapters/vlm_litellm.py`) uses `litellm` for provider-agnostic multimodal routing. The system prompt is enriched with a YOLO detection timeline covering all event clusters, so the VLM has context for events beyond the sampled images.

All `/describe` queries and VLM responses are logged to `{SAVE_DIR}/describe_trace.log` for debugging.

### Q&A System

LangGraph agent with SQL toolkit for alert history queries:

```
User: /ask how many people today?
                |
        LangGraph agent with sql_db_query tool
        (up to 6 iterations)
                |
        SQLite query on alert_history.db
                |
        Answer formatted and returned
```

### Telegram Integration

Three Telegram concerns, two processes:

1. **Alert sender** (`alerts_telegram.py`) -- fires from the detection pipeline (synchronous, in `alert` process)
2. **Q&A bot** (`chat_telegram_bot.py`) -- handles `/ask`, `/describe`, video uploads (async, `ask-telegram` process)
3. **Video uploads** -- bot downloads video, extracts frames, sends to VLM

They share the same bot token. The Q&A/VLM bot runs CPU-only to avoid GPU contention.

## Services (Docker Compose)

| Service | Runtime | Purpose |
|---------|---------|---------|
| `alert` | GPU | Detection pipeline + Telegram alerts + frame capture |
| `ask-telegram` | CPU | Telegram bot (`/ask` Q&A + `/describe` VLM + video uploads) |
| `exporter` | GPU | One-shot: converts .pt to .engine |
| `preview` | GPU | Live MJPEG stream / local display |
| `otel-collector` | CPU | OTLP metrics receiver (optional) |

## Database

Two SQLite databases:

**alert_history.db** -- written by alert pipeline, queried by `/ask`:

```sql
CREATE TABLE alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    count           INTEGER NOT NULL,
    best_conf       REAL NOT NULL,
    image_path      TEXT,
    trigger_classes TEXT DEFAULT '[]',
    context_classes TEXT DEFAULT '[]'
);
```

**frame_index.db** -- written by FrameCaptureStep, queried by `/describe`:

```sql
CREATE TABLE frames (
    ts                TEXT NOT NULL,
    path              TEXT NOT NULL,
    has_detection     INTEGER NOT NULL DEFAULT 0,
    detection_classes TEXT NOT NULL DEFAULT '[]',
    detection_count   INTEGER NOT NULL DEFAULT 0,
    best_conf         REAL NOT NULL DEFAULT 0.0
);
```

## LLM / VLM Provider Routing

**Text LLM** (`llm_litellm.py`): uses `ChatOpenAI` from LangChain pointed at provider endpoints:

```
groq/llama-3.3-70b-versatile  -->  api.groq.com/openai/v1
xai/grok-2-latest              -->  api.x.ai/v1
openai/gpt-4o-mini             -->  api.openai.com/v1
```

**Vision LLM** (`vlm_litellm.py`): uses `litellm.completion()` for multimodal routing:

```
openai/gpt-4o-mini                                -->  OpenAI vision API
groq/meta-llama/llama-4-scout-17b-16e-instruct    -->  Groq vision API (max 5 images)
gemini/gemini-2.0-flash                           -->  Google Gemini API
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Hexagonal architecture | Swap camera, detector, alert sink, VLM without touching core |
| Pipeline as linear steps | Easy to add/remove/reorder; each step independently testable |
| SQLite (not Postgres) | Single file, zero setup, works on Jetson |
| Separate frame capture from alerts | Frame store serves VLM; alert DB serves text Q&A; independent concerns |
| Save nothing when idle | Disk usage proportional to activity, not uptime |
| litellm for VLM routing | Provider-agnostic multimodal; already a dependency |
| LLM time-range parsing | Handles fuzzy natural language ("last night", "tuesday morning") |
| Separate ask-telegram service | CPU-only, no GPU contention with YOLO |
| YOLOv8m default | Better accuracy (50.2 mAP vs 37.3 for v8n); fast enough at adaptive FPS |

## File Map

```
app/
  core/
    ports.py                 -- Interfaces: Camera, Detector, AlertSink, etc.
    pipeline.py              -- Detection pipeline (step chain incl. FrameCaptureStep)
    config.py                -- Env-based configuration
    frame_store.py           -- SQLite + disk frame storage for VLM
    video_understanding.py   -- VideoUnderstandingService (time parsing, sampling, VLM)
    state.py                 -- Presence state machine
    presence_policy.py       -- Presence confirmation rules
    annotate.py              -- Shared bounding box / label drawing (preview + alert snapshots)
    alert_policy.py          -- Alert rate-limiting and grouping (wall-clock cooldown)
    rate_policy.py           -- Adaptive FPS policy
    clock.py                 -- Time abstraction (testable)
    events.py                -- Event types
    alert_history.py         -- SQLite alert read/write
    qa.py                    -- LangGraph Q&A service
    qa_factory.py            -- Wires QAService with DB + LLM
  adapters/
    camera_cv2.py            -- OpenCV camera (USB, RTSP, file)
    detector_ultra.py        -- Ultralytics YOLO detector
    alerts_telegram.py       -- Telegram alert sender
    chat_telegram_bot.py     -- Telegram bot (/ask, /describe, video uploads)
    llm_litellm.py           -- Text LLM provider routing
    vlm_litellm.py           -- Vision LLM provider routing (multimodal)
    telemetry_log.py         -- Logging-based telemetry
    mjpeg_stream.py          -- MJPEG HTTP server for preview
  app/
    run.py                   -- Main detection + alert + frame capture loop
    preview.py               -- Live video preview
    ask_telegram.py          -- Telegram bot entrypoint
  tools/
    export_engine.py         -- YOLO to TensorRT export
    ask.py                   -- CLI Q&A tool
```
