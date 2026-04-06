# Design

## Overview

A real-time object detection system on NVIDIA Jetson that sends Telegram alerts and supports natural language Q&A over alert history.

```
Camera → YOLO/TensorRT → Tracker → Presence → Alert → Telegram
                                                  ↓
                                               SQLite ← LLM Q&A ← Telegram /ask
```

---

## Architecture

### Ports & Adapters (Hexagonal)

Core logic has zero dependency on frameworks. All I/O goes through interfaces defined in `app/core/ports.py`:

```
app/core/ports.py        ← Interfaces (Camera, Detector, ITracker, AlertSink, EventBus, Telemetry)
app/core/pipeline.py     ← Pipeline engine (only depends on ports)
app/adapters/            ← Implementations (camera_cv2, detector_ultra, alerts_telegram, etc.)
```

Swapping a component (e.g. different camera, different alert channel) means writing a new adapter — no core changes.

### Detection Pipeline

The pipeline is a linear chain of steps, each receiving and returning a shared `Ctx` object:

```
RateStep → ReadStep → DetectStep → TriggerFilterStep → PresenceStep → AlertStep → TelemetryStep
```

| Step | What it does |
|---|---|
| **RateStep** | Adaptive FPS — slows down when idle, speeds up on detection |
| **ReadStep** | Grabs a frame from the camera |
| **DetectStep** | Runs YOLO + optional tracker (BoT-SORT/ByteTrack) |
| **TriggerFilterStep** | Filters detections to configured trigger classes |
| **PresenceStep** | State machine: requires N frames + M seconds before confirming presence |
| **AlertStep** | Rate-limited alerts — saves snapshot, writes to SQLite, sends to Telegram |
| **TelemetryStep** | Gauges (FPS target, presence, stride); pipeline timings are emitted earlier via `Telemetry.time_ms` (`read_ms`, `detect_ms`, `pipeline_loop_ms`, …) — see [metrics.md](metrics.md) |

### Q&A System

Lightweight Text-to-SQL with two LLM calls per question:

```
User question
    ↓
LLM call 1: "Generate a SQL query for this question" (~700 tokens)
    ↓
Execute SQL on SQLite (read-only, destructive queries blocked)
    ↓
LLM call 2: "Format these results as a human-readable answer" (~800 tokens)
    ↓
Answer sent back via Telegram
```

**Why not a LangChain agent?** The database has one table with 7 columns. An agent framework adds ~15,000 tokens of overhead per query (tool descriptions, ReAct reasoning, multi-turn loops). The two-call approach uses ~1,500 tokens and works within free-tier API limits.

### Telegram Integration

Two separate Telegram integrations:

1. **Alert sender** (`app/adapters/alerts_telegram.py`) — fires notifications from the detection pipeline. Synchronous, runs in the detection process.
2. **Q&A bot** (`app/adapters/chat_telegram_bot.py`) — handles `/ask` commands. Async, built with `python-telegram-bot`, runs as a separate service.

They share the same bot token but serve different purposes. The Q&A bot runs CPU-only to avoid GPU contention with detection.

---

## Services (Docker Compose)

| Service | Runtime | Purpose |
|---|---|---|
| `alert` | GPU (nvidia) | Detection pipeline + Telegram alerts |
| `ask-telegram` | CPU only | Telegram Q&A bot |
| `exporter` | GPU (nvidia) | One-shot: converts .pt → .engine |
| `preview` | GPU (nvidia) | Live detection overlay; optional local display, optional MJPEG browser UI (`PREVIEW_STREAM_PORT`), optional `PREVIEW_DETECTOR_ONLY` bench mode |
| `otel-collector` | CPU (optional profile) | Receives OTLP metrics from the app; exposes Prometheus scrape for Grafana |

---

## Database

Single SQLite file (`alert_history.db`) with one table:

```sql
CREATE TABLE alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,       -- ISO-8601 UTC
    count           INTEGER NOT NULL,    -- objects detected
    best_conf       REAL NOT NULL,       -- highest confidence
    image_path      TEXT,                -- snapshot path
    trigger_classes TEXT DEFAULT '[]',   -- JSON array: ["person","car"]
    context_classes TEXT DEFAULT '[]'    -- JSON array: all classes in frame
);
```

Written by the detection pipeline, queried by the Q&A system.

---

## LLM Provider Routing

`app/adapters/llm_litellm.py` routes to any OpenAI-compatible provider based on the `LLM_MODEL` string prefix:

```
groq/llama-3.1-8b-instant  →  api.groq.com/openai/v1
xai/grok-2-latest           →  api.x.ai/v1
openai/gpt-4o-mini          →  api.openai.com/v1  (default)
```

Uses `ChatOpenAI` from LangChain pointed at the provider's endpoint. API keys read from `GROQ_API_KEY`, `XAI_API_KEY`, `OPENAI_API_KEY`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Hexagonal architecture (ports/adapters) | Swap camera, detector, alert sink, or LLM without touching core logic |
| Pipeline as linear steps | Easy to add/remove/reorder steps; each step is independently testable |
| SQLite (not Postgres) | Single file, zero setup, works on Jetson with no server process |
| Two-call Text-to-SQL (not agent) | ~1,500 tokens vs ~15,000. Fits free-tier LLM APIs, faster response |
| Separate `ask-telegram` service | CPU-only — no GPU contention with YOLO inference |
| `python-telegram-bot` (async) | Non-blocking I/O, declarative routing, built-in error handling |
| Adaptive FPS | Saves power/thermals on Jetson when scene is empty |

---

## File Map

```
app/
├── core/                    # Business logic (no framework deps)
│   ├── ports.py             # Interfaces: Camera, Detector, AlertSink, etc.
│   ├── pipeline.py          # Detection pipeline (step chain)
│   ├── config.py            # Env-based configuration
│   ├── state.py             # Presence state machine
│   ├── presence_policy.py   # Presence confirmation rules
│   ├── alert_policy.py      # Alert rate-limiting and grouping
│   ├── rate_policy.py       # Adaptive FPS policy
│   ├── clock.py             # Time abstraction (testable)
│   ├── events.py            # Event types (PersonDetected, AlertIssued)
│   ├── alert_history.py     # SQLite read/write
│   ├── qa.py                # Text-to-SQL Q&A service
│   ├── qa_factory.py        # Wires QAService with DB + LLM
│   └── chat_commands.py     # Transport-agnostic command handler
├── adapters/                # I/O implementations
│   ├── camera_cv2.py        # OpenCV camera (USB, RTSP, file)
│   ├── detector_ultra.py    # Ultralytics YOLO detector
│   ├── detector_trt.py      # TensorRT detector
│   ├── tracker_x.py         # BoT-SORT / ByteTrack wrapper
│   ├── alerts_telegram.py   # Telegram alert sender
│   ├── chat_telegram_bot.py # Telegram Q&A bot (python-telegram-bot)
│   ├── llm_litellm.py       # LLM provider routing (Groq, xAI, OpenAI)
│   ├── telemetry_log.py     # Logging-based telemetry
│   └── event_bus_inproc.py  # In-process event bus
├── app/                     # Entrypoints
│   ├── run.py               # Main detection + alert loop
│   ├── preview.py           # Live video preview
│   ├── ask_telegram.py      # Telegram Q&A bot entrypoint
│   ├── commands.py          # CLI commands
│   ├── event_bus.py         # Event bus setup
│   └── rate_limit.py        # Rate limiter setup
└── tools/                   # One-shot utilities
    ├── export_engine.py     # YOLO → TensorRT export
    └── ask.py               # CLI Q&A tool
```
