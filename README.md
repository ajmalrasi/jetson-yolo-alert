# Jetson YOLOv8 TensorRT Pipeline

A Dockerized pipeline for running **YOLOv8 object detection** on NVIDIA Jetson devices using **TensorRT** for optimized inference.
Includes services for model export, live preview, alerting (Telegram notifications), and **LLM-powered natural-language Q&A** over alert history.

---

## 📦 Requirements

* NVIDIA Jetson device (Orin, Xavier, Nano, etc.)
* JetPack 6.x with CUDA and TensorRT installed
* Docker & Docker Compose with NVIDIA runtime
* `.env` file containing your configuration (see below)

---

## ⚙️ Environment Variables (`.env`)

All variables are configured in a single `.env` file. See `.env.example` for a complete template.

**Core**

| Variable | Default | Description |
|---|---|---|
| `SRC` | `0` | Camera source (index or RTSP/HTTP URL) |
| `YOLO_ENGINE` | `yolov8n.engine` | TensorRT engine file |
| `CONF_THRESH` | `0.60` | Detection confidence threshold |
| `TRIGGER_CLASSES` | `person,dog,cat` | Classes that trigger alerts |

**Telegram**

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_TOKEN` | *(none)* | Bot token (also accepts `TG_BOT`) |
| `TELEGRAM_CHAT_ID` | *(none)* | Chat ID for alerts (also accepts `TG_CHAT`) |
| `TG_QA_ALLOWED_CHAT_ID` | *(none)* | Restrict `/ask` to this chat. Defaults to `TELEGRAM_CHAT_ID` |

**LLM Q&A**

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `none` | Provider/model string (see **Supported LLM Providers**). `none` to disable |
| `GROQ_API_KEY` | *(none)* | For `groq/` models |
| `OPENAI_API_KEY` | *(none)* | For `openai/` models |
| `XAI_API_KEY` | *(none)* | For `xai/` models |

<details>
<summary><strong>All other variables</strong></summary>

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | PyTorch model to export |
| `IMG_SIZE` | `640` | Inference image size |
| `VID_STRIDE` | `1` | Process every Nth frame |
| `BASE_FPS` | `5` | Idle FPS |
| `HIGH_FPS` | `0` (uncapped) | FPS during detection boost |
| `BOOST_ARM_FRAMES` | `3` | Frames before boost triggers |
| `BOOST_MIN_SEC` | `1.0` | Min presence time before boost |
| `COOLDOWN_SEC` | `5.0` | FPS cooldown after object leaves |
| `DRAW_CLASSES` | `person,car,dog,cat,motorcycle` | Classes to draw bounding boxes for |
| `MIN_FRAMES` | `3` | Frames before registering presence |
| `MIN_PERSIST_SEC` | `1.0` | Seconds before registering presence |
| `REARM_SEC` | `20` | Re-trigger cooldown per tracked object |
| `RATE_WINDOW_SEC` | `30` | Min time between Telegram alerts |
| `TRACKER` | `botsort.yaml` | Tracker config |
| `TRACKER_ON` | `1` | Enable tracking |
| `SAVE_DIR` | `/workspace/work/alerts` | Alert snapshot directory |
| `ALERT_DB_PATH` | `/workspace/work/alerts/alert_history.db` | SQLite database path |
| `DRAW` | `1` | Enable bounding box drawing |
| `USE_GSTREAMER` | `0` | GStreamer backend for RTSP |
| `RTSP_LATENCY_MS` | `200` | RTSP latency buffer |
| `CAP_PROP_BUFFERSIZE` | `2` | OpenCV capture buffer size |

</details>

---

## 🤖 Supported LLM Providers

The Q&A system supports any OpenAI-compatible LLM provider through a single `LLM_MODEL` variable.

| Provider | `LLM_MODEL` value | Required env var | Notes |
|---|---|---|---|
| Groq (free tier) | `groq/llama-3.1-8b-instant` | `GROQ_API_KEY` | Recommended -- fast, free |
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` | Paid |
| xAI (Grok) | `xai/grok-2-latest` | `XAI_API_KEY` | Paid |
| Ollama (local) | `ollama/llama3` | `OLLAMA_API_BASE` (e.g. `http://localhost:11434`) | Runs on device CPU |

Switch providers by changing one line in `.env`:

```bash
LLM_MODEL=groq/llama-3.1-8b-instant   # fast Groq cloud (free tier)
LLM_MODEL=openai/gpt-4o-mini          # OpenAI cloud (paid)
LLM_MODEL=xai/grok-2-latest           # xAI Grok (paid)
```

**How Q&A works** -- a lightweight two-step Text-to-SQL approach:
1. LLM generates a SQL query from your natural language question (~700 tokens)
2. Query runs against the local SQLite database
3. LLM formats the raw results into a human-readable answer (~800 tokens)

Total: ~1,500 tokens per question. Destructive queries (DROP, DELETE, etc.) are blocked. All timestamps are displayed in IST.

---

## 🐍 Python Dependencies (for non-Docker testing)

```bash
pip install -r requirements.txt
```

---

## 🚀 Features

* **Optimized Inference** — YOLOv8 → TensorRT FP16 engine for NVIDIA Jetson (Orin, Xavier, Nano, etc.).
* **Modular Services**:
  * **`exporter`** — Converts YOLOv8 PyTorch model to TensorRT.
  * **`jetson-yolo`** — Base container for running detection or CLI commands.
  * **`preview`** — Live stream overlay (RTSP, USB, file).
  * **`alert`** — Event-based object detection alerts with tracking & throttling.
* **Object Tracking** — Uses BoT-SORT or ByteTrack for persistent IDs.
* **Adaptive Framerate** — Dynamically adjusts processing FPS to save power when the scene is empty.
* **Telegram Notifications** — Sends text + snapshot images exactly at the moment of detection.
* **LLM-powered Q&A** — Ask natural-language questions about alert history via Telegram `/ask` or CLI. Uses a lightweight Text-to-SQL approach (~1,500 tokens/query) with multi-provider support (Groq, OpenAI, xAI, Ollama).
* **Async Telegram Bot** — Built with [python-telegram-bot](https://python-telegram-bot.org/) for non-blocking I/O, declarative command routing, and graceful shutdown.
* **Configurable via `.env`** — Tune FPS, confidence thresholds, object classes, and camera backends.

---

## 📂 Project Structure

```
- **app/core/** – Core logic: interfaces (ports), state machines, detection/alert policies, configuration, QA service, and pipeline.
- **app/adapters/** – Implementations of interfaces for camera input, detection (YOLO/TensorRT), tracking, alert delivery, LLM client, Telegram bot, and telemetry.
- **app/app/** – Executable scripts for running the system, live preview, and Telegram Q&A bot.
- **app/tools/** – Tools like exporting YOLO models to TensorRT engines, and CLI Q&A.
- **tests/** – Unit and integration tests for policies, QA service, and overall behavior.
```

---

## 🐳 Running the Pipeline

### 1️⃣ Build the Docker image

```bash
docker compose build
```

### 2️⃣ Export TensorRT Engine

If you haven't exported an engine file yet, generate one optimized for your Jetson:
```bash
docker compose run --rm exporter
```

### 3️⃣ Start Detection + Alert Services

Run the system in the background:
```bash
docker compose up -d alert
```

Run detection + Telegram Q&A bot together:
```bash
docker compose up -d alert ask-telegram
```

### 4️⃣ Live Preview (Optional)

If you have a monitor connected (or X11 forwarding), you can view a live debug feed:
```bash
docker compose up preview
```

---

## 🔔 Alert Logic

* Tracks objects using **BoT-SORT / ByteTrack**.
* The pipeline registers a "Presence" and queues an alert only if:
  * Object class is in `TRIGGER_CLASSES`.
  * Confidence ≥ `CONF_THRESH`.
  * Persists for at least `MIN_PERSIST_SEC` (seconds) AND `MIN_FRAMES` (frames).
* When an alert is triggered, it takes a snapshot of the exact moment the object met the conditions.
* Uses **`RATE_WINDOW_SEC`** to prevent spamming notifications (e.g. max 1 Telegram message every 30 seconds). Pending alerts are grouped and flushed when the window expires.
* Uses **`REARM_SEC`** to throttle re-triggering for the exact same object/tracking ID if it lingers in the frame.

---

## 🛠️ Development & Testing

Run commands inside the base container interactively:

```bash
docker compose run --rm jetson-yolo bash
```

Run unit tests locally (host) using the fixed venv path:

```bash
./scripts/test_local.sh
```

Equivalent Make target:

```bash
make test
```

Notes:
* Uses `/home/ajmalrasi/jetson/bin/python`.
* Sets `SAVE_DIR=/tmp` by default to avoid host permission issues from `/workspace/...` defaults.

Example YOLOv8 prediction test:

```bash
yolo predict source="$SRC" model="$YOLO_ENGINE" device=0
```

Ask alert-history questions from CLI:

```bash
python -m app.tools.ask "How many people came on 2026-03-10?"
python -m app.tools.ask "When was the last alert?"
```

Run Telegram Q&A bot (`/ask ...`) using the same QA service:

```bash
python -m app.app.ask_telegram
```

Or via Docker Compose:

```bash
docker compose up -d ask-telegram
```

`ask-telegram` is CPU-only and does not need GPU runtime. This avoids contention with the detector service.

This is transport-modular: Telegram is only an adapter. You can add a WhatsApp adapter later and reuse the same QA service.

---

## 🔄 Migration from v1

If you are upgrading from the previous version (hand-rolled LLM client, manual Text-to-SQL, raw Telegram polling), follow these steps:

### `.env` changes

**Remove** these variables (no longer used):
```
LLM_PROVIDER
OPENAI_BASE_URL
TG_QA_POLL_TIMEOUT_SEC
TG_QA_IDLE_SLEEP_SEC
```

**Update** `LLM_MODEL` to include the provider prefix:
```bash
# Before
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# After
LLM_MODEL=openai/gpt-4o-mini
```

### What changed

| Component | Before | After |
|---|---|---|
| LLM client | Hand-rolled `requests.post` to OpenAI API | `ChatOpenAI` routed to provider-specific endpoints (Groq, xAI, OpenAI) |
| Text-to-SQL | Manual prompt engineering, regex SQL extraction, 1 retry | Lightweight 2-call approach: LLM generates SQL, executes, LLM formats answer (~1,500 tokens) |
| Telegram bot | Raw `getUpdates` polling loop with `requests` | **python-telegram-bot** — async `Application` with `CommandHandler`, graceful shutdown |
| Config | 4 LLM vars + 2 polling tunables | 1 model string (`LLM_MODEL`) |

### Files removed
* `app/adapters/llm_openai.py` — replaced by `app/adapters/llm_litellm.py`
* `app/adapters/chat_telegram_polling.py` — replaced by `app/adapters/chat_telegram_bot.py`

---

## 📜 License

MIT — feel free to modify and adapt.
