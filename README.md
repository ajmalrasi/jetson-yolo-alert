# Jetson YOLOv8 TensorRT Pipeline

A Dockerized pipeline for running **YOLOv8 object detection** on NVIDIA Jetson devices using **TensorRT** for optimized inference.
Includes services for model export, live preview, and alerting (e.g., Telegram notifications).

---

## 📦 Requirements

* NVIDIA Jetson device (Orin, Xavier, Nano, etc.)
* JetPack 6.x with CUDA and TensorRT installed
* Docker & Docker Compose with NVIDIA runtime
* `.env` file containing your configuration (see below)

---

## ⚙️ Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `SRC` | `0` | Camera source (index or RTSP/HTTP URL). |
| `YOLO_MODEL` | `yolov8n.pt` | Original YOLO PyTorch model to export. |
| `YOLO_ENGINE` | `yolov8n.engine` | TensorRT engine file for inference. |
| `CONF_THRESH` | `0.60` | Detection confidence threshold. |
| `IMG_SIZE` | `640` | Inference image size. |
| `VID_STRIDE` | `1` | Process every Nth frame. |
| `BASE_FPS` | `5` | Idle FPS when nothing is detected. |
| `HIGH_FPS` | `0` (uncapped) | FPS when boosting after detection. |
| `BOOST_ARM_FRAMES` | `3` | Frames before boost triggers. |
| `BOOST_MIN_SEC` | `1.0` | Minimum presence before boost triggers. |
| `COOLDOWN_SEC` | `5.0` | FPS cooldown after object leaves the frame. |
| `TRIGGER_CLASSES` | `person,dog,cat` | Classes that trigger alerts. |
| `DRAW_CLASSES` | `person,car,dog,cat,motorcycle`| Classes to draw bounding boxes around in snapshots. |
| `MIN_FRAMES` | `3` | Frames required before registering presence. |
| `MIN_PERSIST_SEC` | `1.0` | Seconds required before registering presence. |
| `REARM_SEC` | `20` | Time before the exact same tracked object can trigger another alert. |
| `RATE_WINDOW_SEC` | `30` | Minimum time between sending Telegram alerts. |
| `TRACKER` | `botsort.yaml` | Tracker config (`botsort.yaml` or `bytetrack.yaml`). |
| `TRACKER_ON` | `1` | Enable tracking. |
| `TELEGRAM_TOKEN` | *(none)* | Telegram bot token (also accepts `TG_BOT`). |
| `TELEGRAM_CHAT_ID`| *(none)* | Telegram chat ID (also accepts `TG_CHAT`). |
| `SAVE_DIR` | `/workspace/work/alerts`| Directory to save alert snapshots. |
| `ALERT_DB_PATH` | `/workspace/work/alerts/alert_history.db` | SQLite file for alert history storage and Q&A. |
| `LLM_PROVIDER` | `none` | LLM backend for Q&A (`none` or `openai`). |
| `LLM_MODEL` | `gpt-4o-mini` | Model name used by the selected LLM provider. |
| `OPENAI_API_KEY` | *(none)* | OpenAI API key used when `LLM_PROVIDER=openai`. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL. |
| `DRAW` | `1` | Enable drawing bounding boxes on snapshots. |
| `USE_GSTREAMER` | `0` | Enable GStreamer backend instead of FFmpeg for RTSP. |
| `RTSP_LATENCY_MS` | `200` | Latency buffer for RTSP streams. |
| `CAP_PROP_BUFFERSIZE` | `2` | OpenCV capture buffer size. |

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
* **Configurable via `.env`** — Tune FPS, confidence thresholds, object classes, and camera backends.

---

## 📂 Project Structure

```
- **app/core/** – Core logic: interfaces (ports), state machines, detection/alert policies, configuration, and pipeline.
- **app/adapters/** – Implementations of interfaces for camera input, detection (YOLO/TensorRT), tracking, alert delivery, and telemetry.
- **app/app/** – Executable scripts for running the system, live preview, and pipeline.
- **app/tools/** – Tools like exporting YOLO models to TensorRT engines.
- **tests/** – Unit and integration tests for policies and overall behavior.
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

---

## 📜 License

MIT — feel free to modify and adapt.