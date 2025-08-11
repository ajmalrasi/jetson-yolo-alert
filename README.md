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

Create a `.env` file in the project root:

```env
# Model configuration
YOLO_MODEL=yolov8n.pt         # Base YOLOv8 model to export
YOLO_ENGINE=yolov8n.engine    # Name of the generated TensorRT engine
CONF_THRESH=0.80               # Detection confidence threshold
IMG_SIZE=608                   # Inference resolution

# Source stream (can be RTSP, USB, or file path)
SRC=rtsp://user:pass@camera-ip:554/stream

# Tracking
TRACKER=bytetrack.yaml
VID_STRIDE=6
MAX_FPS=2

# Triggering & Drawing
DRAW_CLASSES=person,car,dog,cat
TRIGGER_CLASSES=person
MIN_FRAMES=3
MIN_PERSIST_SEC=1.0
REARM_SEC=10
RATE_WINDOW_SEC=5

# Optional zone filtering (polygon points: x,y;x,y;...)
ZONE=

# Telegram alerting (optional)
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

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
* **Object Tracking** — Uses ByteTrack for persistent IDs.
* **Zone-based Triggers** — Alerts only for detections inside defined polygons.
* **Telegram Notifications** — Sends text + image snapshots of detections.
* **Configurable via `.env`** — Tune FPS, confidence thresholds, object classes, etc.

---

## 📂 Project Structure

```
.
├── app/
│   ├── alert.py           # Alerting & tracking service
│   ├── export_engine.py   # YOLOv8 → TensorRT export script
├── work/                  # Mounted workspace for models, logs, alerts
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Running the Pipeline

### 1️⃣ Build the Docker image

```bash
docker compose build
```

### 2️⃣ Export TensorRT Engine

```bash
docker compose run --rm exporter
```

### 3️⃣ Start Detection + Alert Services

```bash
docker compose up alert
```

### 4️⃣ Live Preview (Optional)

```bash
docker compose up preview
```

---

## 🔔 Alert Logic

* Tracks objects using **ByteTrack**.
* Only triggers if:

  * Object class is in `TRIGGER_CLASSES`.
  * Confidence ≥ `CONF_THRESH`.
  * Persists for at least `MIN_PERSIST_SEC` and `MIN_FRAMES` frames.
  * (Optional) Center point inside defined `ZONE`.
* Groups multiple detections into a single Telegram alert per `RATE_WINDOW_SEC`.

---

## 🛠️ Development & Testing

Run commands inside the container:

```bash
docker compose run --rm jetson-yolo bash
```

Example YOLOv8 prediction test:

```bash
yolo predict source="$SRC" model="$YOLO_ENGINE" device=0
```

---

## 📜 License

MIT — feel free to modify and adapt.
