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


| Variable            | Default                      | Description |
|---------------------|------------------------------|-------------|
| `SRC`               | `0`                          | Camera source (index or RTSP/HTTP URL). |
| `YOLO_ENGINE`       | `yolov8n.engine`              | TensorRT engine file. |
| `CONF_THRESH`       | `0.80`                        | Detection confidence threshold. |
| `IMG_SIZE`          | `640`                         | Inference image size. |
| `VID_STRIDE`        | `6`                           | Process every Nth frame. |
| `BASE_FPS`          | `2`                           | Idle FPS. |
| `HIGH_FPS`          | `0` (uncapped)                | FPS when boosting. |
| `BOOST_ARM_FRAMES`  | `3`                           | Frames before boost triggers. |
| `BOOST_MIN_SEC`     | `2.0`                         | Minimum presence before boost. |
| `COOLDOWN_SEC`      | `5.0`                         | FPS cooldown after object leaves. |
| `TRIGGER_CLASSES`   | `person`                      | Classes that trigger presence. |
| `DRAW_CLASSES`      | `person,car,dog,cat`          | Classes to draw in snapshot. |
| `MIN_FRAMES`        | `3`                           | Frames required before presence. |
| `MIN_PERSIST_SEC`   | `1.0`                         | Seconds required before presence. |
| `RATE_WINDOW_SEC`   | `5`                           | Min time between alerts. |
| `TRACKER`           | `bytetrack.yaml`              | Tracker config. |
| `TRACKER_ON`        | `1`                           | Enable tracker. |
| `TG_BOT`            | *(none)*                      | Telegram bot token. |
| `TG_CHAT`           | *(none)*                      | Telegram chat ID. |
| `SAVE_DIR`          | `/workspace/work/alerts`      | Where to save snapshots. |
| `DRAW`              | `1`                           | Enable drawing boxes. |

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
- **core/** – Core logic: interfaces (ports), state machines, detection/alert policies, configuration, and pipeline.
- **adapters/** – Implementations of interfaces for camera input, detection (YOLO/TensorRT), tracking, alert delivery, and telemetry.
- **app/** – Executable scripts for running the system and exporting YOLO models to TensorRT engines.
- **tests/** – Unit and integration tests for policies and overall behavior.

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
