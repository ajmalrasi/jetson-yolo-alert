# Jetson YOLO Alert

Real-time object/person detection and alert system for NVIDIA Jetson devices (or any system with GPU), using YOLO (Ultralytics) + TensorRT for high performance.  
Supports detection rate control, presence-based FPS boost, and Telegram alerts with snapshots.

---

## Features
- **TensorRT acceleration** for YOLO models (`yolov8n` default).
- **Presence policy**: boosts FPS when objects of interest are detected.
- **Rate policy**: configurable frame rate for idle, boost, and cooldown modes.
- **Alert policy**: batches alerts within a configurable window.
- **Telegram integration**: sends annotated snapshots when objects are detected.
- **Configurable via environment variables** (no hardcoding).

---

## Requirements
- NVIDIA Jetson (Nano, Xavier, Orin, etc.) or any CUDA-capable GPU.
- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- Requests

---

## Installation

Clone this repo:

```bash
git clone https://github.com/yourusername/jetson-yolo-alert.git
cd jetson-yolo-alert
```
## Install dependencies:
```bash
pip install -r requirements.txt
```

## Exporting YOLO model to TensorRT

```bash
YOLO_MODEL=yolov8n.pt YOLO_ENGINE=yolov8n.engine python -m app.export_engine
```

- **`YOLO_MODEL`**: Path/name of YOLOv8 model (`.pt`).  
- **`YOLO_ENGINE`**: Output TensorRT engine file name.  

The `.engine` file will be saved to `/workspace/work/`.


## Running the alert pipeline

```bash
SRC=0 YOLO_ENGINE=yolov8n.engine TG_BOT=<bot_token> TG_CHAT=<chat_id> python -m app.run
```

**Key env vars**:

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



### Example: Run with USB camera and Telegram alerts

```bash
YOLO_ENGINE=yolov8n.engine \
SRC=0 \
CONF_THRESH=0.75 \
TG_BOT=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11 \
TG_CHAT=987654321 \
python -m app.run
```
### File Structure
```bash
app/
  core/           # Policies, pipeline, config, events, ports
  adapters/       # Camera, detector, tracker, alerts
  export_engine.py # Export YOLO → TensorRT
  run.py           # Main loop
```
### Docker Usage

```bash
docker build -t jetson-yolo-alert .
docker run --runtime nvidia --network host \
  -e SRC=0 \
  -e YOLO_ENGINE=yolov8n.engine \
  -e TG_BOT=... \
  -e TG_CHAT=... \
  jetson-yolo-alert
```

