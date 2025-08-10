# Jetson YOLOv8 TensorRT Pipeline

A Dockerized pipeline for running **YOLOv8 object detection** on NVIDIA Jetson devices using **TensorRT** for optimized inference.  
Includes services for model export, live preview, and alerting (e.g., Telegram notifications).

---

## 📦 Requirements

- NVIDIA Jetson device (Orin, Xavier, Nano, etc.)
- JetPack 6.x with CUDA and TensorRT installed
- Docker & Docker Compose with NVIDIA runtime
- `.env` file containing your configuration (see below)

---

## ⚙️ Environment Variables (`.env`)

Create a `.env` file in the project root:

```env
# Model configuration
YOLO_MODEL=yolov8n.pt         # Base YOLOv8 model to export
YOLO_ENGINE=yolov8n.engine    # Name of the generated TensorRT engine
CONF_THRESH=0.80               # Detection confidence threshold

# Source stream (can be RTSP, USB, or file path)
SRC=rtsp://user:pass@camera-ip:554/stream

# Telegram alerting (optional)
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
