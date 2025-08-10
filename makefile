SHELL := /bin/bash

.PHONY: up sh build-engine preview alert down

up:
\tdocker compose up -d

sh:
\tdocker compose exec l4tml bash

build-engine: up
\t# export YOLOv8n to TensorRT FP16
\tdocker compose exec l4tml python3 - <<'PY'\nfrom ultralytics import YOLO\nm=YOLO('yolov8n.pt')\nm.export(format='engine', half=True, device=0)\nprint('Exported yolov8n.engine')\nPY

preview: up
\t# live preview; add save=True if headless
\tdocker compose exec l4tml yolo predict source=\"$$SRC\" model=yolov8n.engine device=0 stream=True

alert: up
\t# Telegram alert loop (person @ >=0.80)
\tdocker compose exec -e SRC=\"$$SRC\" -e TG_BOT=\"$$TG_BOT\" -e TG_CHAT=\"$$TG_CHAT\" l4tml python3 /workspace/alert.py

down:
\tdocker compose down -v
