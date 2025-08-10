SHELL := /bin/bash

.PHONY: up sh build-engine preview alert down

up:
	docker compose up -d

sh:
	docker compose exec l4tml bash

build-engine: up
	# export YOLOv8n to TensorRT FP16
	docker compose exec l4tml python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.export(format='engine', half=True, device=0)
print('Exported yolov8n.engine')
PY

preview: up
	# live preview; add save=True if headless
	docker compose exec l4tml yolo predict source="$$SRC" model=yolov8n.engine device=0 stream=True

alert: up
	# Telegram alert loop (person @ >=0.80)
	docker compose exec -e SRC="$$SRC" -e TG_BOT="$$TG_BOT" -e TG_CHAT="$$TG_CHAT" l4tml python3 /workspace/alert.py

down:
	docker compose down -v