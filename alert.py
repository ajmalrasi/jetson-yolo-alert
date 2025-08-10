import os, requests
from ultralytics import YOLO

SRC = os.getenv("SRC", "0")
BOT = os.getenv("TG_BOT")
CHAT= os.getenv("TG_CHAT")

def send(msg, img=None):
    if not (BOT and CHAT): return
    if img:
        requests.post(f"https://api.telegram.org/bot{BOT}/sendPhoto",
                      data={"chat_id": CHAT, "caption": msg},
                      files={"photo": open(img, "rb")})
    else:
        requests.post(f"https://api.telegram.org/bot{BOT}/sendMessage",
                      json={"chat_id": CHAT, "text": msg})

m = YOLO("yolov8n.engine")
for r in m.stream(source=SRC, stream=True, device=0):
    for b in r.boxes:
        cls = int(b.cls[0]); conf = float(b.conf[0])
        if cls == 0 and conf >= 0.80:  # 'person'
            r.save(filename="runs/alerts/frame.jpg")
            send(f"Person detected @ {conf:.2f}", "runs/alerts/frame.jpg")
