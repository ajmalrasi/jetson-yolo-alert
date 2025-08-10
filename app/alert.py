import os, requests
from ultralytics import YOLO

SRC   = os.getenv("SRC", "0")
BOT   = os.getenv("TG_BOT")
CHAT  = os.getenv("TG_CHAT")
MODEL = os.getenv("YOLO_ENGINE", "yolov8n.engine")
THRESH= float(os.getenv("CONF_THRESH", "0.80"))

def send(msg, img=None):
    if not (BOT and CHAT): 
        return
    if img and os.path.exists(img):
        requests.post(f"https://api.telegram.org/bot{BOT}/sendPhoto",
                      data={"chat_id": CHAT, "caption": msg},
                      files={"photo": open(img, "rb")})
    else:
        requests.post(f"https://api.telegram.org/bot{BOT}/sendMessage",
                      json={"chat_id": CHAT, "text": msg})

def main():
    print(f"[alert] loading engine {MODEL}")
    m = YOLO(MODEL, task="detect")  # be explicit for TensorRT engines
    print(f"[alert] streaming from {SRC}")
    for r in m.predict(source=SRC, stream=True, device=0):
        if not hasattr(r, "boxes") or r.boxes is None: 
            continue
        for b in r.boxes:
            cls  = int(b.cls[0]); conf = float(b.conf[0])
            # COCO class 0 == person
            if cls == 0 and conf >= THRESH:
                os.makedirs("runs/alerts", exist_ok=True)
                frame_path = r.save(filename="runs/alerts/frame.jpg")
                send(f"Person detected @ {conf:.2f}", frame_path)

if __name__ == "__main__":
    main()
