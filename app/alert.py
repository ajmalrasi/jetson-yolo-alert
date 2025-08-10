import os, time, logging, requests, cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# ---- env knobs (no code edits needed later) ----
SRC         = os.getenv("SRC", "0")
BOT         = os.getenv("TG_BOT"); CHAT = os.getenv("TG_CHAT")
MODEL       = os.getenv("YOLO_ENGINE", "yolov8n.engine")
THRESH      = float(os.getenv("CONF_THRESH", "0.80"))
COOLDOWN    = float(os.getenv("COOLDOWN_SEC", "10"))   # min seconds between alerts
VID_STRIDE  = int(os.getenv("VID_STRIDE", "4"))        # process every Nth frame
MAX_FPS     = float(os.getenv("MAX_FPS", "3"))         # 0 = unlimited
IMG_SIZE    = int(os.getenv("IMG_SIZE", "640"))        # inference size

# Quiet the Ultralytics logger
LOGGER.setLevel(logging.ERROR)

def send(msg, img=None):
    if not (BOT and CHAT): return
    try:
        if img and os.path.exists(img):
            requests.post(f"https://api.telegram.org/bot{BOT}/sendPhoto",
                          data={"chat_id": CHAT, "caption": msg},
                          files={"photo": open(img, "rb")}, timeout=10)
        else:
            requests.post(f"https://api.telegram.org/bot{BOT}/sendMessage",
                          json={"chat_id": CHAT, "text": msg}, timeout=10)
    except requests.RequestException:
        pass  # stay quiet

def save_crop(result, box, path):
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    h, w = result.orig_img.shape[:2]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, result.orig_img[y1:y2, x1:x2])
    return path

def run_once():
    # Resolve engine path relative to /workspace/work as well
    model_path = MODEL if os.path.isabs(MODEL) else (
        os.path.join("/workspace/work", MODEL) if os.path.exists(os.path.join("/workspace/work", MODEL)) else MODEL
    )
    m = YOLO(model_path, task="detect")

    last_alert, last_proc = 0.0, 0.0
    min_dt = 0.0 if MAX_FPS <= 0 else 1.0 / MAX_FPS

    for r in m.predict(source=SRC, stream=True, device=0, imgsz=IMG_SIZE, vid_stride=VID_STRIDE, verbose=False):
        # soft FPS cap
        now = time.time()
        if min_dt and (now - last_proc) < min_dt:
            time.sleep(min_dt - (now - last_proc))
        last_proc = time.time()

        boxes = getattr(r, "boxes", None)
        if not boxes: 
            continue

        for b in boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            if cls == 0 and conf >= THRESH:   # person only
                if (last_proc - last_alert) >= COOLDOWN:
                    thumb = "/workspace/work/alerts/alert.jpg"
                    save_crop(r, b, thumb)
                    send(f"Person {conf:.2f}", thumb)
                    last_alert = last_proc

def main():
    # reconnect/backoff loop to survive camera/network hiccups
    backoff = 2.0
    while True:
        try:
            run_once()
            backoff = 2.0
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)

if __name__ == "__main__":
    main()