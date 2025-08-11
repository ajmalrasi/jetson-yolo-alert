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
ALLOWED = {0, 2, 15, 16}
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

def save_annotated(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = result.plot()          # full frame WITH boxes & labels
    cv2.imwrite(path, img)
    return path

def save_frame(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, result.orig_img)  # full frame, no boxes
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

        # select only allowed classes above threshold
        sel = [i for i, b in enumerate(boxes)
               if int(b.cls[0]) in ALLOWED and float(b.conf[0]) >= THRESH]

        if not sel:
            continue

        # send ONE annotated full-frame with only the selected boxes
        if (last_proc - last_alert) >= COOLDOWN:
            # keep only filtered boxes for plotting
            r.boxes = r.boxes[sel]
            os.makedirs("/workspace/work/alerts", exist_ok=True)
            frame_path = "/workspace/work/alerts/frame.jpg"
            # full frame WITH boxes for only allowed classes
            img = r.plot()
            cv2.imwrite(frame_path, img)

            # caption: count + best confidence
            best = max(float(boxes[i].conf[0]) for i in sel)
            send(f"Detections: {len(sel)} (best {best:.2f})", frame_path)
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
