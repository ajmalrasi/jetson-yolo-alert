import os
import time
import logging
import requests
import cv2
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

# ---- class config via env (names, not IDs) ----
# default: draw person,car,dog,cat; trigger alerts only when person present
DRAW_CLASSES    = os.getenv("DRAW_CLASSES",    "person,car,dog,cat")
TRIGGER_CLASSES = os.getenv("TRIGGER_CLASSES", "person")

COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
NAME2ID = {n: i for i, n in enumerate(COCO80)}

def _parse_names(csv: str) -> set[int]:
    out = set()
    for s in csv.split(","):
        n = s.strip().lower()
        if n and n in NAME2ID:
            out.add(NAME2ID[n])
    return out

DRAW_IDS    = _parse_names(DRAW_CLASSES)
TRIGGER_IDS = _parse_names(TRIGGER_CLASSES)

# Quiet the Ultralytics logger
LOGGER.setLevel(logging.ERROR)

def send(msg: str, img: str | None = None) -> None:
    if not (BOT and CHAT):
        return
    try:
        if img and os.path.exists(img):
            requests.post(
                f"https://api.telegram.org/bot{BOT}/sendPhoto",
                data={"chat_id": CHAT, "caption": msg},
                files={"photo": open(img, "rb")},
                timeout=10,
            )
        else:
            requests.post(
                f"https://api.telegram.org/bot{BOT}/sendMessage",
                json={"chat_id": CHAT, "text": msg},
                timeout=10,
            )
    except requests.RequestException:
        pass  # stay quiet

def resolve_model_path(p: str) -> str:
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join("/workspace/work", p)
    return cand if os.path.exists(cand) else p

def run_once() -> None:
    model_path = resolve_model_path(MODEL)
    m = YOLO(model_path, task="detect")

    last_alert, last_proc = 0.0, 0.0
    min_dt = 0.0 if MAX_FPS <= 0 else 1.0 / MAX_FPS

    for r in m.predict(
        source=SRC,
        stream=True,
        device=0,
        imgsz=IMG_SIZE,
        vid_stride=VID_STRIDE,
        verbose=False,
    ):
        # soft FPS cap
        now = time.time()
        if min_dt and (now - last_proc) < min_dt:
            time.sleep(min_dt - (now - last_proc))
        last_proc = time.time()

        boxes = getattr(r, "boxes", None)
        if not boxes:
            continue

        # indices to draw / to trigger
        idx_draw = [
            i for i, b in enumerate(boxes)
            if int(b.cls[0]) in DRAW_IDS and float(b.conf[0]) >= THRESH
        ]
        idx_trig = [
            i for i, b in enumerate(boxes)
            if int(b.cls[0]) in TRIGGER_IDS and float(b.conf[0]) >= THRESH
        ]

        # only alert if a trigger class is present (default: person)
        if not idx_trig:
            continue

        if (last_proc - last_alert) >= COOLDOWN:
            # limit plotted boxes to just our chosen classes
            try:
                r.boxes = r.boxes[idx_draw] if idx_draw else r.boxes[:0]
            except Exception:
                # if slicing isn't supported, just leave all boxes (worst case)
                pass

            os.makedirs("/workspace/work/alerts", exist_ok=True)
            frame_path = "/workspace/work/alerts/frame.jpg"

            # full frame WITH filtered boxes & labels
            img = r.plot()
            cv2.imwrite(frame_path, img)

            # caption with count + best confidence among drawn boxes (fallback to trigger)
            if idx_draw:
                best = max(float(boxes[i].conf[0]) for i in idx_draw)
                count = len(idx_draw)
            else:
                best = max(float(boxes[i].conf[0]) for i in idx_trig)
                count = len(idx_trig)

            send(f"Detections: {count} (best {best:.2f})", frame_path)
            last_alert = last_proc

def main() -> None:
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
