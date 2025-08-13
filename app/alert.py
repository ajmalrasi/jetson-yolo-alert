# alert.py — modular, event-based alerts with tracking (small main)
import os, time, logging, requests, cv2
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterable, Optional, Sequence, Tuple
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# ---------------------- config ----------------------
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
LOGGER.setLevel(logging.ERROR)

def ids_from_names(csv: str) -> set[int]:
    out = set()
    for s in csv.split(","):
        n = s.strip().lower()
        if n in NAME2ID: out.add(NAME2ID[n])
    return out

def parse_zone(spec: Optional[str]) -> Optional[list[tuple[int,int]]]:
    if not spec: return None
    try:
        pts = []
        for p in spec.split(";"):
            x,y = p.split(",")
            pts.append((int(float(x)), int(float(y))))
        return pts if len(pts) >= 3 else None
    except Exception:
        return None

def point_in_poly(px:int, py:int, poly:Sequence[Tuple[int,int]]) -> bool:
    inside = False
    for i in range(len(poly)):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
        hit = ((y1 > py) != (y2 > py)) and (px < (x2-x1)*(py-y1)/(y2-y1+1e-9) + x1)
        if hit: inside = not inside
    return inside

def resolve_model_path(p: str) -> str:
    if os.path.isabs(p) and os.path.exists(p): return p
    cand = os.path.join("/workspace/work", p)
    return cand if os.path.exists(cand) else p

def send_telegram(bot: str, chat: str, text: str, photo_path: Optional[str] = None):
    if not (bot and chat): return
    try:
        if photo_path and os.path.exists(photo_path):
            requests.post(
                f"https://api.telegram.org/bot{bot}/sendPhoto",
                data={"chat_id": chat, "caption": text},
                files={"photo": open(photo_path, "rb")},
                timeout=10
            )
        else:
            requests.post(
                f"https://api.telegram.org/bot{bot}/sendMessage",
                json={"chat_id": chat, "text": text},
                timeout=10
            )
    except requests.RequestException:
        pass

# ---------------------- state ----------------------
@dataclass
class Params:
    src: str = os.getenv("SRC", "0")
    bot: Optional[str] = os.getenv("TG_BOT") or os.getenv("TELEGRAM_TOKEN")
    chat: Optional[str] = os.getenv("TG_CHAT") or os.getenv("TELEGRAM_CHAT_ID")
    model: str = os.getenv("YOLO_ENGINE", "yolov8n.engine")
    conf: float = float(os.getenv("CONF_THRESH", "0.80"))
    vid_stride: int = int(os.getenv("VID_STRIDE", "6"))
    max_fps: float = float(os.getenv("MAX_FPS", "2"))
    max_fps_on_detect: float = float(os.getenv("MAX_FPS_ON_DETECT", "0"))  # 0 = unlimited
    rearm_time_on_detect: float = float(os.getenv("REARM_TIME_ON_DETECT", "5"))  # seconds to stay hot
    img_size: int = int(os.getenv("IMG_SIZE", "608"))
    tracker: str = os.getenv("TRACKER", "bytetrack.yaml")
    draw_ids: set[int] = field(default_factory=lambda: ids_from_names(os.getenv("DRAW_CLASSES", "person,car,dog,cat")))
    trig_ids: set[int] = field(default_factory=lambda: ids_from_names(os.getenv("TRIGGER_CLASSES", "person")))
    min_frames: int = int(os.getenv("MIN_FRAMES", "3"))
    min_persist: float = float(os.getenv("MIN_PERSIST_SEC", "1.0"))
    rearm_sec: float = float(os.getenv("REARM_SEC", "10"))
    rate_window: float = float(os.getenv("RATE_WINDOW_SEC", "5"))
    zone: Optional[list[tuple[int,int]]] = field(default_factory=lambda: parse_zone(os.getenv("ZONE")))

class TrackState:
    __slots__ = ("first_ts","last_ts","frames","alerted")
    def __init__(self, t: float):
        self.first_ts = t
        self.last_ts = t
        self.frames = 1
        self.alerted = False

class TrackManager:
    def __init__(self, p: Params):
        self.p = p
        self.tracks: dict[int, TrackState] = {}

    def update_and_get_entrants(self, xyxy, cls, conf, ids, now: float) -> tuple[list[int], float]:
        entrants, best = [], 0.0
        for i in range(len(xyxy)):
            if int(cls[i]) not in self.p.trig_ids or float(conf[i]) < self.p.conf:
                continue
            x1,y1,x2,y2 = xyxy[i]
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            if self.p.zone and not point_in_poly(cx, cy, self.p.zone):
                continue
            tid = int(ids[i])
            st = self.tracks.get(tid)
            if st is None:
                st = TrackState(now); self.tracks[tid] = st
            else:
                st.last_ts = now; st.frames += 1
            best = max(best, float(conf[i]))
            if (not st.alerted) and st.frames >= self.p.min_frames and (now - st.first_ts) >= self.p.min_persist:
                st.alerted = True
                entrants.append(tid)
        # prune old
        for tid in [t for t, st in self.tracks.items() if (now - st.last_ts) > self.p.rearm_sec]:
            self.tracks.pop(tid, None)
        return entrants, best

class Throttler:
    def __init__(self, window_sec: float):
        self.window = window_sec
        self.last_sent = 0.0
        self.pending_ids: set[int] = set()
        self.pending_best = 0.0

    def add(self, ids: Iterable[int], best_conf: float):
        if ids:
            self.pending_ids.update(ids)
            self.pending_best = max(self.pending_best, best_conf)

    def should_send(self, now: float) -> bool:
        return bool(self.pending_ids) and (now - self.last_sent) >= self.window

    def flush(self) -> tuple[int, float]:
        n, b = len(self.pending_ids), self.pending_best
        self.pending_ids.clear(); self.pending_best = 0.0; self.last_sent = time.time()
        return n, b

# ---------------------- pipeline ----------------------
def soft_fps_sleep(last_proc: float, max_fps: float) -> float:
    if max_fps <= 0: return time.time()
    min_dt = 1.0 / max_fps
    now = time.time()
    if (now - last_proc) < min_dt:
        time.sleep(min_dt - (now - last_proc))
    return time.time()

def draw_filtered_boxes(r, cls, conf, keep_ids: set[int], thr: float):
    try:
        keep = [i for i in range(len(cls)) if int(cls[i]) in keep_ids and float(conf[i]) >= thr]
        r.boxes = r.boxes[keep] if keep else r.boxes[:0]
    except Exception:
        pass
    return r.plot()

def iterate_tracks(p: Params):
    model = YOLO(resolve_model_path(p.model), task="detect")
    return model.track(
        source=p.src, stream=True, device=0, imgsz=p.img_size,
        vid_stride=p.vid_stride, verbose=False, tracker=p.tracker, persist=True
    )

def handle_frame(r, p: Params, tm: TrackManager, thr: Throttler, frame_path: str, last_proc: float) -> float:
    # Respect max FPS (soft cap)
    last_proc = soft_fps_sleep(last_proc, p.max_fps)

    # No detections in this frame?
    boxes = getattr(r, "boxes", None)
    if not boxes or len(boxes) == 0:
        return last_proc

    # tensors -> numpy
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
    conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
    cls  = boxes.cls.cpu().numpy()  if hasattr(boxes, "cls")  else None
    ids  = boxes.id.cpu().numpy()   if hasattr(boxes, "id") and boxes.id is not None else None
    if xyxy is None or conf is None or cls is None or ids is None:
        return last_proc

    now = time.time()
    if not hasattr(p, "_pending_path"):
        p._pending_path = "/workspace/work/alerts/pending.jpg"
    if not hasattr(p, "rearm_time"):
        p.rearm_time = 0.0

    # Track + decide who "entered" (meets min frames & persist)
    entrants, best = tm.update_and_get_entrants(xyxy, cls, conf, ids, now)
    had_pending = bool(thr.pending_ids) # track if this is the first entrant in the batch
    thr.add(entrants, best)

# If this is the first time we got entrants for this batch, capture that frame
    if entrants and not had_pending:
        img0 = draw_filtered_boxes(r, cls, conf, p.draw_ids, p.conf)
        cv2.imwrite(p._pending_path, img0)
    # Adaptive FPS: go high when we see entrants, fall back after a quiet period
    if entrants:
        p.max_fps = p.max_fps_on_detect            # temporarily speed up tracking
        p.rearm_time = now + p.rearm_time_on_detect    # keep high FPS for 5s after last entrant
    elif now > p.rearm_time:
        p.max_fps = p.max_fps             # idle FPS

    # IMPORTANT: always add to the throttler (was missed during high-FPS before)
    thr.add(entrants, best)

    # If we're due to send, snapshot + Telegram
    if thr.should_send(now):
       # prefer the saved detection-time frame; fall back to current
       use_path = p._pending_path if os.path.exists(p._pending_path) else frame_path
       if use_path == frame_path:
           img = draw_filtered_boxes(r, cls, conf, p.draw_ids, p.conf)
           cv2.imwrite(frame_path, img)
       n, b = thr.flush()
       send_telegram(p.bot, p.chat, f"Person alerts: {n} new (best {b:.2f})", use_path)
       # cleanup
       if os.path.exists(p._pending_path):
           try: os.remove(p._pending_path)
           except: pass

    return last_proc
# ---------------------- small main ----------------------
def main():
    p = Params()
    tm = TrackManager(p)
    thr = Throttler(p.rate_window)
    os.makedirs("/workspace/work/alerts", exist_ok=True)
    frame_path = "/workspace/work/alerts/frame.jpg"
    last_proc = 0.0
    for r in iterate_tracks(p):
        last_proc = handle_frame(r, p, tm, thr, frame_path, last_proc)

if __name__ == "__main__":
    main()
