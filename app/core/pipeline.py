from dataclasses import dataclass
from typing import Optional, Sequence
import cv2

from .ports import Detection, Frame
from .state import PresenceState
from .rate_policy import RateTarget

@dataclass
class Ctx:
    frame: Optional[Frame] = None
    dets: Sequence[Detection] = ()
    trigger_dets: Sequence[Detection] = ()
    now: float = 0.0
    target: RateTarget = RateTarget(fps=0.0, vid_stride=1)
    snapshot_path: Optional[str] = None

class Pipeline:
    def __init__(self, *, camera, detector, tracker, sink, clock, tel,
                 pres, rate, alerts,
                 draw_classes: set[str], conf_thresh: float, save_dir: str, draw: bool,
                 trigger_classes: set[str]):
        self.cam, self.det, self.trk, self.sink = camera, detector, tracker, sink
        self.clock, self.tel = clock, tel
        self.pres, self.rate, self.alerts = pres, rate, alerts
        self.draw_classes, self.conf, self.save_dir, self.draw = draw_classes, conf_thresh, save_dir, draw
        self.trigger_classes = trigger_classes
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        self.state = PresenceState()
        self.frame_idx = 0

        # read rearm from env via config (import avoided to keep deps clean)
        import os
        self.rearm_sec = float(os.getenv("REARM_SEC", "10"))

    def _soft_sleep(self, last_t: float, max_fps: float) -> float:
        if max_fps <= 0: return self.clock.now()
        min_dt = 1.0 / max_fps
        now = self.clock.now()
        dt = now - last_t
        if dt < min_dt: self.clock.sleep(min_dt - dt)
        return self.clock.now()

    def run(self):
        self.cam.open()
        last_proc = 0.0
        try:
            while True:
                last_proc = self._soft_sleep(last_proc, self.rate.base_fps if not self.state.present else (self.rate.high_fps or 0))
                frame = self.cam.read()
                if frame is None: continue
                self.frame_idx += 1
                now = self.clock.now()
                ts = f"{self.frame_idx}_{int(now*1000)}"

                # detect
                dets = self.det.detect(frame)
                if self.trk: dets = self.trk.update(frame, dets)

                # filter trigger classes + collect best conf
                trig = [d for d in dets if d.conf >= self.conf and d.cls_id in _COCO_NAME_TO_ID(self.trigger_classes)]
                best = max((d.conf for d in trig), default=0.0)

                # presence / rate
                self.state, became_present, _ = self.pres.update(self.state, now, trig)
                target = self.rate.decide(self.state, now)

                # snapshot when presence arms
                if became_present and self.draw:
                    path = os.path.join(self.save_dir, f"pending_{ts}.jpg")
                    _save_snapshot(path, frame, dets, self.draw_classes, self.conf)
                else:
                    path = None

                # alert only on session start (no periodic resend)
                if became_present:
                    enter_ids = [d.track_id for d in trig if d.track_id is not None] or [-1]
                    self.alerts.add(enter_ids, best, now, self.rearm_sec)

                if self.alerts.due(now) and self.sink:
                    use_path = path or os.path.join(self.save_dir, f"frame_{ts}.jpg")
                    if path is None and self.draw:
                        _save_snapshot(use_path, frame, dets, self.draw_classes, self.conf)
                    n, b = self.alerts.flush(now)
                    label = ",".join(sorted(self.trigger_classes)) or "object"
                    self.sink.send(f"{label} alerts: {n} new (best {b:.2f})", use_path)

                # telemetry
                self.tel.incr("frames")
                self.tel.gauge("present", 1 if self.state.present else 0)
                self.tel.gauge("fps_target", target.fps)
                last_proc = now
        finally:
            self.cam.close()

# ---- helpers ----
_COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"]
_NAME2ID = {n:i for i,n in enumerate(_COCO)}

def _COCO_NAME_TO_ID(names: set[str]) -> set[int]:
    return { _NAME2ID[n] for n in names if n in _NAME2ID }

def _save_snapshot(path: str, frame: Frame, dets: Sequence[Detection], draw_names: set[str], conf: float):
    try:
        img = frame.image.copy()
        keep = [d for d in dets if d.conf >= conf and d.cls_id in _COCO_NAME_TO_ID(draw_names)]
        for d in keep:
            x1,y1,x2,y2 = d.xyxy
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imwrite(path, img)
    except Exception:
        pass
