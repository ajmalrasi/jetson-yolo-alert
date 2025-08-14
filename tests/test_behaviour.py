import types
import numpy as np
from app.core.ports import Camera, Detector, Detection, Frame
from app.core.clock import SystemClock
from app.core.presence_policy import PresencePolicy
from app.core.rate_policy import RatePolicy
from app.core.alert_policy import AlertPolicy
from app.core.pipeline import Pipeline

class FakeCamera(Camera):
    def __init__(self):
        self.t = 0.0
        self.i = 0
    def open(self): pass
    def read(self):
        self.t += 0.2; self.i += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        return Frame(image=img, t=self.t, index=self.i, w=640, h=480)
    def close(self): pass

class FakeDetector(Detector):
    def __init__(self): self.count = 0
    def detect(self, frame: Frame):
        self.count += 1
        if self.count < 5: return []
        return [Detection((100,100,200,200), 0.9, 0, track_id=1)]

class NullSink:
    def send(self, text, image_path=None): pass

class NullTel:
    def incr(self, *a, **k): pass
    def gauge(self, *a, **k): pass
    def time_ms(self, *a, **k): pass

def test_smoke_runs_for_a_few_frames(monkeypatch):
    cam = FakeCamera()
    det = FakeDetector()
    pipe = Pipeline(
        camera=cam, detector=det, tracker=None, sink=NullSink(),
        clock=SystemClock(), tel=NullTel(),
        pres=PresencePolicy(min_frames=3, min_persist_sec=0.5),
        rate=RatePolicy(base_fps=2, high_fps=20, boost_arm_frames=3, boost_min_sec=0.5, cooldown_sec=3, base_stride=2),
        alerts=AlertPolicy(window_sec=1.0),
        draw_classes={"person"}, conf_thresh=0.8, save_dir="/tmp", draw=False
    )
    # Run only a handful of iterations
    iters = 0
    orig_run = pipe.run
    def limited_run():
        nonlocal iters
        pipe.cam.open()
        try:
            while iters < 12:
                iters += 1
                # inline the top of loop to avoid sleeping in tests
                frame = pipe.cam.read()
                if frame is None: continue
                dets = pipe.det.detect(frame)
                # we don't validate more in this smoke test
        finally:
            pipe.cam.close()
    pipe.run = limited_run
    limited_run()
