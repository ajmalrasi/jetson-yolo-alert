import numpy as np
from app.core.ports import Camera, Detector, Detection, Frame
from app.core.clock import SystemClock
from app.core.config import Config
from app.core.presence_policy import PresencePolicy
from app.core.rate_policy import RatePolicy
from app.core.alert_policy import AlertPolicy
from app.core.pipeline import Pipeline


class FakeCamera(Camera):
    def __init__(self):
        self.t = 0.0
        self.i = 0
    def open(self): pass
    def grab(self):
        self.t += 0.2
        return True
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


def _make_cfg(tmp_path):
    cfg = Config()
    cfg.save_dir = str(tmp_path / "alerts")
    cfg.raw_frames_dir = str(tmp_path / "raw")
    cfg.alert_db_path = str(tmp_path / "alert_history.db")
    return cfg


def test_smoke_runs_for_a_few_frames(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIGGER_CLASSES", "person")
    monkeypatch.setenv("DRAW_CLASSES", "person")
    monkeypatch.setenv("CONF_THRESH", "0.8")
    monkeypatch.setenv("DRAW", "0")
    monkeypatch.setenv("REARM_SEC", "10")
    monkeypatch.setenv("RATE_WINDOW_SEC", "1")
    cfg = _make_cfg(tmp_path)
    cam = FakeCamera()
    det = FakeDetector()
    det.labels = ["person"]
    pipe = Pipeline(
        cfg=cfg,
        clock=SystemClock(),
        camera=cam,
        detector=det,
        tracker=None,
        presence=PresencePolicy(min_frames=3, min_persist_sec=0.5),
        rate=RatePolicy(base_fps=2, high_fps=20, boost_arm_frames=3, boost_min_sec=0.5, cooldown_sec=3, base_stride=2),
        alerts=AlertPolicy(window_sec=1.0),
        sink=NullSink(),
        telemetry=NullTel(),
    )
    iters = 0
    for ctx in pipe.iter_frames():
        iters += 1
        if iters >= 12:
            break
    assert iters == 12
