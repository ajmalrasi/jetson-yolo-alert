"""Tests for pipeline step ordering and FrameCaptureStep reusing ctx.trigger_dets."""
import numpy as np

from app.core.ports import Detection, Frame
from app.core.config import Config
from app.core.clock import SystemClock
from app.core.presence_policy import PresencePolicy
from app.core.rate_policy import RatePolicy
from app.core.alert_policy import AlertPolicy
from app.core.pipeline import (
    Pipeline,
    FrameCaptureStep,
    TriggerFilterStep,
    DetectStep,
    Ctx,
)


class FakeCamera:
    def __init__(self):
        self.t = 0.0
        self.i = 0
    def open(self): pass
    def grab(self):
        self.t += 0.1
        return True
    def read(self):
        self.t += 0.1; self.i += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        return Frame(image=img, t=self.t, index=self.i, w=640, h=480)
    def close(self): pass


class FakeDetector:
    def __init__(self, dets=None):
        self._dets = dets or []
    def detect(self, frame):
        return self._dets


class NullSink:
    def send(self, text, image_path=None): pass


class NullTel:
    def incr(self, *a, **k): pass
    def gauge(self, *a, **k): pass
    def time_ms(self, *a, **k): pass


class RecordingFrameStore:
    """Tracks save_frame calls for assertions."""
    def __init__(self):
        self.saved = []

    def save_frame(self, image, ts, detections=None, class_names_by_id=None, jpeg_quality=80):
        self.saved.append({
            "ts": ts,
            "detections": list(detections) if detections else [],
            "class_names": class_names_by_id,
        })
        return "/tmp/fake.jpg"


# ---------------------------------------------------------------------------
# Step ordering: TriggerFilterStep runs before FrameCaptureStep
# ---------------------------------------------------------------------------

def test_trigger_filter_before_frame_capture(tmp_path):
    """In a pipeline with frame_store, TriggerFilterStep must precede FrameCaptureStep."""
    cfg = Config()
    cfg.save_dir = str(tmp_path / "alerts")
    cfg.raw_frames_dir = str(tmp_path / "raw")
    cfg.alert_db_path = str(tmp_path / "alert_history.db")

    det = FakeDetector()
    det.labels = ["person", "car"]

    store = RecordingFrameStore()

    pipe = Pipeline(
        cfg=cfg,
        clock=SystemClock(),
        camera=FakeCamera(),
        detector=det,
        tracker=None,
        presence=PresencePolicy(min_frames=3, min_persist_sec=0.5),
        rate=RatePolicy(base_fps=2, high_fps=20, boost_arm_frames=3,
                        boost_min_sec=0.5, cooldown_sec=3, base_stride=1),
        alerts=AlertPolicy(window_sec=1.0),
        sink=NullSink(),
        telemetry=NullTel(),
        frame_store=store,
    )

    step_names = [type(s).__name__ for s in pipe.steps]
    tf_idx = step_names.index("TriggerFilterStep")
    fc_idx = step_names.index("FrameCaptureStep")
    assert tf_idx < fc_idx, (
        f"TriggerFilterStep (idx={tf_idx}) must run before "
        f"FrameCaptureStep (idx={fc_idx}), got order: {step_names}"
    )


# ---------------------------------------------------------------------------
# FrameCaptureStep uses ctx.trigger_dets (not re-filtering)
# ---------------------------------------------------------------------------

def test_frame_capture_uses_ctx_trigger_dets(tmp_path):
    """FrameCaptureStep should use pre-computed ctx.trigger_dets."""
    store = RecordingFrameStore()
    step = FrameCaptureStep(
        frame_store=store,
        class_names_by_id={0: "person"},
        active_fps=100,
        cooldown_sec=10,
    )

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(image=img, t=1.0, index=1, w=640, h=480)

    trigger_det = Detection((10, 10, 50, 50), 0.9, 0, track_id=1)
    non_trigger_det = Detection((100, 100, 200, 200), 0.8, 1, track_id=2)

    ctx = Ctx()
    ctx.frame = frame
    ctx.now = 1.0
    ctx.dets = (trigger_det, non_trigger_det)
    ctx.trigger_dets = (trigger_det,)

    step.run(ctx)

    assert len(store.saved) == 1
    assert len(store.saved[0]["detections"]) == 1
    assert store.saved[0]["detections"][0].cls_id == 0


def test_frame_capture_skips_when_no_triggers():
    """FrameCaptureStep should not save when trigger_dets is empty and outside cooldown."""
    store = RecordingFrameStore()
    step = FrameCaptureStep(
        frame_store=store,
        class_names_by_id={0: "person"},
        active_fps=100,
        cooldown_sec=5,
    )

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(image=img, t=100.0, index=1, w=640, h=480)

    ctx = Ctx()
    ctx.frame = frame
    ctx.now = 100.0
    ctx.dets = ()
    ctx.trigger_dets = ()

    step.run(ctx)
    assert len(store.saved) == 0


def test_frame_capture_respects_cooldown():
    """After last detection, FrameCaptureStep should keep saving during cooldown."""
    store = RecordingFrameStore()
    step = FrameCaptureStep(
        frame_store=store,
        class_names_by_id={0: "person"},
        active_fps=100,
        cooldown_sec=5,
    )

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # First frame: has trigger
    ctx = Ctx()
    ctx.frame = Frame(image=img, t=1.0, index=1, w=640, h=480)
    ctx.now = 1.0
    ctx.trigger_dets = (Detection((10, 10, 50, 50), 0.9, 0),)
    step.run(ctx)
    assert len(store.saved) == 1

    # Second frame: no trigger, but within cooldown (t=3.0, cooldown=5s)
    ctx2 = Ctx()
    ctx2.frame = Frame(image=img, t=3.0, index=2, w=640, h=480)
    ctx2.now = 3.0
    ctx2.trigger_dets = ()
    step.run(ctx2)
    assert len(store.saved) == 2

    # Third frame: no trigger, past cooldown (t=7.0, last detection was at 1.0)
    ctx3 = Ctx()
    ctx3.frame = Frame(image=img, t=7.0, index=3, w=640, h=480)
    ctx3.now = 7.0
    ctx3.trigger_dets = ()
    step.run(ctx3)
    assert len(store.saved) == 2  # not saved


def test_frame_capture_respects_active_fps():
    """FrameCaptureStep should not save faster than active_fps."""
    store = RecordingFrameStore()
    step = FrameCaptureStep(
        frame_store=store,
        class_names_by_id={0: "person"},
        active_fps=2.0,
        cooldown_sec=10,
    )

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    trigger = (Detection((10, 10, 50, 50), 0.9, 0),)

    # Frame at t=1.0 (non-zero to avoid _last_save_t=0.0 edge case)
    ctx1 = Ctx()
    ctx1.frame = Frame(image=img, t=1.0, index=1, w=640, h=480)
    ctx1.now = 1.0
    ctx1.trigger_dets = trigger
    step.run(ctx1)
    assert len(store.saved) == 1

    # Frame at t=1.1 (too soon for 2fps = 0.5s interval)
    ctx2 = Ctx()
    ctx2.frame = Frame(image=img, t=1.1, index=2, w=640, h=480)
    ctx2.now = 1.1
    ctx2.trigger_dets = trigger
    step.run(ctx2)
    assert len(store.saved) == 1  # skipped

    # Frame at t=1.6 (past interval)
    ctx3 = Ctx()
    ctx3.frame = Frame(image=img, t=1.6, index=3, w=640, h=480)
    ctx3.now = 1.6
    ctx3.trigger_dets = trigger
    step.run(ctx3)
    assert len(store.saved) == 2  # saved
