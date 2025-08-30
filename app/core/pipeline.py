from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Protocol, Iterable, Any, Set
import os, cv2

from .ports import Detection, Frame, Detector, ITracker, Camera, AlertSink, EventBus, Telemetry
from .state import PresenceState
from .rate_policy import RatePolicy, RateTarget
from .presence_policy import PresencePolicy
from .alert_policy import AlertPolicy
from .clock import Clock
from .config import Config

# ------------------------------
# Context passed through steps
# ------------------------------
@dataclass
class Ctx:
    # IO
    frame: Optional[Frame] = None
    dets: Sequence[Detection] = ()
    trigger_dets: Sequence[Detection] = ()

    # Time/Rate
    now: float = 0.0
    target: RateTarget = RateTarget(fps=0.0, vid_stride=1)
    frame_index: int = 0  # for stride

    # Presence
    state: PresenceState = field(default_factory=PresenceState)
    became_present: bool = False
    became_idle: bool = False

    # Alerts
    snapshot_path: Optional[str] = None
    alert_count: int = 0
    alert_best_conf: float = 0.0

# ------------------------------
# Step Protocol
# ------------------------------
class PipelineStep(Protocol):
    def run(self, ctx: Ctx) -> Ctx: ...

def _names_to_ids(names: Set[str], name2id: dict[str, int]) -> Set[int]:
    return {name2id[n] for n in names if n in name2id}

def _save_snapshot(path: str, frame: Frame, dets: Sequence[Detection], draw_ids: Set[int], conf: float):
    img = frame.image.copy()
    keep = [d for d in dets if d.conf >= conf and (d.cls_id in draw_ids if draw_ids else True)]
    for d in keep:
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(path, img)

# ------------------------------
# Steps
# ------------------------------
@dataclass
class RateStep(PipelineStep):
    clock: Clock
    rate: RatePolicy

    def run(self, ctx: Ctx) -> Ctx:
        # decide target
        ctx.target = self.rate.decide(ctx.state, ctx.now)
        # soft sleep for FPS
        if ctx.target.fps > 0:
            min_dt = 1.0 / ctx.target.fps
            now = self.clock.now()
            dt = now - ctx.now
            if dt < min_dt:
                self.clock.sleep(min_dt - dt)
        ctx.now = self.clock.now()
        return ctx

@dataclass
class ReadStep(PipelineStep):
    cam: Camera
    telemetry: Telemetry

    def run(self, ctx: Ctx) -> Ctx:
        frame = self.cam.read()
        if frame is None:
            # no frame: keep timing monotonic
            ctx.now = ctx.now
            return ctx
        ctx.frame = frame
        ctx.now = frame.t
        ctx.frame_index += 1
        self.telemetry.incr("frames")
        return ctx

@dataclass
class DetectStep(PipelineStep):
    det: Detector
    tracker: Optional[ITracker]
    conf_thresh: float
    telemetry: Telemetry

    def run(self, ctx: Ctx) -> Ctx:
        if ctx.frame is None:
            return ctx
        # stride: skip heavy work if not the chosen frame
        if ctx.target.vid_stride > 1 and (ctx.frame_index % ctx.target.vid_stride != 0):
            ctx.dets = ()
            return ctx

        try:
            dets = self.det.detect(ctx.frame.image)
            if self.tracker:
                dets = self.tracker.update(ctx.frame, dets)
            ctx.dets = dets
        except Exception as e:
            self.telemetry.incr("detect_errors")
            self.telemetry.gauge("last_detect_exc", 1.0, msg=str(e))
            ctx.dets = ()
        return ctx

@dataclass
class TriggerFilterStep(PipelineStep):
    trigger_ids: Set[int]
    telemetry: Telemetry

    def run(self, ctx: Ctx) -> Ctx:
        if not ctx.dets:
            ctx.trigger_dets = ()
            return ctx
        trig = [d for d in ctx.dets if (d.cls_id in self.trigger_ids)]
        ctx.trigger_dets = trig
        if trig:
            best = max(d.conf for d in trig)
            self.telemetry.gauge("trigger_best_conf", float(best))
        return ctx

@dataclass
class PresenceStep(PipelineStep):
    policy: PresencePolicy

    def run(self, ctx: Ctx) -> Ctx:
        ctx.state, ctx.became_present, ctx.became_idle = self.policy.update(
            ctx.state, ctx.now, ctx.trigger_dets
        )
        return ctx

@dataclass
class AlertStep(PipelineStep):
    alert: AlertPolicy
    sink: AlertSink
    event_bus: Optional[EventBus]
    rearm_sec: float
    save_dir: str
    draw_ids: Set[int]
    conf_thresh: float
    draw: bool
    telemetry: Telemetry

    def run(self, ctx: Ctx) -> Ctx:
        # accumulate alerts whenever we see triggers
        if ctx.trigger_dets:
            ids = [d.track_id for d in ctx.trigger_dets if d.track_id is not None]
            best = max(d.conf for d in ctx.trigger_dets) if ctx.trigger_dets else 0.0
            self.alert.add(ids, best_conf=best, now=ctx.now, rearm_sec=self.rearm_sec)

        # if due, flush window
        if self.alert.due(ctx.now):
            count, best = self.alert.flush(ctx.now)
            ctx.alert_count = count
            ctx.alert_best_conf = best

            # snapshot (optional)
            img_path = None
            if ctx.frame is not None and self.draw:
                os.makedirs(self.save_dir, exist_ok=True)
                img_path = os.path.join(self.save_dir, f"snapshot_{int(ctx.now*1000)}.jpg")
                try:
                    _save_snapshot(img_path, ctx.frame, ctx.dets, self.draw_ids, self.conf_thresh)
                except Exception:
                    img_path = None
            ctx.snapshot_path = img_path

            # send
            try:
                msg = f"Alert: {count} object(s) (best={best:.2f})"
                self.sink.send(msg, image_path=img_path)
                if self.event_bus:
                    from .events import AlertIssued
                    self.event_bus.publish("alerts", AlertIssued(count=count, best_conf=best, image_path=img_path))
            except Exception as e:
                self.telemetry.incr("alert_errors")
                self.telemetry.gauge("last_alert_exc", 1.0, msg=str(e))

        # publish presence transitions as events
        if self.event_bus and ctx.became_present:
            from .events import PersonDetected
            tr_ids = [d.track_id for d in ctx.trigger_dets if d.track_id is not None]
            best = max((d.conf for d in ctx.trigger_dets), default=0.0)
            self.event_bus.publish("presence", PersonDetected(track_ids=tr_ids, best_conf=best))
        if self.event_bus and ctx.became_idle:
            from .events import PersonLost
            self.event_bus.publish("presence", PersonLost(last_seen_t=ctx.state.last_t))

        return ctx

@dataclass
class TelemetryStep(PipelineStep):
    telemetry: Telemetry

    def run(self, ctx: Ctx) -> Ctx:
        # state gauges
        self.telemetry.gauge("present", 1.0 if ctx.state.present else 0.0)
        self.telemetry.gauge("fps_target", float(ctx.target.fps or 0.0))
        self.telemetry.gauge("vid_stride", float(ctx.target.vid_stride))
        return ctx

# ------------------------------
# Pipeline
# ------------------------------
@dataclass
class Pipeline:
    cfg: Config
    clock: Clock
    camera: Camera
    detector: Detector
    tracker: Optional[ITracker]
    presence: PresencePolicy
    rate: RatePolicy
    alerts: AlertPolicy
    sink: AlertSink
    telemetry: Telemetry
    event_bus: Optional[EventBus] = None

    def __post_init__(self):

        # Build name→id map from the active detector's labels
        labels = getattr(self.detector, "labels", None)
        if isinstance(labels, dict):
            # Ultralytics often exposes {id: "name"}
            name2id = {str(v).lower(): int(k) for k, v in labels.items()}
        else:
            # Or a list indexed by id
            labels = labels or []
            name2id = {str(n).lower(): i for i, n in enumerate(labels)}

        self._name2id = name2id  # keep if you want later use/telemetry
        self._trigger_ids = _names_to_ids(self.cfg.trigger_classes, name2id)
        self._draw_ids    = _names_to_ids(self.cfg.draw_classes, name2id)

        os.makedirs(self.cfg.save_dir, exist_ok=True)

        # wire steps
        self.steps: list[PipelineStep] = [
            RateStep(clock=self.clock, rate=self.rate),
            ReadStep(cam=self.camera, telemetry=self.telemetry),
            DetectStep(det=self.detector, tracker=self.tracker, conf_thresh=self.cfg.conf_thresh, telemetry=self.telemetry),
            TriggerFilterStep(trigger_ids=self._trigger_ids, telemetry=self.telemetry),
            PresenceStep(policy=self.presence),
            AlertStep(
                alert=self.alerts, sink=self.sink, event_bus=self.event_bus,
                rearm_sec=self.cfg.rearm_sec, save_dir=self.cfg.save_dir,
                draw_ids=self._draw_ids, conf_thresh=self.cfg.conf_thresh,
                draw=self.cfg.draw, telemetry=self.telemetry
            ),
            TelemetryStep(telemetry=self.telemetry),
        ]

    def run(self):
        ctx = Ctx(now=self.clock.now())
        while True:
            for step in self.steps:
                ctx = step.run(ctx)
            # loop continues forever; stopping conditions (signals) handled by outer app runner
