from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Protocol, Set, TYPE_CHECKING
import os
import time
import cv2

logger = logging.getLogger(__name__)

from .ports import Detection, Frame, Detector, ITracker, Camera, AlertSink, EventBus, Telemetry
from .state import PresenceState
from .rate_policy import RatePolicy, RateTarget, FullSpeedRatePolicy
from .presence_policy import PresencePolicy
from .alert_policy import AlertPolicy
from .clock import Clock
from .config import Config

if TYPE_CHECKING:
    from .alert_history import AlertHistoryStore
    from .frame_store import FrameStore

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

def _build_alert_message(
    count: int,
    best: float,
    trigger_class_names: Set[str],
    context_class_names: Set[str],
) -> str:
    noun = "object" if count == 1 else "objects"
    classes = ", ".join(sorted(trigger_class_names)) if trigger_class_names else "unknown"
    msg = (
        "Alert detected\n"
        f"- Triggered: {count} {noun}\n"
        f"- Best confidence: {best:.2f}\n"
        f"- Triggered classes: {classes}"
    )
    if context_class_names:
        context = ", ".join(sorted(context_class_names))
        msg += f"\n- Context classes in image: {context}"
    return msg

def _save_snapshot(
    path: str,
    frame: Frame,
    dets: Sequence[Detection],
    draw_ids: Set[int],
    conf: float,
    class_names_by_id: dict[int, str] | None = None,
    tracker_on: bool = False,
):
    from .annotate import draw_detections

    img = frame.image.copy()
    draw_detections(
        img,
        dets,
        class_names_by_id=class_names_by_id or {},
        draw_ids=draw_ids,
        conf_thresh=conf,
        tracker_on=tracker_on,
    )
    cv2.imwrite(path, img)

# ------------------------------
# Steps
# ------------------------------
@dataclass
class RateStep(PipelineStep):
    clock: Clock
    rate: RatePolicy
    telemetry: Telemetry

    def run(self, ctx: Ctx) -> Ctx:
        # decide target
        ctx.target = self.rate.decide(ctx.state, ctx.now)
        # soft sleep for FPS
        if ctx.target.fps > 0:
            min_dt = 1.0 / ctx.target.fps
            now = self.clock.now()
            dt = now - ctx.now
            if dt < min_dt:
                t0 = time.perf_counter()
                self.clock.sleep(min_dt - dt)
                sleep_ms = (time.perf_counter() - t0) * 1000.0
                self.telemetry.time_ms("rate_sleep_ms", sleep_ms)
        ctx.now = self.clock.now()
        return ctx

@dataclass
class ReadStep(PipelineStep):
    cam: Camera
    telemetry: Telemetry

    def run(self, ctx: Ctx) -> Ctx:
        ctx.frame_index += 1

        # Stride: on skip frames, grab() advances the buffer without decoding (~0.1ms).
        # Only retrieve() + decode on the frame we actually need.
        if ctx.target.vid_stride > 1 and (ctx.frame_index % ctx.target.vid_stride != 0):
            self.cam.grab()
            ctx.frame = None
            ctx.dets = ()
            return ctx

        t0 = time.perf_counter()
        frame = self.cam.read()
        read_ms = (time.perf_counter() - t0) * 1000.0
        self.telemetry.time_ms("read_ms", read_ms)
        if frame is None:
            return ctx
        ctx.frame = frame
        ctx.now = frame.t
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

        try:
            t0 = time.perf_counter()
            dets = self.det.detect(ctx.frame)
            self.telemetry.time_ms("detect_ms", (time.perf_counter() - t0) * 1000.0)
            if self.tracker:
                t1 = time.perf_counter()
                dets = self.tracker.update(ctx.frame, dets)
                self.telemetry.time_ms("track_ms", (time.perf_counter() - t1) * 1000.0)
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
    class_names_by_id: dict[int, str]
    telemetry: Telemetry
    tracker_on: bool = False
    history: Optional["AlertHistoryStore"] = None
    save_raw_frames: bool = False
    raw_frames_dir: str = ""

    def run(self, ctx: Ctx) -> Ctx:
        t_alert0 = time.perf_counter()
        # accumulate alerts whenever we see triggers (presence policy still tracks state separately)
        if ctx.trigger_dets:
            ids = [d.track_id for d in ctx.trigger_dets if d.track_id is not None] or [-1]
            best = max(d.conf for d in ctx.trigger_dets) if ctx.trigger_dets else 0.0
            frame_count = len(ctx.trigger_dets)
            frame_best = best
            frame_classes = {
                self.class_names_by_id.get(d.cls_id, f"class_{d.cls_id}") for d in ctx.trigger_dets
            }
            frame_context_classes = {
                self.class_names_by_id.get(d.cls_id, f"class_{d.cls_id}")
                for d in ctx.dets
                if d.conf >= self.conf_thresh and (d.cls_id in self.draw_ids if self.draw_ids else True)
            }

            # take a snapshot when we have trigger detections (if drawing is enabled)
            img_path = None
            if ctx.frame is not None and self.draw:
                os.makedirs(self.save_dir, exist_ok=True)
                img_path = os.path.join(self.save_dir, f"snapshot_{int(ctx.now*1000)}.jpg")
                t_snap = time.perf_counter()
                try:
                    _save_snapshot(
                        img_path, ctx.frame, ctx.dets, self.draw_ids, self.conf_thresh,
                        class_names_by_id=self.class_names_by_id,
                        tracker_on=self.tracker_on,
                    )
                except Exception:
                    img_path = None
                self.telemetry.time_ms(
                    "alert_snapshot_ms", (time.perf_counter() - t_snap) * 1000.0
                )

            if ctx.frame is not None and self.save_raw_frames:
                os.makedirs(self.raw_frames_dir, exist_ok=True)
                raw_path = os.path.join(self.raw_frames_dir, f"frame_{int(ctx.now*1000)}.jpg")
                t_raw = time.perf_counter()
                try:
                    cv2.imwrite(raw_path, ctx.frame.image)
                except Exception:
                    pass
                self.telemetry.time_ms(
                    "alert_raw_frame_ms", (time.perf_counter() - t_raw) * 1000.0
                )

            t_add = time.perf_counter()
            self.alert.add(
                ids,
                best_conf=best,
                now=ctx.now,
                rearm_sec=self.rearm_sec,
                frame_img_path=img_path,
                frame_count=frame_count,
                frame_best_conf=frame_best,
                frame_class_names=frame_classes,
                frame_context_class_names=frame_context_classes,
            )
            self.telemetry.time_ms("alert_add_ms", (time.perf_counter() - t_add) * 1000.0)

        # if due, flush window (even if scene is now empty)
        if self.alert.due(ctx.now):
            t_flush = time.perf_counter()
            count, best, img_path, frame_classes, context_classes = self.alert.flush(ctx.now)
            self.telemetry.time_ms("alert_flush_ms", (time.perf_counter() - t_flush) * 1000.0)
            ctx.alert_count = count
            ctx.alert_best_conf = best
            ctx.snapshot_path = img_path

            if self.history:
                t_hist = time.perf_counter()
                try:
                    self.history.insert_alert(
                        ts=ctx.now,
                        count=count,
                        best_conf=best,
                        image_path=img_path,
                        trigger_classes=frame_classes,
                        context_classes=context_classes,
                    )
                except Exception as e:
                    self.telemetry.incr("alert_history_errors")
                    self.telemetry.gauge("last_alert_history_exc", 1.0, msg=str(e))
                self.telemetry.time_ms(
                    "alert_history_ms", (time.perf_counter() - t_hist) * 1000.0
                )

            # send
            t_send = time.perf_counter()
            try:
                msg = _build_alert_message(count, best, frame_classes, context_classes)
                self.sink.send(msg, image_path=img_path)
                if self.event_bus:
                    from .events import AlertIssued
                    self.event_bus.publish("alerts", AlertIssued(count=count, best_conf=best, image_path=img_path))
            except Exception as e:
                self.telemetry.incr("alert_errors")
                self.telemetry.gauge("last_alert_exc", 1.0, msg=str(e))
            finally:
                self.telemetry.time_ms("alert_send_ms", (time.perf_counter() - t_send) * 1000.0)

        # publish presence transitions as events
        if self.event_bus and (ctx.became_present or ctx.became_idle):
            t_pres = time.perf_counter()
            if ctx.became_present:
                from .events import PersonDetected
                tr_ids = [d.track_id for d in ctx.trigger_dets if d.track_id is not None]
                best = max((d.conf for d in ctx.trigger_dets), default=0.0)
                self.event_bus.publish("presence", PersonDetected(track_ids=tr_ids, best_conf=best))
            if ctx.became_idle:
                from .events import PersonLost
                self.event_bus.publish("presence", PersonLost(last_seen_t=ctx.state.last_t))
            self.telemetry.time_ms("alert_presence_ms", (time.perf_counter() - t_pres) * 1000.0)

        self.telemetry.time_ms("alert_ms", (time.perf_counter() - t_alert0) * 1000.0)
        return ctx


@dataclass
class NoOpAlertStep(PipelineStep):
    """Skip alert accumulation, snapshots, Telegram, and DB (preview detector-only mode)."""

    def run(self, ctx: Ctx) -> Ctx:
        return ctx


@dataclass
class FrameCaptureStep(PipelineStep):
    """Save frames to disk at active_fps when detections are present.

    Does nothing when idle. Stays active for cooldown_sec after the last
    detection so we capture the tail of an event.
    Runs after TriggerFilterStep so ctx.trigger_dets is already populated.
    """
    frame_store: "FrameStore"
    class_names_by_id: dict[int, str]
    active_fps: float = 2.0
    cooldown_sec: float = 10.0
    _last_detection_t: float = field(default=0.0, init=False, repr=False)
    _last_save_t: float = field(default=0.0, init=False, repr=False)

    def run(self, ctx: Ctx) -> Ctx:
        if ctx.frame is None:
            return ctx

        if ctx.trigger_dets:
            self._last_detection_t = ctx.now

        in_active_window = (ctx.now - self._last_detection_t) < self.cooldown_sec
        if not in_active_window:
            return ctx

        min_interval = 1.0 / self.active_fps if self.active_fps > 0 else 0.5
        if (ctx.now - self._last_save_t) < min_interval:
            return ctx

        try:
            self.frame_store.save_frame(
                image=ctx.frame.image,
                ts=ctx.now,
                detections=ctx.trigger_dets or None,
                class_names_by_id=self.class_names_by_id,
            )
            self._last_save_t = ctx.now
        except Exception:
            logger.warning("FrameCaptureStep: failed to save frame", exc_info=True)

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
    alert_history: Optional["AlertHistoryStore"] = None
    frame_store: Optional["FrameStore"] = None
    preview_detector_only: bool = False

    def __post_init__(self):
        if self.preview_detector_only:
            self.alert_history = None
        elif self.alert_history is None:
            try:
                from .alert_history import AlertHistoryStore

                self.alert_history = AlertHistoryStore(self.cfg.alert_db_path)
            except Exception as e:
                self.alert_history = None
                self.telemetry.incr("alert_history_init_errors")
                self.telemetry.gauge("last_alert_history_init_exc", 1.0, msg=str(e))

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

        if not self.preview_detector_only:
            os.makedirs(self.cfg.save_dir, exist_ok=True)

        eff_rate: RatePolicy | FullSpeedRatePolicy = (
            FullSpeedRatePolicy() if self.preview_detector_only else self.rate
        )
        alert_step: PipelineStep = (
            NoOpAlertStep()
            if self.preview_detector_only
            else AlertStep(
                alert=self.alerts,
                sink=self.sink,
                event_bus=self.event_bus,
                rearm_sec=self.cfg.rearm_sec,
                save_dir=self.cfg.save_dir,
                draw_ids=self._draw_ids,
                conf_thresh=self.cfg.conf_thresh,
                draw=self.cfg.draw,
                class_names_by_id={v: k for k, v in self._name2id.items()},
                telemetry=self.telemetry,
                tracker_on=self.cfg.tracker_on,
                history=self.alert_history,
                save_raw_frames=self.cfg.save_raw_frames,
                raw_frames_dir=self.cfg.raw_frames_dir,
            )
        )

        frame_capture_step: Optional[PipelineStep] = None
        if self.frame_store is not None and not self.preview_detector_only:
            frame_capture_step = FrameCaptureStep(
                frame_store=self.frame_store,
                class_names_by_id={v: k for k, v in self._name2id.items()},
                active_fps=self.cfg.capture_active_fps,
                cooldown_sec=self.cfg.capture_cooldown_sec,
            )

        # wire steps
        steps: list[PipelineStep] = [
            RateStep(clock=self.clock, rate=eff_rate, telemetry=self.telemetry),
            ReadStep(cam=self.camera, telemetry=self.telemetry),
            DetectStep(
                det=self.detector,
                tracker=self.tracker,
                conf_thresh=self.cfg.conf_thresh,
                telemetry=self.telemetry,
            ),
            TriggerFilterStep(trigger_ids=self._trigger_ids, telemetry=self.telemetry),
        ]
        if frame_capture_step is not None:
            steps.append(frame_capture_step)
        steps.extend([
            PresenceStep(policy=self.presence),
            alert_step,
            TelemetryStep(telemetry=self.telemetry),
        ])
        self.steps = steps

    def iter_frames(self):
        """Yield context after each full pipeline pass (for preview / tooling)."""
        ctx = Ctx(now=self.clock.now())
        while True:
            loop_t0 = time.perf_counter()
            for step in self.steps:
                ctx = step.run(ctx)
            self.telemetry.time_ms(
                "pipeline_loop_ms", (time.perf_counter() - loop_t0) * 1000.0
            )
            yield ctx

    def run(self):
        for _ in self.iter_frames():
            pass
