import logging
import os
import threading
import time

from ..core.clock import SystemClock
from ..core.config import Config
from ..core.presence_policy import PresencePolicy
from ..core.rate_policy import RatePolicy
from ..core.alert_policy import AlertPolicy
from ..core.pipeline import Pipeline
from ..adapters.camera_cv2 import Cv2Camera
from ..adapters.detector_ultra import UltralyticsDetector
from ..adapters.alerts_telegram import TelegramSink
from ..adapters.telemetry_setup import get_telemetry

logger = logging.getLogger(__name__)


def resolve_path(p: str) -> str:
    if os.path.isabs(p): return p
    cand = os.path.join("/workspace/work", p)
    return cand if os.path.exists(cand) else p


def _build_frame_store(cfg: Config):
    """Build FrameStore if frames_dir is configured."""
    if not cfg.frames_dir or cfg.frames_dir == "none":
        return None
    try:
        from ..core.frame_store import FrameStore
        return FrameStore(cfg.frames_dir)
    except Exception:
        logger.exception("Failed to initialize FrameStore")
        return None


def _start_cleanup_thread(frame_store, retention_days: int) -> None:
    """Periodically clean up old frames in a background thread."""
    def _cleanup_loop():
        while True:
            time.sleep(3600)
            try:
                frame_store.cleanup(retention_days)
            except Exception:
                logger.exception("Frame cleanup failed")

    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()


def main():
    cfg = Config()
    clock = SystemClock()
    tel = get_telemetry()

    cam = Cv2Camera(cfg.src, clock=clock)
    det = UltralyticsDetector(
        engine_path=resolve_path(cfg.engine),
        conf=cfg.conf_thresh,
        imgsz=cfg.img_size,
        vid_stride=cfg.vid_stride,
        tracker_cfg=resolve_path(cfg.tracker_cfg),
    )

    sink = TelegramSink(cfg.tg_token, cfg.tg_chat)

    pres = PresencePolicy(min_frames=cfg.min_frames, min_persist_sec=cfg.min_persist_sec)
    rate = RatePolicy(
        base_fps=cfg.base_fps, high_fps=cfg.high_fps,
        boost_arm_frames=cfg.boost_arm_frames, boost_min_sec=cfg.boost_min_sec,
        cooldown_sec=cfg.cooldown_sec, base_stride=cfg.vid_stride
    )
    alerts = AlertPolicy(window_sec=cfg.rate_window_sec, cooldown_sec=cfg.alert_cooldown_sec)

    frame_store = _build_frame_store(cfg)
    if frame_store:
        logger.info(
            "Frame capture enabled: dir=%s, active_fps=%.1f, cooldown=%.0fs, retention=%dd",
            cfg.frames_dir, cfg.capture_active_fps, cfg.capture_cooldown_sec, cfg.frames_retention_days,
        )
        _start_cleanup_thread(frame_store, cfg.frames_retention_days)

    pipe = Pipeline(
        cfg=cfg,
        clock=clock,
        camera=cam,
        detector=det,
        tracker=None,
        presence=pres,
        rate=rate,
        alerts=alerts,
        sink=sink,
        telemetry=tel,
        frame_store=frame_store,
    )
    cam.open()
    try:
        pipe.run()
    finally:
        cam.close()

if __name__ == "__main__":
    main()
