import os
from ..core.clock import SystemClock
from ..core.config import Config
from ..core.presence_policy import PresencePolicy
from ..core.rate_policy import RatePolicy
from ..core.alert_policy import AlertPolicy
from ..core.pipeline import Pipeline
from ..adapters.camera_cv2 import Cv2Camera
from ..adapters.detector_ultra import UltralyticsDetector
from ..adapters.alerts_telegram import TelegramSink
from ..adapters.telemetry_log import LogTelemetry
# External tracker not required when using YOLO.track()

def resolve_path(p: str) -> str:
    if os.path.isabs(p): return p
    cand = os.path.join("/workspace/work", p)
    return cand if os.path.exists(cand) else p

def main():
    cfg = Config()
    clock = SystemClock()
    tel = LogTelemetry()

    cam = Cv2Camera(cfg.src, clock=clock)
    det = UltralyticsDetector(
        engine_path=resolve_path(cfg.engine),
        conf=cfg.conf_thresh,
        imgsz=cfg.img_size,
        vid_stride=cfg.vid_stride,
        tracker_cfg=resolve_path(cfg.tracker_cfg),  # <- use TRACKER from .env
    )

    sink = TelegramSink(cfg.tg_token, cfg.tg_chat)

    pres = PresencePolicy(min_frames=cfg.min_frames, min_persist_sec=cfg.min_persist_sec)
    rate = RatePolicy(
        base_fps=cfg.base_fps, high_fps=cfg.high_fps,
        boost_arm_frames=cfg.boost_arm_frames, boost_min_sec=cfg.boost_min_sec,
        cooldown_sec=cfg.cooldown_sec, base_stride=cfg.vid_stride
    )
    alerts = AlertPolicy(window_sec=cfg.rate_window_sec)

    pipe = Pipeline(camera=cam, detector=det, tracker=None, sink=sink,
                    clock=clock, tel=tel, pres=pres, rate=rate, alerts=alerts,
                    draw_classes=cfg.draw_classes, conf_thresh=cfg.conf_thresh,
                    save_dir=cfg.save_dir, draw=cfg.draw,
                    trigger_classes=cfg.trigger_classes)
    pipe.run()

if __name__ == "__main__":
    main()
