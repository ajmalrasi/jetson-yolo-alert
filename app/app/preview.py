# app/app/preview.py
import sys
import cv2

from app.core.config import Config
from app.core.clock import SystemClock
from app.core.presence_policy import PresencePolicy
from app.core.rate_policy import RatePolicy
from app.core.alert_policy import AlertPolicy
from app.core.pipeline import Pipeline

from app.adapters.camera_cv2 import Cv2Camera
from app.adapters.detector_ultra import UltralyticsDetector
from app.adapters.telemetry_log import LogTelemetry

import os
def resolve_path(p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join("/workspace/work", p)


preview_dir = "/workspace/work/preview"
os.makedirs(preview_dir, exist_ok=True)


def main():
    # Fail fast if there's no display (you said to exit the process)
    try:
        cv2.namedWindow("YOLO Preview Test")
        cv2.destroyWindow("YOLO Preview Test")
    except cv2.error:
        print("❌ No display available: cannot run preview mode.")
        sys.exit(1)

    cfg = Config()                 # <-- instantiate your dataclass (envs are read here)
    clock = SystemClock()
    tel = LogTelemetry()

    cam = Cv2Camera(cfg.src, clock=clock)

    det = UltralyticsDetector(
        engine_path=resolve_path(cfg.engine),
        conf=cfg.conf_thresh,
        imgsz=cfg.img_size,
        vid_stride=cfg.vid_stride,
        tracker_cfg=(cfg.tracker_cfg if cfg.tracker_on else None),
    )

    pres = PresencePolicy(cfg.min_frames, cfg.min_persist_sec)
    rate = RatePolicy(
        base_fps=cfg.base_fps,
        high_fps=cfg.high_fps,
        boost_arm_frames=cfg.boost_arm_frames,
        boost_min_sec=cfg.boost_min_sec,
        cooldown_sec=cfg.cooldown_sec,
        base_stride=cfg.vid_stride,
    )
    alerts = AlertPolicy(cfg.rate_window_sec)   # keep windowing; alerts are disabled below

    # 🚫 Alerts disabled for preview: sink=None, save_dir=None but draw=True to display boxes
    pipe = Pipeline(
        camera=cam,
        detector=det,
        tracker=None,
        sink=None,                               # disable Telegram (no alerts)
        clock=clock,
        tel=tel,
        pres=pres,
        rate=rate,
        alerts=alerts,
        draw_classes=cfg.draw_classes,
        conf_thresh=cfg.conf_thresh,
        save_dir=preview_dir,
        draw=True,
        trigger_classes=cfg.trigger_classes,
    )

    print("▶ Preview running. Press 'q' to quit.")
    for ctx in pipe.run():
        if ctx.frame is not None:
            cv2.namedWindow("YOLO Preview", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLO Preview", ctx.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
