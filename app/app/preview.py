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

def main():
    """Run full pipeline in preview mode (imshow, no alerts)."""
    cfg = Config()
    clock = SystemClock()
    tel = LogTelemetry()

    cam = Cv2Camera(cfg.src, clock=clock)
    det = UltralyticsDetector(
        engine_path=cfg.engine,
        conf=cfg.conf,
        imgsz=cfg.img_size,
        vid_stride=cfg.vid_stride,
        tracker_cfg=cfg.tracker
    )

    pres = PresencePolicy(cfg.min_frames, cfg.min_persist_sec)
    rate = RatePolicy(
        base_fps=cfg.base_fps,
        high_fps=cfg.high_fps,
        boost_arm_frames=cfg.boost_arm_frames,
        boost_min_sec=cfg.boost_min_sec,
        cooldown_sec=cfg.cooldown_sec,
        base_stride=cfg.vid_stride
    )
    alerts = AlertPolicy(cfg.rate_window_sec)

    pipe = Pipeline(
        camera=cam,
        detector=det,
        tracker=None,
        sink=None,   # 🚨 no alerts
        clock=clock,
        tel=tel,
        pres=pres,
        rate=rate,
        alerts=alerts,
        draw_classes=cfg.draw_classes,
        conf_thresh=cfg.conf,
        save_dir=None,
        draw=True,
        trigger_classes=cfg.trigger_classes,
    )

    for ctx in pipe.run():
        if ctx.frame is not None:
            cv2.imshow("YOLO Preview", ctx.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
