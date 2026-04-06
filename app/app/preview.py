# app/app/preview.py
import os
import sys
import cv2

from app.core.config import Config
from app.core.clock import SystemClock
from app.core.presence_policy import PresencePolicy
from app.core.rate_policy import RatePolicy
from app.core.alert_policy import AlertPolicy
from app.core.pipeline import Pipeline, _names_to_ids
from app.core.ports import Frame, Detection
from typing import Optional, Sequence, Set

from app.adapters.camera_cv2 import Cv2Camera
from app.adapters.detector_ultra import UltralyticsDetector
from app.adapters.telemetry_log import LogTelemetry
from app.adapters.mjpeg_stream import MjpegStreamServer


preview_dir = "/workspace/work/preview"
os.makedirs(preview_dir, exist_ok=True)


def resolve_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    cand = os.path.join("/workspace/work", p)
    return cand if os.path.exists(cand) else p


class NullSink:
    def send(self, text: str, image_path=None) -> None:
        pass


def _draw_class_ids(det, cfg: Config) -> Set[int]:
    labels = getattr(det, "labels", None)
    if isinstance(labels, dict):
        name2id = {str(v).lower(): int(k) for k, v in labels.items()}
    else:
        labels = labels or []
        name2id = {str(n).lower(): i for i, n in enumerate(labels)}
    return _names_to_ids(cfg.draw_classes, name2id)


def _annotate_bgr(
    frame: Frame,
    dets: Sequence[Detection],
    draw_ids: Set[int],
    conf: float,
    label_track_ids: bool = False,
):
    img = frame.image.copy()
    keep = [
        d
        for d in dets
        if d.conf >= conf and (d.cls_id in draw_ids if draw_ids else True)
    ]
    for d in keep:
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label_track_ids and d.track_id is not None:
            cv2.putText(
                img,
                str(int(d.track_id)),
                (x1, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return img


def _use_local_window() -> bool:
    """
    Whether to call cv2.imshow. Never probe with cv2.namedWindow: with the Qt
    backend, no display can abort the process instead of raising cv2.error.
    """
    override = os.getenv("PREVIEW_USE_DISPLAY", "").strip().lower()
    if override in ("0", "false", "no"):
        return False
    if override in ("1", "true", "yes"):
        return bool(os.environ.get("DISPLAY", "").strip())
    # auto: local window only when X11/Wayland display is advertised
    return bool(os.environ.get("DISPLAY", "").strip())


def main():
    stream_port = int(os.getenv("PREVIEW_STREAM_PORT", "0"))
    use_display = _use_local_window()
    if not use_display and stream_port <= 0:
        print(
            "❌ No display: set PREVIEW_STREAM_PORT (e.g. 8080) for MJPEG, "
            "or use X11 / a monitor."
        )
        sys.exit(1)

    cfg = Config()
    detector_only = os.getenv("PREVIEW_DETECTOR_ONLY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    clock = SystemClock()
    tel = LogTelemetry()

    cam = Cv2Camera(cfg.src, clock=clock)

    # Bench mode: process every frame in the pipeline + Ultralytics track() stride 1
    vid_stride_eff = 1 if detector_only else cfg.vid_stride
    det = UltralyticsDetector(
        engine_path=resolve_path(cfg.engine),
        conf=cfg.conf_thresh,
        imgsz=cfg.img_size,
        vid_stride=vid_stride_eff,
        tracker_cfg=resolve_path(cfg.tracker_cfg) if cfg.tracker_on else None,
    )

    pres = PresencePolicy(
        min_frames=cfg.min_frames, min_persist_sec=cfg.min_persist_sec
    )
    rate = RatePolicy(
        base_fps=cfg.base_fps,
        high_fps=cfg.high_fps,
        boost_arm_frames=cfg.boost_arm_frames,
        boost_min_sec=cfg.boost_min_sec,
        cooldown_sec=cfg.cooldown_sec,
        base_stride=cfg.vid_stride,
    )
    alerts = AlertPolicy(
        window_sec=cfg.rate_window_sec, cooldown_sec=cfg.alert_cooldown_sec
    )

    pipe = Pipeline(
        cfg=cfg,
        clock=clock,
        camera=cam,
        detector=det,
        tracker=None,
        presence=pres,
        rate=rate,
        alerts=alerts,
        sink=NullSink(),
        telemetry=tel,
        preview_detector_only=detector_only,
    )

    draw_ids = _draw_class_ids(det, cfg)
    stream: Optional[MjpegStreamServer] = None
    if stream_port > 0:
        bind = os.getenv("PREVIEW_STREAM_BIND", "0.0.0.0")
        max_w = int(os.getenv("PREVIEW_STREAM_MAX_WIDTH", "1280"))
        quality = int(os.getenv("PREVIEW_STREAM_QUALITY", "82"))
        max_fps = float(os.getenv("PREVIEW_STREAM_FPS", "25"))
        stream = MjpegStreamServer(
            host=bind,
            port=stream_port,
            max_width=max_w,
            quality=quality,
            max_fps=max_fps,
        )
        stream.start()
        print(
            f"▶ Browser UI (buttons): http://<this-host>:{stream_port}/  "
            f"← open this path, not /stream"
        )
        print(
            f"   /stream = video-only (no HTML). Rebuild image after code changes. "
            f"Bind {bind}:{stream_port}"
        )

    if use_display:
        print("▶ Local window: press 'q' to quit.")
    if detector_only:
        print(
            "▶ PREVIEW_DETECTOR_ONLY: full-speed loop, no alerts/DB/snapshots; "
            "track IDs on boxes (cyan). docker compose alert is unchanged."
        )
    cam.open()
    try:
        for ctx in pipe.iter_frames():
            if ctx.frame is None:
                continue
            vis = _annotate_bgr(
                ctx.frame,
                ctx.dets,
                draw_ids,
                cfg.conf_thresh,
                label_track_ids=detector_only and cfg.tracker_on,
            )
            if stream is not None:
                stream.submit_frame(vis)
            if use_display:
                cv2.namedWindow("YOLO Preview", cv2.WINDOW_NORMAL)
                cv2.imshow("YOLO Preview", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if stream is not None:
            stream.stop()
        cam.close()
        if use_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
