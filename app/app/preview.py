# app/app/preview.py
import colorsys
import os
import sys
import time

import cv2
import numpy as np

from app.core.config import Config
from app.core.clock import SystemClock
from app.core.presence_policy import PresencePolicy
from app.core.rate_policy import RatePolicy
from app.core.alert_policy import AlertPolicy
from app.core.pipeline import Pipeline, _names_to_ids
from app.core.ports import Frame, Detection
from typing import Optional, Sequence, Set, Dict, List

from app.adapters.camera_cv2 import Cv2Camera
from app.adapters.detector_ultra import UltralyticsDetector
from app.adapters.telemetry_setup import get_telemetry
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


def _class_names_by_id(det) -> Dict[int, str]:
    labels = getattr(det, "labels", None)
    if isinstance(labels, dict):
        return {int(k): str(v) for k, v in labels.items()}
    if isinstance(labels, (list, tuple)):
        return {i: str(n) for i, n in enumerate(labels)}
    return {}


def _color_bgr_for_det(d: Detection) -> tuple:
    """Muted, distinct BGR per track (preferred) or class."""
    hue_seed = int(d.track_id) * 997 if d.track_id is not None else d.cls_id * 41 + 17
    h = (hue_seed % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.5, 0.88)
    return (int(b * 255), int(g * 255), int(r * 255))


PANEL_W = 268
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT = (220, 216, 208)
TEXT_DIM = (140, 136, 130)
ACCENT = (160, 190, 230)


def _annotate_preview_frame(
    frame: Frame,
    dets: Sequence[Detection],
    draw_ids: Set[int],
    conf: float,
    class_names_by_id: Dict[int, str],
    fps: float,
    *,
    tracker_on: bool,
) -> np.ndarray:
    img = frame.image.copy()
    keep: List[Detection] = [
        d
        for d in dets
        if d.conf >= conf and (d.cls_id in draw_ids if draw_ids else True)
    ]

    for d in keep:
        col = _color_bgr_for_det(d)
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, lineType=cv2.LINE_AA)
        name = class_names_by_id.get(d.cls_id, f"c{d.cls_id}")
        label = f"{name}"
        if tracker_on and d.track_id is not None:
            label = f"{name} · id {int(d.track_id)}"
        (tw, th), bl = cv2.getTextSize(label, FONT, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(
            img,
            (x1, ty - th - 6),
            (x1 + tw + 6, ty + 2),
            (28, 26, 24),
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (x1 + 3, ty - 2),
            FONT,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

    # FPS chip (top-left on video)
    fps_t = f"{fps:.1f} FPS"
    (tw, th), _ = cv2.getTextSize(fps_t, FONT, 0.75, 2)
    pad = 10
    cv2.rectangle(
        img,
        (pad, pad),
        (pad + tw + 14, pad + th + 14),
        (36, 34, 40),
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        fps_t,
        (pad + 7, pad + th + 4),
        FONT,
        0.75,
        ACCENT,
        2,
        cv2.LINE_AA,
    )

    # Right sidebar
    h, w = img.shape[:2]
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (42, 40, 38)
    cv2.line(panel, (0, 0), (0, h - 1), (72, 70, 68), 1)

    y = 22
    lh = 20

    def line(txt, color=TEXT, scale=0.5, thick=1):
        nonlocal y
        cv2.putText(panel, txt, (12, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += lh

    line("Preview stats", ACCENT, 0.55, 1)
    y += 4
    line(f"FPS   {fps:.1f}", TEXT)
    line(f"Objects {len(keep)}", TEXT)
    y += 8
    line("Track · class · conf", TEXT_DIM, 0.42, 1)
    y += 4

    rows = sorted(
        keep,
        key=lambda d: (d.track_id is None, d.track_id or -1, d.cls_id),
    )
    max_rows = max(1, (h - y - 24) // lh)
    for i, d in enumerate(rows[:max_rows]):
        tid = f"{int(d.track_id)}" if d.track_id is not None else "—"
        nm = class_names_by_id.get(d.cls_id, "?")[:14]
        c = _color_bgr_for_det(d)
        txt = f"{tid:>4}  {nm:14}  {d.conf:.2f}"
        cv2.putText(panel, txt, (12, y), FONT, 0.42, c, 1, cv2.LINE_AA)
        y += lh
    if len(rows) > max_rows:
        line(f"+{len(rows) - max_rows} more", TEXT_DIM, 0.42, 1)

    out = np.hstack([img, panel])
    return out


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
    tel = get_telemetry()

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
    class_names_by_id = _class_names_by_id(det)
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
            "▶ PREVIEW_DETECTOR_ONLY: full-speed loop, no alerts/DB/snapshots. "
            "docker compose alert is unchanged."
        )
    cam.open()
    fps_ema = 0.0
    last_t: Optional[float] = None
    fps_alpha = 0.12
    try:
        for ctx in pipe.iter_frames():
            if ctx.frame is None:
                continue
            now = time.perf_counter()
            if last_t is not None:
                dt = now - last_t
                if 1e-6 < dt < 5.0:
                    inst = 1.0 / dt
                    fps_ema = (
                        inst
                        if fps_ema <= 0
                        else (fps_alpha * inst + (1.0 - fps_alpha) * fps_ema)
                    )
            last_t = now
            vis = _annotate_preview_frame(
                ctx.frame,
                ctx.dets,
                draw_ids,
                cfg.conf_thresh,
                class_names_by_id,
                fps_ema,
                tracker_on=cfg.tracker_on,
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
