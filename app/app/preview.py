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
from app.core.annotate import draw_detections, color_bgr_for_det
from typing import Any, Optional, Sequence, Set, Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_OK = True
except ImportError:
    _PIL_OK = False

from app.adapters.camera_cv2 import Cv2Camera, ThreadedCamera
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


_color_bgr_for_det = color_bgr_for_det


PANEL_W = 370
FONT = cv2.FONT_HERSHEY_DUPLEX
TEXT = (220, 216, 208)
TEXT_DIM = (140, 136, 130)
ACCENT = (160, 190, 230)

# Prefer a readable system sans for the stats column (PIL); bbox labels stay OpenCV.
_PANEL_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
)
_panel_font_triple: Optional[Tuple[Any, Any, Any]] = None
_panel_fonts_failed = False


def _bgr_to_rgb(bgr: tuple) -> Tuple[int, int, int]:
    b, g, r = bgr
    return (int(r), int(g), int(b))


def _load_panel_fonts() -> Optional[Tuple[Any, Any, Any]]:
    """DejaVu/Liberation/Noto TTF for sidebar; None if unavailable."""
    global _panel_font_triple, _panel_fonts_failed
    if _panel_fonts_failed:
        return None
    if _panel_font_triple is not None:
        return _panel_font_triple
    if not _PIL_OK:
        _panel_fonts_failed = True
        return None
    path = next((p for p in _PANEL_FONT_PATHS if os.path.isfile(p)), None)
    if not path:
        _panel_fonts_failed = True
        return None
    bold = path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
    if not os.path.isfile(bold):
        bold = path
    try:
        _panel_font_triple = (
            ImageFont.truetype(bold, 24),
            ImageFont.truetype(path, 20),
            ImageFont.truetype(path, 18),
        )
        return _panel_font_triple
    except OSError:
        _panel_fonts_failed = True
        return None


def _stats_panel_cv2(
    h: int,
    fps: float,
    keep: List[Detection],
    class_names_by_id: Dict[int, str],
) -> np.ndarray:
    panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (42, 40, 38)
    cv2.line(panel, (0, 0), (0, h - 1), (72, 70, 68), 1)

    y = 28
    lh = 26

    def line(txt: str, color=TEXT, scale: float = 0.65, thick: int = 1) -> None:
        nonlocal y
        cv2.putText(panel, txt, (14, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += lh

    line("Preview stats", ACCENT, 0.70, 1)
    y += 4
    line(f"FPS   {fps:.1f}", TEXT)
    line(f"Objects {len(keep)}", TEXT)
    y += 8
    line("Track · class · conf", TEXT_DIM, 0.55, 1)
    y += 4

    rows = sorted(
        keep,
        key=lambda d: (d.track_id is None, d.track_id or -1, d.cls_id),
    )
    max_rows = max(1, (h - y - 28) // lh)
    for d in rows[:max_rows]:
        tid = f"{int(d.track_id)}" if d.track_id is not None else "—"
        nm = class_names_by_id.get(d.cls_id, "?")[:14]
        c = _color_bgr_for_det(d)
        txt = f"{tid:>4}  {nm:14}  {d.conf:.2f}"
        cv2.putText(panel, txt, (14, y), FONT, 0.62, c, 1, cv2.LINE_AA)
        y += lh
    if len(rows) > max_rows:
        line(f"+{len(rows) - max_rows} more", TEXT_DIM, 0.52, 1)
    return panel


def _stats_panel_pil(
    h: int,
    fps: float,
    keep: List[Detection],
    class_names_by_id: Dict[int, str],
) -> Optional[np.ndarray]:
    fonts = _load_panel_fonts()
    if fonts is None:
        return None
    title_f, body_f, small_f = fonts
    bg_rgb = _bgr_to_rgb((42, 40, 38))
    im = Image.new("RGB", (PANEL_W, h), bg_rgb)
    draw = ImageDraw.Draw(im)
    x = 14
    y = 22
    draw.text((x, y), "Preview stats", font=title_f, fill=_bgr_to_rgb(ACCENT))
    y += 34
    draw.text((x, y), f"FPS   {fps:.1f}", font=body_f, fill=_bgr_to_rgb(TEXT))
    y += 26
    draw.text((x, y), f"Objects {len(keep)}", font=body_f, fill=_bgr_to_rgb(TEXT))
    y += 30
    draw.text((x, y), "Track · class · conf", font=small_f, fill=_bgr_to_rgb(TEXT_DIM))
    y += 28
    lh = 28
    rows = sorted(
        keep,
        key=lambda d: (d.track_id is None, d.track_id or -1, d.cls_id),
    )
    max_rows = max(1, (h - y - 26) // lh)
    for d in rows[:max_rows]:
        tid = f"{int(d.track_id)}" if d.track_id is not None else "—"
        nm = class_names_by_id.get(d.cls_id, "?")[:14]
        txt = f"{tid:>4}  {nm:14}  {d.conf:.2f}"
        draw.text((x, y), txt, font=small_f, fill=_bgr_to_rgb(_color_bgr_for_det(d)))
        y += lh
    if len(rows) > max_rows:
        draw.text(
            (x, y),
            f"+{len(rows) - max_rows} more",
            font=small_f,
            fill=_bgr_to_rgb(TEXT_DIM),
        )
    arr = np.asarray(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _build_stats_panel(
    h: int,
    fps: float,
    keep: List[Detection],
    class_names_by_id: Dict[int, str],
) -> np.ndarray:
    if _PIL_OK:
        try:
            pil_panel = _stats_panel_pil(h, fps, keep, class_names_by_id)
            if pil_panel is not None:
                cv2.line(pil_panel, (0, 0), (0, h - 1), (72, 70, 68), 1)
                return pil_panel
        except Exception:
            pass
    return _stats_panel_cv2(h, fps, keep, class_names_by_id)


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

    draw_detections(
        img,
        keep,
        class_names_by_id=class_names_by_id,
        conf_thresh=0.0,
        tracker_on=tracker_on,
    )

    h = img.shape[0]
    panel = _build_stats_panel(h, fps, keep, class_names_by_id)

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

    raw_cam = Cv2Camera(cfg.src, clock=clock)
    cam = ThreadedCamera(raw_cam)

    det = UltralyticsDetector(
        engine_path=resolve_path(cfg.engine),
        conf=cfg.conf_thresh,
        imgsz=cfg.img_size,
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
    if use_display:
        cv2.namedWindow("YOLO Preview", cv2.WINDOW_NORMAL)
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
