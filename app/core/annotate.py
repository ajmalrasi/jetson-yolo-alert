"""Shared frame annotation utilities for bounding boxes and labels.

Used by both the live preview and the alert snapshot pipeline so
Telegram images get the same quality annotations as the browser UI.
"""
from __future__ import annotations

import colorsys
from typing import Dict, Sequence, Set

import cv2
import numpy as np

from .ports import Detection, Frame

FONT = cv2.FONT_HERSHEY_DUPLEX
_LABEL_BG = (28, 26, 24)


def color_bgr_for_det(det: Detection) -> tuple:
    """Muted, distinct BGR colour keyed by track-id (preferred) or class-id."""
    seed = int(det.track_id) * 997 if det.track_id is not None else det.cls_id * 41 + 17
    h = (seed % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.5, 0.88)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_detections(
    image: np.ndarray,
    dets: Sequence[Detection],
    *,
    class_names_by_id: Dict[int, str],
    draw_ids: Set[int] | None = None,
    conf_thresh: float = 0.0,
    tracker_on: bool = False,
    font_scale: float = 0.60,
    box_thickness: int = 2,
) -> np.ndarray:
    """Draw coloured bounding boxes with class/track labels onto *image* (mutates in-place).

    Returns the same array for convenience.
    """
    for d in dets:
        if d.conf < conf_thresh:
            continue
        if draw_ids and d.cls_id not in draw_ids:
            continue

        col = color_bgr_for_det(d)
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), col, box_thickness, lineType=cv2.LINE_AA)

        name = class_names_by_id.get(d.cls_id, f"c{d.cls_id}")
        label = name
        if tracker_on and d.track_id is not None:
            label = f"{name} - id {int(d.track_id)}"

        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, 1)
        ty = max(y1 - 6, th + 6)
        cv2.rectangle(
            image,
            (x1, ty - th - 8),
            (x1 + tw + 8, ty + 4),
            _LABEL_BG,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, ty - 2),
            FONT,
            font_scale,
            col,
            1,
            cv2.LINE_AA,
        )
    return image
