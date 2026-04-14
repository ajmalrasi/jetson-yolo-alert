"""Tests for the shared annotation module app/core/annotate.py."""
import numpy as np

from app.core.annotate import color_bgr_for_det, draw_detections
from app.core.ports import Detection


def _img(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# color_bgr_for_det
# ---------------------------------------------------------------------------

def test_color_deterministic_by_track_id():
    d = Detection((0, 0, 10, 10), 0.9, cls_id=0, track_id=5)
    assert color_bgr_for_det(d) == color_bgr_for_det(d)


def test_color_differs_by_track_id():
    d1 = Detection((0, 0, 10, 10), 0.9, cls_id=0, track_id=1)
    d2 = Detection((0, 0, 10, 10), 0.9, cls_id=0, track_id=2)
    assert color_bgr_for_det(d1) != color_bgr_for_det(d2)


def test_color_falls_back_to_cls_id_when_no_track():
    d1 = Detection((0, 0, 10, 10), 0.9, cls_id=0)
    d2 = Detection((0, 0, 10, 10), 0.9, cls_id=1)
    assert color_bgr_for_det(d1) != color_bgr_for_det(d2)


def test_color_returns_bgr_tuple():
    d = Detection((0, 0, 10, 10), 0.9, cls_id=0)
    c = color_bgr_for_det(d)
    assert isinstance(c, tuple) and len(c) == 3
    assert all(0 <= ch <= 255 for ch in c)


# ---------------------------------------------------------------------------
# draw_detections — filtering
# ---------------------------------------------------------------------------

def test_draw_filters_below_conf_thresh():
    img = _img()
    low = Detection((10, 10, 50, 50), 0.3, cls_id=0)
    high = Detection((60, 60, 100, 100), 0.8, cls_id=0)
    original = img.copy()

    draw_detections(img, [low], class_names_by_id={0: "person"}, conf_thresh=0.5)
    assert np.array_equal(img, original), "Low-conf detection should not be drawn"

    draw_detections(img, [high], class_names_by_id={0: "person"}, conf_thresh=0.5)
    assert not np.array_equal(img, original), "High-conf detection should be drawn"


def test_draw_filters_by_draw_ids():
    img = _img()
    d_person = Detection((10, 10, 50, 50), 0.9, cls_id=0)
    d_car = Detection((60, 60, 100, 100), 0.9, cls_id=1)
    original = img.copy()

    draw_detections(
        img, [d_car], class_names_by_id={0: "person", 1: "car"}, draw_ids={0},
    )
    assert np.array_equal(img, original), "car (cls_id=1) should be filtered out"

    draw_detections(
        img, [d_person], class_names_by_id={0: "person", 1: "car"}, draw_ids={0},
    )
    assert not np.array_equal(img, original), "person (cls_id=0) should be drawn"


def test_draw_no_draw_ids_draws_all():
    img = _img()
    d = Detection((10, 10, 50, 50), 0.9, cls_id=5)
    original = img.copy()

    draw_detections(img, [d], class_names_by_id={5: "cat"})
    assert not np.array_equal(img, original)


# ---------------------------------------------------------------------------
# draw_detections — labels
# ---------------------------------------------------------------------------

def test_draw_returns_same_array():
    img = _img()
    d = Detection((10, 10, 50, 50), 0.9, cls_id=0)
    result = draw_detections(img, [d], class_names_by_id={0: "person"})
    assert result is img


def test_draw_with_tracker_on():
    """When tracker_on=True and track_id is set, label should include id."""
    img = _img()
    d = Detection((10, 10, 100, 100), 0.9, cls_id=0, track_id=7)
    draw_detections(
        img, [d], class_names_by_id={0: "person"}, tracker_on=True,
    )
    assert not np.array_equal(img, _img())


def test_draw_without_tracker_on():
    """When tracker_on=False, detection should still be drawn (just no id label)."""
    img = _img()
    d = Detection((10, 10, 100, 100), 0.9, cls_id=0, track_id=7)
    draw_detections(
        img, [d], class_names_by_id={0: "person"}, tracker_on=False,
    )
    assert not np.array_equal(img, _img())


def test_draw_unknown_class_id_uses_fallback_name():
    """Class IDs not in the map should get a 'cN' fallback label."""
    img = _img()
    d = Detection((10, 10, 100, 100), 0.9, cls_id=99)
    draw_detections(img, [d], class_names_by_id={})
    assert not np.array_equal(img, _img())


def test_draw_empty_dets_is_noop():
    img = _img()
    original = img.copy()
    draw_detections(img, [], class_names_by_id={0: "person"})
    assert np.array_equal(img, original)
