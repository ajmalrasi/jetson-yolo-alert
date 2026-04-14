"""Tests for FrameStore: persistent connection, save/query cycle, cleanup, threading."""
import os
import sqlite3
import threading

import numpy as np

from app.core.frame_store import FrameStore
from app.core.ports import Detection


def _dummy_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Persistent connection
# ---------------------------------------------------------------------------

def test_persistent_connection_reused(tmp_path):
    """FrameStore should reuse a single SQLite connection across operations."""
    store = FrameStore(str(tmp_path / "frames"))
    conn1 = store._get_conn()
    conn2 = store._get_conn()
    assert conn1 is conn2


def test_init_creates_tables(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    conn = store._get_conn()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "frames" in tables


def test_init_creates_indexes(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    conn = store._get_conn()
    indexes = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()]
    assert "idx_frames_ts" in indexes
    assert "idx_frames_det" in indexes


# ---------------------------------------------------------------------------
# Save / query round-trip
# ---------------------------------------------------------------------------

def test_save_and_query_range(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    img = _dummy_image()

    from datetime import datetime, timezone
    ts1 = datetime(2026, 4, 14, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    ts2 = datetime(2026, 4, 14, 10, 5, 0, tzinfo=timezone.utc).timestamp()

    store.save_frame(img, ts1)
    store.save_frame(img, ts2, detections=[Detection((10, 10, 50, 50), 0.9, 0)],
                     class_names_by_id={0: "person"})

    records = store.query_range("2026-04-14T09:00:00", "2026-04-14T11:00:00")
    assert len(records) == 2
    assert records[0].has_detection is False
    assert records[1].has_detection is True
    assert "person" in records[1].detection_classes
    assert records[1].detection_count == 1
    assert records[1].best_conf == 0.9


def test_save_creates_jpeg_on_disk(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    img = _dummy_image()

    from datetime import datetime, timezone
    ts = datetime(2026, 4, 14, 12, 30, 0, tzinfo=timezone.utc).timestamp()
    path = store.save_frame(img, ts)

    assert os.path.isfile(path)
    assert path.endswith(".jpg")


def test_count_range(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    img = _dummy_image()

    from datetime import datetime, timezone
    for minute in range(5):
        ts = datetime(2026, 4, 14, 10, minute, 0, tzinfo=timezone.utc).timestamp()
        store.save_frame(img, ts)

    count = store.count_range("2026-04-14T10:00:00", "2026-04-14T10:05:00")
    assert count == 5


def test_query_with_class(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    img = _dummy_image()

    from datetime import datetime, timezone
    ts1 = datetime(2026, 4, 14, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    ts2 = datetime(2026, 4, 14, 10, 1, 0, tzinfo=timezone.utc).timestamp()

    store.save_frame(img, ts1, detections=[Detection((10, 10, 50, 50), 0.9, 0)],
                     class_names_by_id={0: "person"})
    store.save_frame(img, ts2, detections=[Detection((10, 10, 50, 50), 0.8, 1)],
                     class_names_by_id={1: "car"})

    person_records = store.query_with_class("2026-04-14T09:00:00", "2026-04-14T11:00:00", "person")
    assert len(person_records) == 1
    assert person_records[0].detection_classes == ("person",)

    car_records = store.query_with_class("2026-04-14T09:00:00", "2026-04-14T11:00:00", "car")
    assert len(car_records) == 1


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def test_cleanup_removes_old_frames(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    img = _dummy_image()

    import time
    old_ts = time.time() - 40 * 86400  # 40 days ago
    new_ts = time.time()

    old_path = store.save_frame(img, old_ts)
    new_path = store.save_frame(img, new_ts)

    deleted = store.cleanup(max_age_days=30)
    assert deleted == 1
    assert not os.path.isfile(old_path)
    assert os.path.isfile(new_path)

    remaining = store.count_range("1970-01-01T00:00:00", "2099-01-01T00:00:00")
    assert remaining == 1


# ---------------------------------------------------------------------------
# Cross-thread access (check_same_thread=False)
# ---------------------------------------------------------------------------

def test_query_from_different_thread(tmp_path):
    """FrameStore should allow reads from a thread other than the creator."""
    store = FrameStore(str(tmp_path / "frames"))
    img = _dummy_image()

    from datetime import datetime, timezone
    ts = datetime(2026, 4, 14, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    store.save_frame(img, ts)

    results = []
    errors = []

    def _read():
        try:
            records = store.query_range("2026-04-14T09:00:00", "2026-04-14T11:00:00")
            results.extend(records)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=_read)
    t.start()
    t.join(timeout=5)

    assert not errors, f"Cross-thread query failed: {errors[0]}"
    assert len(results) == 1
