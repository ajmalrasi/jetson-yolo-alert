"""Tests for video understanding: grab-based skipping, frame count header, clustering."""
from unittest.mock import MagicMock, patch

import numpy as np

from app.core.video_understanding import (
    VideoUnderstandingService,
    _cluster_by_time,
    _build_cluster,
)
from app.core.frame_store import FrameStore, FrameRecord


# ---------------------------------------------------------------------------
# describe_video: grab() for skipped frames
# ---------------------------------------------------------------------------

def test_describe_video_uses_grab_for_skipped_frames(tmp_path):
    """describe_video should call grab() on skipped frames, not read()."""
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store,
        vlm_model="test/model",
        llm_model="test/llm",
        vlm_max_frames=3,
        vlm_max_width=128,
    )

    total_frames = 90  # 3 seconds at 30fps
    grabbed = []
    read_count = [0]

    class FakeCap:
        def __init__(self):
            self._idx = 0

        def isOpened(self):
            return True

        def get(self, prop):
            import cv2
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return total_frames
            return 0

        def grab(self):
            if self._idx >= total_frames:
                return False
            grabbed.append(self._idx)
            self._idx += 1
            return True

        def read(self):
            if self._idx >= total_frames:
                return False, None
            read_count[0] += 1
            self._idx += 1
            img = np.zeros((120, 160, 3), dtype=np.uint8)
            return True, img

        def release(self):
            pass

    with patch("cv2.VideoCapture", return_value=FakeCap()), \
         patch("app.core.video_understanding.describe_frames", return_value="test narrative"):
        result = svc.describe_video("/fake/video.mp4")

    assert "3 frames sampled" in result
    assert len(grabbed) > 0, "grab() should have been called for skipped frames"
    assert read_count[0] == 3, f"Only 3 reads expected (vlm_max_frames=3), got {read_count[0]}"


# ---------------------------------------------------------------------------
# describe_timerange: accurate "Analyzed X of Y" header
# ---------------------------------------------------------------------------

def test_describe_timerange_header_shows_correct_counts(tmp_path):
    """Header should say 'Analyzed <vlm_frames> of <total_frames>'."""
    store = FrameStore(str(tmp_path / "frames"))

    from datetime import datetime, timezone
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    # 20 frames across 10 distinct clusters (2 frames each, 5 min apart)
    for cluster in range(10):
        for j in range(2):
            ts = datetime(2026, 4, 14, 10, cluster * 5, j, tzinfo=timezone.utc).timestamp()
            store.save_frame(img, ts)

    svc = VideoUnderstandingService(
        frame_store=store,
        vlm_model="test/model",
        llm_model="test/llm",
        vlm_max_frames=5,
        vlm_max_width=128,
    )

    with patch.object(svc, "_parse_time_range",
                      return_value=("2026-04-14T09:00:00", "2026-04-14T11:00:00")), \
         patch("app.core.video_understanding.describe_frames", return_value="test"):
        result = svc.describe_timerange("last hour")

    assert "5 of 20 frames" in result


def test_describe_timerange_small_range_shows_all(tmp_path):
    """When total frames <= vlm_max_frames, should show 'N of N'."""
    store = FrameStore(str(tmp_path / "frames"))

    from datetime import datetime, timezone
    for i in range(3):
        ts = datetime(2026, 4, 14, 10, 0, i, tzinfo=timezone.utc).timestamp()
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        store.save_frame(img, ts)

    svc = VideoUnderstandingService(
        frame_store=store,
        vlm_model="test/model",
        llm_model="test/llm",
        vlm_max_frames=15,
        vlm_max_width=128,
    )

    with patch.object(svc, "_parse_time_range",
                      return_value=("2026-04-14T09:00:00", "2026-04-14T11:00:00")), \
         patch("app.core.video_understanding.describe_frames", return_value="test"):
        result = svc.describe_timerange("last hour")

    assert "3 of 3 frames" in result


# ---------------------------------------------------------------------------
# _cluster_by_time
# ---------------------------------------------------------------------------

def _make_record(ts: str, has_det=False, classes=(), conf=0.0, count=0) -> FrameRecord:
    return FrameRecord(
        ts=ts,
        path=f"/fake/{ts}.jpg",
        has_detection=has_det,
        detection_classes=tuple(classes),
        detection_count=count,
        best_conf=conf,
    )


def test_cluster_splits_on_time_gap():
    """Frames >60s apart should land in different clusters."""
    records = [
        _make_record("2026-04-14T10:00:00"),
        _make_record("2026-04-14T10:00:05"),
        _make_record("2026-04-14T10:05:00"),  # 5 min gap -> new cluster
        _make_record("2026-04-14T10:05:10"),
    ]
    clusters = _cluster_by_time(records, gap_sec=60)
    assert len(clusters) == 2
    assert clusters[0]["count"] == 2
    assert clusters[1]["count"] == 2


def test_cluster_single_event_stays_together():
    """Frames within 60s should be one cluster."""
    records = [
        _make_record("2026-04-14T10:00:00"),
        _make_record("2026-04-14T10:00:10"),
        _make_record("2026-04-14T10:00:30"),
        _make_record("2026-04-14T10:00:50"),
    ]
    clusters = _cluster_by_time(records, gap_sec=60)
    assert len(clusters) == 1
    assert clusters[0]["count"] == 4


def test_cluster_empty_records():
    assert _cluster_by_time([], gap_sec=60) == []


def test_cluster_best_frame_is_highest_conf():
    """best_frame should be the detection frame with highest confidence."""
    records = [
        _make_record("2026-04-14T10:00:00", has_det=True, classes=("person",), conf=0.7, count=1),
        _make_record("2026-04-14T10:00:05", has_det=True, classes=("person",), conf=0.9, count=1),
        _make_record("2026-04-14T10:00:10", has_det=True, classes=("person",), conf=0.6, count=1),
    ]
    clusters = _cluster_by_time(records, gap_sec=60)
    assert len(clusters) == 1
    assert clusters[0]["best_frame"].best_conf == 0.9


def test_cluster_idle_best_frame_is_middle():
    """For an all-idle cluster, best_frame should be the middle frame."""
    records = [
        _make_record("2026-04-14T10:00:00"),
        _make_record("2026-04-14T10:00:10"),
        _make_record("2026-04-14T10:00:20"),
        _make_record("2026-04-14T10:00:30"),
        _make_record("2026-04-14T10:00:40"),
    ]
    clusters = _cluster_by_time(records, gap_sec=60)
    assert clusters[0]["best_frame"].ts == "2026-04-14T10:00:20"


def test_cluster_aggregates_classes():
    """Cluster should collect all detection classes from its frames."""
    records = [
        _make_record("2026-04-14T10:00:00", has_det=True, classes=("person",), conf=0.8, count=1),
        _make_record("2026-04-14T10:00:05", has_det=True, classes=("car",), conf=0.7, count=1),
        _make_record("2026-04-14T10:00:10", has_det=True, classes=("person", "dog"), conf=0.9, count=2),
    ]
    clusters = _cluster_by_time(records, gap_sec=60)
    assert set(clusters[0]["classes"]) == {"car", "dog", "person"}


def test_cluster_duration():
    records = [
        _make_record("2026-04-14T10:00:00"),
        _make_record("2026-04-14T10:00:30"),
    ]
    clusters = _cluster_by_time(records, gap_sec=60)
    assert clusters[0]["duration_sec"] == 30.0


# ---------------------------------------------------------------------------
# _smart_sample
# ---------------------------------------------------------------------------

def test_smart_sample_picks_one_per_cluster(tmp_path):
    """With 3 clusters and max_frames=3, should pick one from each."""
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store, vlm_model="x", llm_model="x", vlm_max_frames=3,
    )
    records = [
        _make_record("2026-04-14T10:00:00", has_det=True, classes=("person",), conf=0.8, count=1),
        _make_record("2026-04-14T10:00:05", has_det=True, classes=("person",), conf=0.7, count=1),
        # gap
        _make_record("2026-04-14T10:05:00", has_det=True, classes=("car",), conf=0.9, count=1),
        _make_record("2026-04-14T10:05:10"),
        # gap
        _make_record("2026-04-14T10:10:00", has_det=True, classes=("dog",), conf=0.6, count=1),
        _make_record("2026-04-14T10:10:05"),
    ]
    sampled = svc._smart_sample(records)
    assert len(sampled) == 3
    timestamps = {r.ts for r in sampled}
    assert "2026-04-14T10:00:00" in timestamps  # best from cluster 1
    assert "2026-04-14T10:05:00" in timestamps  # best from cluster 2
    assert "2026-04-14T10:10:00" in timestamps  # best from cluster 3


def test_smart_sample_prefers_detections_over_idle(tmp_path):
    """Clusters with detections should be chosen over idle clusters."""
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store, vlm_model="x", llm_model="x", vlm_max_frames=2,
    )
    records = [
        _make_record("2026-04-14T10:00:00"),  # idle cluster
        _make_record("2026-04-14T10:00:05"),
        # gap
        _make_record("2026-04-14T10:05:00", has_det=True, classes=("person",), conf=0.8, count=1),
        # gap
        _make_record("2026-04-14T10:10:00"),  # idle cluster
        _make_record("2026-04-14T10:10:05"),
        # gap
        _make_record("2026-04-14T10:15:00", has_det=True, classes=("car",), conf=0.7, count=1),
    ]
    sampled = svc._smart_sample(records)
    assert len(sampled) == 2
    assert all(r.has_detection for r in sampled)


def test_smart_sample_returns_all_when_under_limit(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store, vlm_model="x", llm_model="x", vlm_max_frames=10,
    )
    records = [_make_record(f"2026-04-14T10:00:0{i}") for i in range(3)]
    sampled = svc._smart_sample(records)
    assert len(sampled) == 3


def test_smart_sample_output_sorted_by_time(tmp_path):
    """Sampled frames should be in chronological order."""
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store, vlm_model="x", llm_model="x", vlm_max_frames=3,
    )
    records = [
        _make_record("2026-04-14T10:00:00", has_det=True, classes=("person",), conf=0.5, count=1),
        # gap
        _make_record("2026-04-14T10:05:00", has_det=True, classes=("car",), conf=0.9, count=1),
        # gap
        _make_record("2026-04-14T10:10:00", has_det=True, classes=("dog",), conf=0.7, count=1),
    ]
    sampled = svc._smart_sample(records)
    timestamps = [r.ts for r in sampled]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# _build_event_timeline
# ---------------------------------------------------------------------------

def test_event_timeline_includes_all_detection_clusters(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store, vlm_model="x", llm_model="x", vlm_max_frames=2,
    )
    records = [
        _make_record("2026-04-14T10:00:00", has_det=True, classes=("person",), conf=0.8, count=1),
        # gap
        _make_record("2026-04-14T10:05:00"),  # idle
        # gap
        _make_record("2026-04-14T10:10:00", has_det=True, classes=("car",), conf=0.7, count=1),
        # gap
        _make_record("2026-04-14T10:15:00", has_det=True, classes=("dog",), conf=0.6, count=1),
    ]
    timeline = svc._build_event_timeline(records)
    assert "person" in timeline
    assert "car" in timeline
    assert "dog" in timeline
    assert "YOLO detection timeline" in timeline


def test_event_timeline_empty_for_no_detections(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store, vlm_model="x", llm_model="x", vlm_max_frames=5,
    )
    records = [
        _make_record("2026-04-14T10:00:00"),
        _make_record("2026-04-14T10:00:10"),
    ]
    timeline = svc._build_event_timeline(records)
    assert timeline == ""


def test_event_timeline_shows_frame_count_for_multi_frame_clusters(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    svc = VideoUnderstandingService(
        frame_store=store, vlm_model="x", llm_model="x", vlm_max_frames=5,
    )
    records = [
        _make_record("2026-04-14T10:00:00", has_det=True, classes=("person",), conf=0.8, count=1),
        _make_record("2026-04-14T10:00:10", has_det=True, classes=("person",), conf=0.7, count=1),
        _make_record("2026-04-14T10:00:20", has_det=True, classes=("person",), conf=0.9, count=1),
    ]
    timeline = svc._build_event_timeline(records)
    assert "3 frames" in timeline
