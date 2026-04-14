"""Tests for video understanding: grab-based skipping, frame count header."""
from unittest.mock import MagicMock, patch

import numpy as np

from app.core.video_understanding import VideoUnderstandingService
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

    # Insert 20 frames
    from datetime import datetime, timezone
    for i in range(20):
        ts = datetime(2026, 4, 14, 10, 0, i, tzinfo=timezone.utc).timestamp()
        img = np.zeros((120, 160, 3), dtype=np.uint8)
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
