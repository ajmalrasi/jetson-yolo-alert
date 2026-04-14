import threading
import time

import numpy as np

from app.core.ports import Frame
from app.adapters.camera_cv2 import Cv2Camera, ThreadedCamera


class StubCv2Camera:
    """Minimal stub that quacks like Cv2Camera without needing real hardware."""

    def __init__(self, fps: float = 30.0, grab_flush: int = 0):
        self.cap = _FakeCap(fps)
        self.clock = _FakeClock()
        self.idx = 0
        self._grab_flush = grab_flush

    def open(self):
        pass

    def grab(self) -> bool:
        return self.cap.grab()

    def read(self):
        for _ in range(self._grab_flush):
            self.cap.grab()
        ok, img = self.cap.read()
        if not ok:
            return None
        self.idx += 1
        h, w = img.shape[:2]
        return Frame(image=img, t=self.clock.now(), index=self.idx, w=w, h=h)

    def close(self):
        pass

    def release(self):
        self.close()

    def shape(self):
        return (480, 640)


class _FakeCap:
    def __init__(self, fps: float):
        self._interval = 1.0 / fps
        self._grab_count = 0
        self._read_count = 0

    def grab(self) -> bool:
        self._grab_count += 1
        return True

    def read(self):
        self._read_count += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, img


class _FakeClock:
    def __init__(self):
        self._t = 0.0

    def now(self) -> float:
        self._t += 0.033
        return self._t


# ---------------------------------------------------------------------------
# Cv2Camera: _grab_flush hoisted to __init__
# ---------------------------------------------------------------------------

def test_grab_flush_hoisted_to_init(monkeypatch):
    """CAMERA_GRAB_FLUSH should be read once at init, not per-read."""
    monkeypatch.setenv("CAMERA_GRAB_FLUSH", "5")
    cam = Cv2Camera("0")
    assert cam._grab_flush == 5


def test_grab_flush_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("CAMERA_GRAB_FLUSH", raising=False)
    cam = Cv2Camera("0")
    assert cam._grab_flush == 0


def test_grab_flush_negative_clamped(monkeypatch):
    monkeypatch.setenv("CAMERA_GRAB_FLUSH", "-3")
    cam = Cv2Camera("0")
    assert cam._grab_flush == 0


# ---------------------------------------------------------------------------
# ThreadedCamera
# ---------------------------------------------------------------------------

def test_threaded_camera_returns_latest_frame():
    inner = StubCv2Camera()
    cam = ThreadedCamera(inner)
    cam.open()
    time.sleep(0.1)
    frame = cam.read()
    assert frame is not None
    assert isinstance(frame, Frame)
    assert frame.w == 640
    assert frame.h == 480
    cam.close()


def test_threaded_camera_consumes_frame_on_read():
    """read() should return the frame and clear it (consume-once semantics)."""
    inner = StubCv2Camera()
    cam = ThreadedCamera(inner)
    cam.open()
    time.sleep(0.1)
    f1 = cam.read()
    assert f1 is not None
    # Immediately read again before the drain thread refills
    f2 = cam.read()
    # f2 might be None (consumed) or a newer frame -- but should NOT be
    # the exact same object as f1
    if f2 is not None:
        assert f2.index != f1.index or f2.t != f1.t
    cam.close()


def test_threaded_camera_grab_is_noop():
    inner = StubCv2Camera()
    cam = ThreadedCamera(inner)
    assert cam.grab() is True


def test_threaded_camera_close_stops_thread():
    inner = StubCv2Camera()
    cam = ThreadedCamera(inner)
    cam.open()
    assert cam._thread is not None
    assert cam._thread.is_alive()
    cam.close()
    assert not cam._running
    assert not cam._thread.is_alive()
