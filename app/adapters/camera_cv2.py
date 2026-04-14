import logging
import os
import threading
import time
import cv2
from typing import Optional, Tuple
from ..core.ports import Camera, Frame
from ..core.clock import SystemClock

logger = logging.getLogger(__name__)


class Cv2Camera(Camera):
    def __init__(self, src: str, clock=None):
        # Accept integer-like strings ("0", "1") as device index
        self.src = int(src) if isinstance(src, str) and src.isdigit() else src
        self.cap: Optional[cv2.VideoCapture] = None
        self.clock = clock or SystemClock()
        self.idx = 0
        self._grab_flush = max(0, int(os.getenv("CAMERA_GRAB_FLUSH", "0")))

    def open(self) -> None:
        if isinstance(self.src, str) and self.src.startswith("rtsp://"):
            use_gst = os.getenv("USE_GSTREAMER", "0") not in ("0", "false", "False", "")
            latency = int(os.getenv("RTSP_LATENCY_MS", "200"))

            if not use_gst:
                # Try FFmpeg first
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                try:
                    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                        self.cap.set(
                            cv2.CAP_PROP_BUFFERSIZE,
                            int(os.getenv("CAP_PROP_BUFFERSIZE", "1")),
                        )
                except Exception:
                    logger.debug("CAP_PROP_BUFFERSIZE not supported by backend")

                # Auto-fallback to GStreamer if FFmpeg failed
                if not self.cap or not self.cap.isOpened():
                    use_gst = True

            if use_gst:
                pipe = (
                    f"rtspsrc location={self.src} protocols=tcp latency={latency} ! "
                    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
                    "appsink drop=1 sync=false max-buffers=1"
                )
                self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        else:
            # USB cam / file / numeric index
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            try:
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    self.cap.set(
                        cv2.CAP_PROP_BUFFERSIZE,
                        int(os.getenv("CAP_PROP_BUFFERSIZE", "1")),
                    )
            except Exception:
                logger.debug("CAP_PROP_BUFFERSIZE not supported by backend")

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.src}")

    def grab(self) -> bool:
        """Advance the camera buffer without decoding. Very cheap (~0.1ms)."""
        if self.cap is None:
            return False
        return self.cap.grab()

    def read(self) -> Optional[Frame]:
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        for _ in range(self._grab_flush):
            self.cap.grab()
        ok, img = self.cap.read()
        if not ok:
            return None

        # increment sequential index
        self.idx += 1

        # width & height from the actual frame (robust across backends)
        try:
            h, w = img.shape[:2]
        except Exception:
            # fallback to capture props if needed
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        return Frame(
            image=img,
            t=self.clock.now(),
            index=self.idx,
            w=w,
            h=h,
        )

    def release(self) -> None:
        self.close()

    def close(self) -> None:
        if self.cap:
            try:
                self.cap.release()
            finally:
                self.cap = None

    def shape(self) -> Optional[Tuple[int, int]]:
        if self.cap is None:
            return None
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (h, w) if (h and w) else None


class ThreadedCamera:
    """Always-latest-frame wrapper for zero-lag RTSP preview.

    A background thread continuously reads from the underlying camera so the
    consumer always gets the newest frame — the RTSP/FFmpeg internal buffer
    never builds up regardless of how long the main thread spends on inference.
    """

    def __init__(self, inner: Cv2Camera):
        self._inner = inner
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        self._inner.open()
        self._running = True
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        cap = self._inner.cap
        clock = self._inner.clock
        idx = 0
        while self._running:
            ok, img = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            idx += 1
            h, w = img.shape[:2]
            with self._lock:
                self._latest = Frame(image=img, t=clock.now(), index=idx, w=w, h=h)

    def grab(self) -> bool:
        return True

    def read(self) -> Optional[Frame]:
        with self._lock:
            frame = self._latest
            self._latest = None
        if frame is None:
            time.sleep(0.005)
        return frame

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._inner.close()

    def release(self) -> None:
        self.close()

    def shape(self) -> Optional[Tuple[int, int]]:
        return self._inner.shape()
