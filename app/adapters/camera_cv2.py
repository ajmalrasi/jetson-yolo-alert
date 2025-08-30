import cv2, numpy as np
from typing import Optional
from ..core.ports import Camera, Frame
from ..core.clock import SystemClock
import os

class Cv2Camera(Camera):
    def __init__(self, src: str, clock=None):
        self.src = int(src) if src.isdigit() else src
        self.cap = None
        self.clock = clock or SystemClock()
        self.idx = 0

def open(self) -> None:
    # Prefer explicit backend selection; no pre-open
    if isinstance(self.src, str) and self.src.startswith("rtsp://"):
        use_gst = os.getenv("USE_GSTREAMER", "0") not in ("0", "false", "False", "")
        latency = int(os.getenv("RTSP_LATENCY_MS", "200"))

        if not use_gst:
            # Try FFmpeg first
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv("CAP_PROP_BUFFERSIZE", "2")))
            except Exception:
                pass

            # Auto-fallback to GStreamer if FFmpeg failed
            if not self.cap.isOpened():
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

    if not self.cap or not self.cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {self.src}")

    def release(self) -> None:
        self.close()

    def close(self) -> None:
        if self.cap:
            try:
                self.cap.release()
            finally:
                self.cap = None