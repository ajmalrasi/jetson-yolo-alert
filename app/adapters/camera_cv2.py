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
        self.cap = cv2.VideoCapture(self.src)
        if isinstance(self.src, str) and self.src.startswith("rtsp://") and os.getenv("USE_GSTREAMER", "0") not in ("0","false","False",""):
            latency = int(os.getenv("RTSP_LATENCY_MS", "200"))
            pipe = (
                f"rtspsrc location={self.src} protocols=tcp latency={latency} ! "
                "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
            )
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv("CAP_PROP_BUFFERSIZE", "2")))
            except: pass
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.src}")

    def read(self) -> Optional[Frame]:
        ok, img = self.cap.read()
        if not ok: return None
        self.idx += 1
        h, w = img.shape[:2]
        return Frame(image=img, t=self.clock.now(), index=self.idx, w=w, h=h)

    def close(self) -> None:
        if self.cap: self.cap.release()
