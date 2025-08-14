import cv2, numpy as np
from typing import Optional
from ..core.ports import Camera, Frame
from ..core.clock import SystemClock

class Cv2Camera(Camera):
    def __init__(self, src: str, clock=None):
        self.src = int(src) if src.isdigit() else src
        self.cap = None
        self.clock = clock or SystemClock()
        self.idx = 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.src)
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
