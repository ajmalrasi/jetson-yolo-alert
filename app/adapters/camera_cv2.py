import os
import cv2
from typing import Optional, Tuple
from ..core.ports import Camera, Frame
from ..core.clock import SystemClock


class Cv2Camera(Camera):
    def __init__(self, src: str, clock=None):
        # Accept integer-like strings ("0", "1") as device index
        self.src = int(src) if isinstance(src, str) and src.isdigit() else src
        self.cap: Optional[cv2.VideoCapture] = None
        self.clock = clock or SystemClock()
        self.idx = 0

    def open(self) -> None:
        if isinstance(self.src, str) and self.src.startswith("rtsp://"):
            use_gst = os.getenv("USE_GSTREAMER", "0") not in ("0", "false", "False", "")
            latency = int(os.getenv("RTSP_LATENCY_MS", "200"))

            if not use_gst:
                # Try FFmpeg first
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                try:
                    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv("CAP_PROP_BUFFERSIZE", "2")))
                except Exception:
                    pass

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

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.src}")

    def read(self) -> Optional[Frame]:
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
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
