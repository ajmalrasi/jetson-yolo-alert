"""
MJPEG over HTTP for remote preview: encoder thread + drop-old-frames queue so
inference never blocks on network or slow clients.
"""
from __future__ import annotations

import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np


class MjpegStreamServer:
    def __init__(
        self,
        host: str,
        port: int,
        max_width: int = 1280,
        quality: int = 80,
        max_fps: float = 25.0,
    ):
        self._host = host
        self._port = port
        self._max_width = max(0, int(max_width))
        self._quality = int(np.clip(quality, 30, 95))
        self._min_frame_interval = 1.0 / max(1.0, float(max_fps))
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._jpeg: Optional[bytes] = None
        self._jpeg_lock = threading.Lock()
        self._stop = threading.Event()
        self._encoder_thread = threading.Thread(
            target=self._encode_loop, name="mjpeg-encode", daemon=True
        )
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._encoder_thread.start()
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self._host, self._port), handler)
        self._httpd.daemon_threads = True
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, name="mjpeg-http", daemon=True
        )
        self._http_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
        self._encoder_thread.join(timeout=3.0)

    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        if self._stop.is_set():
            return
        try:
            self._frame_q.put_nowait(frame_bgr)
        except queue.Full:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_q.put_nowait(frame_bgr)
            except queue.Full:
                pass

    def _encode_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frame_q.get(timeout=0.25)
            except queue.Empty:
                continue
            h, w = frame.shape[:2]
            if self._max_width > 0 and w > self._max_width:
                scale = self._max_width / float(w)
                frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._quality],
            )
            if ok:
                data = buf.tobytes()
                with self._jpeg_lock:
                    self._jpeg = data

    def _make_handler(self):
        server = self
        min_interval = self._min_frame_interval

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def do_GET(self) -> None:
                if self.path not in ("/", "/stream"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                boundary = b"--frame\r\n"
                last = 0.0
                try:
                    while not server._stop.is_set():
                        now = time.monotonic()
                        if now - last < min_interval:
                            time.sleep(min_interval - (now - last))
                        last = time.monotonic()
                        with server._jpeg_lock:
                            jpg = server._jpeg
                        if jpg is None:
                            time.sleep(0.02)
                            continue
                        self.wfile.write(boundary)
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler
