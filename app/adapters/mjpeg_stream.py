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


def _viewer_html() -> bytes:
    """Single-page viewer: document loads immediately; <img> holds the long-lived MJPEG stream."""
    page = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>YOLO preview</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3d9eff;
      --border: #2d3a4d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: radial-gradient(1200px 800px at 50% -20%, #1a2840 0%, var(--bg) 55%);
      color: var(--text);
    }
    header {
      max-width: 1200px;
      margin: 0 auto;
      padding: 1.25rem 1.5rem 0.5rem;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem 1rem;
    }
    h1 {
      margin: 0;
      font-size: 1.35rem;
      font-weight: 600;
      letter-spacing: -0.02em;
    }
    .badge {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent);
      border: 1px solid var(--border);
      background: rgba(61, 158, 255, 0.08);
      padding: 0.35rem 0.65rem;
      border-radius: 999px;
    }
    .hint {
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 1.5rem 1rem;
      font-size: 0.875rem;
      color: var(--muted);
      line-height: 1.45;
    }
    .frame-wrap {
      max-width: 1200px;
      margin: 0 auto 2rem;
      padding: 0 1rem;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 24px 48px rgba(0, 0, 0, 0.35);
    }
    .card img {
      display: block;
      width: 100%;
      height: auto;
      vertical-align: middle;
      background: #000;
      min-height: 200px;
      object-fit: contain;
    }
    footer {
      text-align: center;
      padding: 1rem;
      font-size: 0.8rem;
      color: var(--muted);
    }
    code { font-size: 0.9em; background: rgba(0,0,0,0.25); padding: 0.1em 0.35em; border-radius: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>Live detection preview</h1>
    <span class="badge">MJPEG</span>
  </header>
  <p class="hint">
    This page finishes loading right away; the video loads inside the frame below.
    Direct stream URL: <code>/stream</code> (for VLC or embedding).
  </p>
  <div class="frame-wrap">
    <div class="card">
      <img src="/stream" alt="Live YOLO preview stream" width="1280" height="720"/>
    </div>
  </div>
  <footer>jetson-yolo-alert preview</footer>
</body>
</html>
"""
    return page.encode("utf-8")


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
        self._jpeg_seq: int = 0
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
                    self._jpeg_seq += 1

    def _make_handler(self):
        server = self
        min_interval = self._min_frame_interval
        html_body = _viewer_html()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(html_body)
                    return
                if path == "/stream":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.end_headers()
                    boundary = b"--frame\r\n"
                    last_sent_seq = -1
                    last_write_t = 0.0
                    try:
                        while not server._stop.is_set():
                            with server._jpeg_lock:
                                jpg = server._jpeg
                                seq = server._jpeg_seq
                            if jpg is None:
                                time.sleep(0.02)
                                continue
                            if seq == last_sent_seq:
                                time.sleep(0.004)
                                continue
                            now = time.monotonic()
                            if last_write_t > 0 and now - last_write_t < min_interval:
                                time.sleep(min_interval - (now - last_write_t))
                            last_sent_seq = seq
                            self.wfile.write(boundary)
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            last_write_t = time.monotonic()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_error(404)

        return Handler
