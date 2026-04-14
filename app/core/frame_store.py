from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set

import cv2

logger = logging.getLogger(__name__)

_UTC_FMT = "%Y-%m-%dT%H:%M:%S"


@dataclass(frozen=True)
class FrameRecord:
    ts: str
    path: str
    has_detection: bool
    detection_classes: tuple[str, ...]
    detection_count: int
    best_conf: float


class FrameStore:
    """Saves captured frames to disk and indexes metadata in SQLite."""

    def __init__(self, frames_dir: str, db_path: str | None = None):
        self.frames_dir = frames_dir
        os.makedirs(frames_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(frames_dir, "frame_index.db")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS frames (
                    ts TEXT NOT NULL,
                    path TEXT NOT NULL,
                    has_detection INTEGER NOT NULL DEFAULT 0,
                    detection_classes TEXT NOT NULL DEFAULT '[]',
                    detection_count INTEGER NOT NULL DEFAULT 0,
                    best_conf REAL NOT NULL DEFAULT 0.0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_frames_det ON frames(has_detection, ts)"
            )
            conn.commit()

    def save_frame(
        self,
        image,
        ts: float,
        detections: Sequence | None = None,
        class_names_by_id: Dict[int, str] | None = None,
        jpeg_quality: int = 80,
    ) -> str:
        """Write a JPEG to disk and index it. Returns the absolute image path."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        ts_iso = dt.strftime(_UTC_FMT)

        day_dir = os.path.join(self.frames_dir, dt.strftime("%Y-%m-%d"), dt.strftime("%H"))
        os.makedirs(day_dir, exist_ok=True)
        fname = dt.strftime("%M-%S") + f"-{int(ts * 1000) % 1000:03d}.jpg"
        img_path = os.path.join(day_dir, fname)

        cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        det_classes: list[str] = []
        det_count = 0
        best_conf = 0.0
        if detections:
            names_map = class_names_by_id or {}
            cls_set: Set[str] = set()
            for d in detections:
                cls_set.add(names_map.get(d.cls_id, f"class_{d.cls_id}"))
                if d.conf > best_conf:
                    best_conf = d.conf
            det_classes = sorted(cls_set)
            det_count = len(detections)

        has_det = 1 if det_count > 0 else 0
        classes_json = json.dumps(det_classes)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO frames(ts, path, has_detection, detection_classes, detection_count, best_conf) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (ts_iso, img_path, has_det, classes_json, det_count, best_conf),
            )
            conn.commit()

        return img_path

    def query_range(self, start_utc: str, end_utc: str) -> List[FrameRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, path, has_detection, detection_classes, detection_count, best_conf "
                "FROM frames WHERE ts >= ? AND ts < ? ORDER BY ts ASC",
                (start_utc, end_utc),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def query_with_class(
        self, start_utc: str, end_utc: str, class_name: str
    ) -> List[FrameRecord]:
        pattern = f'%"{class_name}"%'
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, path, has_detection, detection_classes, detection_count, best_conf "
                "FROM frames WHERE ts >= ? AND ts < ? AND detection_classes LIKE ? ORDER BY ts ASC",
                (start_utc, end_utc, pattern),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def count_range(self, start_utc: str, end_utc: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM frames WHERE ts >= ? AND ts < ?",
                (start_utc, end_utc),
            ).fetchone()
        return int(row["c"]) if row else 0

    def cleanup(self, max_age_days: int) -> int:
        """Delete frames older than max_age_days. Returns count of deleted rows."""
        cutoff_ts = time.time() - max_age_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime(_UTC_FMT)

        deleted = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path FROM frames WHERE ts < ?", (cutoff_iso,)
            ).fetchall()
            for row in rows:
                try:
                    os.remove(row["path"])
                except OSError:
                    pass
                deleted += 1
            conn.execute("DELETE FROM frames WHERE ts < ?", (cutoff_iso,))
            conn.commit()

        self._cleanup_empty_dirs()
        logger.info("Frame cleanup: removed %d frames older than %d days", deleted, max_age_days)
        return deleted

    def _cleanup_empty_dirs(self) -> None:
        for root, dirs, files in os.walk(self.frames_dir, topdown=False):
            if root == self.frames_dir:
                continue
            if not dirs and not files:
                try:
                    shutil.rmtree(root, ignore_errors=True)
                except OSError:
                    pass


def _parse_classes(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return ()
    if isinstance(parsed, list):
        return tuple(str(c) for c in parsed if c)
    return ()


def _row_to_record(row: sqlite3.Row) -> FrameRecord:
    return FrameRecord(
        ts=str(row["ts"]),
        path=str(row["path"]),
        has_detection=bool(row["has_detection"]),
        detection_classes=_parse_classes(row["detection_classes"]),
        detection_count=int(row["detection_count"]),
        best_conf=float(row["best_conf"]),
    )
