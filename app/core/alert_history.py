from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import sqlite3
from typing import Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

_UTC_TS_FMT = "%Y-%m-%dT%H:%M:%S"


@dataclass(frozen=True)
class AlertRecord:
    id: int
    ts: datetime
    count: int
    best_conf: float
    image_path: Optional[str]
    trigger_classes: Tuple[str, ...]
    context_classes: Tuple[str, ...]


class AlertHistoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_parent_dir()
        self.init_db()

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    best_conf REAL NOT NULL,
                    image_path TEXT,
                    trigger_classes TEXT NOT NULL DEFAULT '[]',
                    context_classes TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            _ensure_column(
                conn,
                table_name="alerts",
                column_name="trigger_classes",
                column_def="TEXT NOT NULL DEFAULT '[]'",
            )
            _ensure_column(
                conn,
                table_name="alerts",
                column_name="context_classes",
                column_def="TEXT NOT NULL DEFAULT '[]'",
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts)")
            self._migrate_ts_format(conn)
            conn.commit()

    @staticmethod
    def _migrate_ts_format(conn: sqlite3.Connection) -> None:
        """Strip '+00:00' suffix from legacy timestamps for consistent text comparisons."""
        updated = conn.execute(
            "UPDATE alerts SET ts = REPLACE(ts, '+00:00', '') WHERE ts LIKE '%+00:00'"
        ).rowcount
        if updated:
            logger.info("Migrated %d timestamps: stripped +00:00 suffix", updated)

    def insert_alert(
        self,
        ts: float,
        count: int,
        best_conf: float,
        image_path: Optional[str],
        trigger_classes: Optional[Iterable[str]] = None,
        context_classes: Optional[Iterable[str]] = None,
    ) -> None:
        ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(_UTC_TS_FMT)
        trigger_classes_json = _classes_to_json(trigger_classes)
        context_classes_json = _classes_to_json(context_classes)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts(ts, count, best_conf, image_path, trigger_classes, context_classes)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    int(count),
                    float(best_conf),
                    image_path,
                    trigger_classes_json,
                    context_classes_json,
                ),
            )
            conn.commit()

    def get_alerts_between(self, start_ts: datetime, end_ts: datetime) -> List[AlertRecord]:
        start_utc = _to_utc(start_ts).strftime(_UTC_TS_FMT)
        end_utc = _to_utc(end_ts).strftime(_UTC_TS_FMT)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, ts, count, best_conf, image_path, trigger_classes, context_classes
                FROM alerts
                WHERE ts >= ? AND ts < ?
                ORDER BY ts ASC
                """,
                (start_utc, end_utc),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def get_alerts_on_date(self, date_str: str) -> List[AlertRecord]:
        day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=IST)
        next_day = day + timedelta(days=1)
        return self.get_alerts_between(day, next_day)

    def get_last_alert(self) -> Optional[AlertRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, ts, count, best_conf, image_path, trigger_classes, context_classes
                FROM alerts
                ORDER BY ts DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _classes_to_json(classes: Optional[Iterable[str]]) -> str:
    if not classes:
        return "[]"
    normalized = sorted({str(c).strip().lower() for c in classes if str(c).strip()})
    return json.dumps(normalized)


def _parse_classes(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return ()
    if not isinstance(parsed, list):
        return ()
    items = []
    for item in parsed:
        if isinstance(item, str) and item.strip():
            items.append(item.strip().lower())
    return tuple(sorted(set(items)))


def _has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row["name"]) == column_name for row in rows)


def _ensure_column(
    conn: sqlite3.Connection, table_name: str, column_name: str, column_def: str
) -> None:
    if _has_column(conn, table_name=table_name, column_name=column_name):
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def _row_to_record(row: sqlite3.Row) -> AlertRecord:
    ts = datetime.fromisoformat(str(row["ts"]))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return AlertRecord(
        id=int(row["id"]),
        ts=ts,
        count=int(row["count"]),
        best_conf=float(row["best_conf"]),
        image_path=row["image_path"],
        trigger_classes=_parse_classes(row["trigger_classes"]),
        context_classes=_parse_classes(row["context_classes"]),
    )
