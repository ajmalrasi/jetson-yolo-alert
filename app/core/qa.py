from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import sqlite3
from typing import Any, List, Optional, Protocol
from zoneinfo import ZoneInfo

from .alert_history import AlertHistoryStore

IST = ZoneInfo("Asia/Kolkata")

MAX_ROWS = 50
SQL_TIMEOUT_SEC = 5

SCHEMA_DESCRIPTION = """
Table: alerts
Columns:
  id            INTEGER PRIMARY KEY AUTOINCREMENT
  ts            TEXT NOT NULL   -- ISO-8601 UTC timestamp of the alert
  count         INTEGER NOT NULL -- number of detected objects in this alert
  best_conf     REAL NOT NULL   -- highest detection confidence (0.0–1.0)
  image_path    TEXT            -- file path to snapshot image (may be NULL)
  trigger_classes TEXT NOT NULL DEFAULT '[]'  -- JSON array of class names that triggered the alert, e.g. '["person","dog"]'
  context_classes TEXT NOT NULL DEFAULT '[]'  -- JSON array of all class names visible in the scene

Index: idx_alerts_ts on ts
""".strip()


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_message: str) -> str: ...


@dataclass
class QAService:
    history: AlertHistoryStore
    llm: Optional[LLMClient] = None

    def answer_question(self, question: str) -> str:
        clean_q = question.strip()
        if not clean_q:
            return "Please provide a question."

        if self.llm is None:
            return "LLM is not configured. Set LLM_PROVIDER and API key in .env to enable Q&A."

        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        sql = self._generate_sql(clean_q, now_ist)
        if not sql:
            return "I could not understand your question. Try rephrasing it."

        rows, columns, error = _execute_readonly_sql(self.history.db_path, sql)
        if error:
            retry_sql = self._fix_sql(clean_q, sql, error, now_ist)
            if retry_sql:
                rows, columns, error = _execute_readonly_sql(self.history.db_path, retry_sql)
            if error:
                return f"I understood your question but the database query failed: {error}"

        result_text = _format_result_table(columns, rows)
        answer = self._generate_answer(clean_q, sql, result_text, now_ist)
        return answer

    def _generate_sql(self, question: str, now_ist: str) -> Optional[str]:
        system_prompt = (
            "You are a SQL assistant. Given a user question about an alert history database, "
            "write a single SQLite SELECT query to answer it.\n\n"
            f"Database schema:\n{SCHEMA_DESCRIPTION}\n\n"
            "Rules:\n"
            "- ONLY write SELECT queries. No INSERT, UPDATE, DELETE, DROP, ALTER, etc.\n"
            "- Timestamps (ts) are stored as ISO-8601 UTC strings.\n"
            "- The user is in India (IST = UTC+5:30). Convert dates accordingly.\n"
            "  For example, 'today' in IST means: ts >= '<IST midnight in UTC>' AND ts < '<IST next midnight in UTC>'\n"
            "  IST midnight = UTC previous day 18:30:00\n"
            "- trigger_classes and context_classes are JSON arrays stored as TEXT.\n"
            "  To filter by class, use: trigger_classes LIKE '%\"car\"%' or json_each.\n"
            f"- Limit results to {MAX_ROWS} rows.\n"
            "- Return ONLY the SQL query, no explanation, no markdown fences."
        )
        user_message = (
            f"Current time: {now_ist}\n"
            f"Question: {question}"
        )
        try:
            raw = self.llm.complete(system_prompt=system_prompt, user_message=user_message)
        except Exception:
            return None
        return _extract_sql(raw)

    def _fix_sql(self, question: str, bad_sql: str, error: str, now_ist: str) -> Optional[str]:
        system_prompt = (
            "You are a SQL assistant. A previous query failed. "
            "Fix the query based on the error.\n\n"
            f"Database schema:\n{SCHEMA_DESCRIPTION}\n\n"
            "Rules:\n"
            "- ONLY write SELECT queries.\n"
            f"- Limit results to {MAX_ROWS} rows.\n"
            "- Return ONLY the fixed SQL query, no explanation."
        )
        user_message = (
            f"Current time: {now_ist}\n"
            f"Question: {question}\n"
            f"Failed SQL: {bad_sql}\n"
            f"Error: {error}"
        )
        try:
            raw = self.llm.complete(system_prompt=system_prompt, user_message=user_message)
        except Exception:
            return None
        return _extract_sql(raw)

    def _generate_answer(self, question: str, sql: str, result_text: str, now_ist: str) -> str:
        system_prompt = (
            "You answer questions about an alert/detection history database. "
            "You are given the user's question, the SQL query that was run, and the results. "
            "Write a clear, concise, helpful answer using the data. "
            "Use India time (IST) for all timestamps. "
            "Do not show SQL or technical details. "
            "If the result is empty, say so clearly."
        )
        user_message = (
            f"Current time: {now_ist}\n"
            f"Question: {question}\n\n"
            f"SQL executed:\n{sql}\n\n"
            f"Results:\n{result_text}"
        )
        try:
            answer = self.llm.complete(system_prompt=system_prompt, user_message=user_message)
        except Exception:
            return result_text if result_text.strip() else "Query returned no results."
        return answer or result_text


def _extract_sql(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if not text.upper().lstrip().startswith("SELECT"):
        match = re.search(r"(SELECT\s.+)", text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            return None

    text = text.rstrip(";").strip()

    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
        re.IGNORECASE,
    )
    if forbidden.search(text):
        return None

    return text


def _execute_readonly_sql(
    db_path: str, sql: str
) -> tuple[List[tuple], List[str], Optional[str]]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=SQL_TIMEOUT_SEC)
        conn.row_factory = None
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_ROWS)
        conn.close()
        return rows, columns, None
    except Exception as e:
        return [], [], str(e)


def _format_result_table(columns: List[str], rows: List[tuple]) -> str:
    if not rows:
        return "(no results)"

    lines = [" | ".join(columns)]
    lines.append("-+-".join("-" * max(len(c), 5) for c in columns))
    for row in rows:
        lines.append(" | ".join(_fmt_cell(v) for v in row))

    if len(rows) >= MAX_ROWS:
        lines.append(f"... (limited to {MAX_ROWS} rows)")
    return "\n".join(lines)


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)
