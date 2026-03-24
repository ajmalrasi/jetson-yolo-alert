from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerResult:
    """Structured result from QAService.answer_question."""
    text: str
    image_path: Optional[str] = None

    def __str__(self) -> str:
        return self.text

IST = ZoneInfo("Asia/Kolkata")

_SCHEMA = """\
TABLE: alerts
  id            INTEGER PRIMARY KEY
  ts            TEXT    -- ISO-8601 UTC timestamp
  count         INTEGER -- number of objects detected
  best_conf     REAL    -- highest confidence score
  image_path    TEXT
  trigger_classes TEXT  -- JSON array, e.g. '["person","car"]'
  context_classes TEXT  -- JSON array, e.g. '["person","car","dog"]'
INDEX: idx_alerts_ts ON alerts(ts)"""

_SQL_PROMPT = """\
You are a SQLite expert. Given the schema and question, write ONE SELECT query.

{schema}

Rules:
- ts is stored as ISO-8601 UTC. Current time: {now_utc} UTC ({now_ist} IST). User is in IST (UTC+5:30).
- For "today" in IST, convert to UTC range: WHERE ts >= datetime('{today_utc_start}') AND ts < datetime('{tomorrow_utc_start}')
- Classes are JSON arrays in TEXT columns. Use json_each() or LIKE '%"classname"%' to filter.
- When filtering by a class name, ALWAYS search BOTH trigger_classes AND context_classes:
  WHERE (trigger_classes LIKE '%"dog"%' OR context_classes LIKE '%"dog"%')
- When the user asks for a picture/image/photo/pic, always include image_path in SELECT and add WHERE image_path IS NOT NULL.
- Return ONLY the SQL query, nothing else. No markdown, no explanation."""

_ANSWER_PROMPT = """\
You answer questions about object-detection alerts. User is in India (IST).
Convert any UTC timestamps to IST (UTC+5:30) when displaying.
Current time: {now_ist}

Question: {question}
SQL result: {result}

Give a concise, helpful answer based on the data. If result is empty, say no matching data was found."""

MAX_RESULT_ROWS = 50


@dataclass
class QAService:
    db_path: str
    llm: Optional[BaseChatModel] = None
    _schema: str = field(default=_SCHEMA, repr=False)

    def answer_question(self, question: str) -> AnswerResult:
        clean_q = question.strip()
        if not clean_q:
            return AnswerResult("Please provide a question.")

        if self.llm is None:
            return AnswerResult(
                "LLM is not configured. Set LLM_MODEL and your provider's "
                "API key in .env to enable Q&A."
            )

        now_utc = datetime.now(tz=IST).astimezone(ZoneInfo("UTC"))
        now_ist = datetime.now(IST)
        today_ist = now_ist.date()
        today_utc_start = datetime(
            today_ist.year, today_ist.month, today_ist.day, tzinfo=IST
        ).astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S")
        from datetime import timedelta
        tomorrow_utc_start = (
            datetime(today_ist.year, today_ist.month, today_ist.day, tzinfo=IST)
            + timedelta(days=1)
        ).astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            sql = self._generate_sql(
                clean_q, now_utc.strftime("%Y-%m-%dT%H:%M:%S"),
                now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                today_utc_start, tomorrow_utc_start,
            )
            result_text, image_path = self._execute_sql(sql)
            answer = self._format_answer(
                clean_q, result_text, now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
            )
            return AnswerResult(text=answer, image_path=image_path)
        except Exception:
            logger.exception("QA failed for: %s", clean_q)
            return AnswerResult("Something went wrong while processing your question. Please try again.")

    def _generate_sql(
        self, question: str, now_utc: str, now_ist: str,
        today_utc_start: str, tomorrow_utc_start: str,
    ) -> str:
        prompt = _SQL_PROMPT.format(
            schema=self._schema, now_utc=now_utc, now_ist=now_ist,
            today_utc_start=today_utc_start, tomorrow_utc_start=tomorrow_utc_start,
        )
        resp = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        sql = resp.content.strip()
        sql = re.sub(r"^```(?:sql)?\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
        sql = sql.strip().rstrip(";") + ";"
        logger.info("Generated SQL: %s", sql)

        upper = sql.upper()
        if any(kw in upper for kw in ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE")):
            raise ValueError(f"Unsafe SQL blocked: {sql}")
        return sql

    def _execute_sql(self, sql: str) -> Tuple[str, Optional[str]]:
        """Execute SQL and return (result_text, first_image_path_or_None)."""
        conn = sqlite3.connect(self.db_path, timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchmany(MAX_RESULT_ROWS)
            if not rows:
                return "(no rows)", None
            cols = rows[0].keys()
            image_path = None
            if "image_path" in cols and rows[0]["image_path"]:
                image_path = str(rows[0]["image_path"])
            lines = [" | ".join(cols)]
            for r in rows:
                lines.append(" | ".join(str(r[c]) for c in cols))
            return "\n".join(lines), image_path
        finally:
            conn.close()

    def _format_answer(self, question: str, result: str, now_ist: str) -> str:
        prompt = _ANSWER_PROMPT.format(
            question=question, result=result, now_ist=now_ist,
        )
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
