from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import calendar
import json
import re
from typing import List, Optional, Protocol, Set, Tuple
from zoneinfo import ZoneInfo

from .alert_history import AlertHistoryStore, AlertRecord

IST = ZoneInfo("Asia/Kolkata")


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_message: str) -> str: ...


@dataclass
class QAService:
    history: AlertHistoryStore
    llm: Optional[LLMClient] = None
    known_classes: Set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.known_classes is None:
            self.known_classes = set()

    def answer_question(self, question: str) -> str:
        clean_q = question.strip()
        if not clean_q:
            return "Please provide a question."

        q_lower = clean_q.lower()
        if "last alert" in q_lower:
            return self._answer_last_alert(clean_q)

        now_ist = datetime.now(IST)
        day = _extract_day(clean_q, now_local=now_ist)
        inferred_intent: Optional[str] = None
        inferred_classes: Optional[List[str]] = None
        if day is None and self.llm is not None:
            inferred_intent, inferred_date, inferred_classes = self._infer_query_with_llm(clean_q)
            if inferred_intent == "last_alert":
                return self._answer_last_alert(clean_q)
            if inferred_date:
                try:
                    day = datetime.strptime(inferred_date, "%Y-%m-%d").replace(tzinfo=IST)
                except ValueError:
                    day = None
        if day is None:
            return (
                "I could not find a specific date in your question. "
                "Try a date like '2026-03-10', 'March 10', or use 'last alert'."
            )

        class_filter = _extract_class_filter(clean_q, self.known_classes)
        if not class_filter and inferred_classes:
            class_filter = set(inferred_classes)

        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        rows = self.history.get_alerts_between(day_start, day_end)

        if class_filter:
            rows = _filter_rows_by_class(rows, class_filter)

        alert_events = len(rows)
        total_objects = sum(r.count for r in rows)
        best_conf = max((r.best_conf for r in rows), default=0.0)

        all_trigger = sorted({c for r in rows for c in r.trigger_classes})
        all_context = sorted({c for r in rows for c in r.context_classes})
        trigger_str = ", ".join(all_trigger) if all_trigger else "-"
        context_str = ", ".join(all_context) if all_context else "-"
        filter_str = ", ".join(sorted(class_filter)) if class_filter else None
        date_str = day.date().isoformat()

        if alert_events == 0:
            if filter_str:
                return f"No {filter_str} alerts were recorded on {date_str} (IST)."
            return f"No alerts were recorded on {date_str} (IST)."

        alert_lines = []
        for r in rows:
            t = r.ts.astimezone(IST).strftime("%H:%M:%S IST")
            classes = ", ".join(r.trigger_classes) if r.trigger_classes else "-"
            alert_lines.append(f"  {t} | {r.count} objects | classes: {classes}")
        timeline = "\n".join(alert_lines)

        summary = (
            f"date={date_str} IST\n"
            f"alerts={alert_events}\n"
            f"total_detected_objects={total_objects}\n"
            f"best_confidence={best_conf:.2f}\n"
            f"trigger_classes={trigger_str}\n"
            f"context_classes={context_str}\n"
            f"timeline:\n{timeline}"
        )
        if filter_str:
            summary += f"\nfiltered_by_class={filter_str}"

        if filter_str:
            default_answer = (
                f"On {date_str} (IST), there were {alert_events} alerts involving {filter_str}:\n"
                f"{timeline}"
            )
        else:
            default_answer = (
                f"On {date_str} (IST), there were {alert_events} alerts "
                f"with a total of {total_objects} detected objects:\n"
                f"{timeline}"
            )
        return self._format_with_llm(question=clean_q, summary=summary, default_answer=default_answer)

    def _answer_last_alert(self, question: str) -> str:
        last = self.history.get_last_alert()
        if last is None:
            return "No alerts have been recorded yet."

        ts = last.ts.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        trigger_classes = ", ".join(last.trigger_classes) if last.trigger_classes else "-"
        context_classes = ", ".join(last.context_classes) if last.context_classes else "-"
        summary = (
            f"last_alert_at={ts}\n"
            f"detected_objects={last.count}\n"
            f"best_confidence={last.best_conf:.2f}\n"
            f"trigger_classes={trigger_classes}\n"
            f"context_classes={context_classes}\n"
            f"image_path={last.image_path or '-'}"
        )
        default_answer = (
            f"The last alert was at {ts}, with {last.count} detected objects "
            f"(best confidence {last.best_conf:.2f}, trigger classes: {trigger_classes})."
        )
        return self._format_with_llm(question=question, summary=summary, default_answer=default_answer)

    def _format_with_llm(self, question: str, summary: str, default_answer: str) -> str:
        if self.llm is None:
            return default_answer

        system_prompt = (
            "You answer questions about alert history using only provided facts. "
            "Be concise, direct, and do not invent missing data."
        )
        user_message = (
            f"User question:\n{question}\n\n"
            f"Facts:\n{summary}\n\n"
            "Write a one-sentence answer."
        )
        try:
            answer = self.llm.complete(system_prompt=system_prompt, user_message=user_message)
        except Exception:
            return default_answer
        return answer or default_answer

    def _infer_query_with_llm(self, question: str) -> tuple[Optional[str], Optional[str], Optional[List[str]]]:
        if self.llm is None:
            return None, None, None

        now = datetime.now(IST).strftime("%Y-%m-%d")
        system_prompt = (
            "You convert user questions about alert history into strict JSON. "
            "Return only JSON with keys: intent, date, classes. "
            "intent must be one of: date_query, last_alert, count_alerts, unknown. "
            "date must be YYYY-MM-DD for date-based questions, otherwise null. "
            "classes must be a list of object class names the user is asking about "
            "(e.g. [\"car\"], [\"dog\",\"cat\"]), or [] if asking about all classes. "
            "Resolve relative terms like today, yesterday, this morning using India time (IST)."
        )
        user_message = (
            f"Current IST date: {now}\n"
            f"Question: {question}\n"
            "JSON:"
        )
        try:
            raw = self.llm.complete(system_prompt=system_prompt, user_message=user_message)
        except Exception:
            return None, None, None
        intent, date_value, classes = _parse_llm_query_json(raw)
        return intent, date_value, classes


def _extract_class_filter(question: str, known_classes: Set[str]) -> Set[str]:
    q = question.strip().lower()
    found = set()
    for cls in known_classes:
        if re.search(rf"\b{re.escape(cls)}s?\b", q):
            found.add(cls)
    return found


def _filter_rows_by_class(rows: List[AlertRecord], class_filter: Set[str]) -> List[AlertRecord]:
    filtered = []
    for r in rows:
        all_classes = set(r.trigger_classes) | set(r.context_classes)
        if class_filter & all_classes:
            filtered.append(r)
    return filtered


def _extract_day(question: str, now_local: datetime) -> Optional[datetime]:
    q = question.strip().lower()

    if "today" in q:
        return now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    if "yesterday" in q:
        day = now_local - timedelta(days=1)
        return day.replace(hour=0, minute=0, second=0, microsecond=0)

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
    if iso_match:
        try:
            day = datetime.strptime(iso_match.group(1), "%Y-%m-%d")
            return day.replace(tzinfo=now_local.tzinfo or IST)
        except ValueError:
            pass

    month_names = list(calendar.month_name)[1:]
    short_months = list(calendar.month_abbr)[1:]
    all_month_tokens = sorted(month_names + short_months, key=len, reverse=True)
    month_pattern = "|".join(re.escape(m.lower()) for m in all_month_tokens)
    long_match = re.search(rf"\b({month_pattern})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b", q)
    if not long_match:
        return None

    month_token = long_match.group(1).lower()
    day_token = int(long_match.group(2))
    year_token = int(long_match.group(3)) if long_match.group(3) else now_local.year

    month_num = _month_to_number(month_token)
    if month_num is None:
        return None
    try:
        return datetime(year_token, month_num, day_token, tzinfo=now_local.tzinfo or IST)
    except ValueError:
        return None


def _month_to_number(token: str) -> Optional[int]:
    for idx, name in enumerate(calendar.month_name):
        if idx == 0:
            continue
        if token == name.lower():
            return idx
    for idx, name in enumerate(calendar.month_abbr):
        if idx == 0:
            continue
        if token == name.lower():
            return idx
    return None


def _parse_llm_query_json(raw: str) -> tuple[Optional[str], Optional[str], Optional[List[str]]]:
    text = (raw or "").strip()
    if not text:
        return None, None, None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None

    intent = data.get("intent")
    if isinstance(intent, str):
        intent = intent.strip().lower()
    else:
        intent = None

    date_value = data.get("date")
    if isinstance(date_value, str):
        date_value = date_value.strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_value):
            date_value = None
    else:
        date_value = None

    classes: Optional[List[str]] = None
    raw_classes = data.get("classes")
    if isinstance(raw_classes, list):
        classes = [str(c).strip().lower() for c in raw_classes if isinstance(c, str) and c.strip()]
    return intent, date_value, classes or None
