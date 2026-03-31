from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional, Tuple
from zoneinfo import ZoneInfo

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)
qa_trace = logging.getLogger("qa.trace")
qa_trace.setLevel(logging.DEBUG)
_log_dir = os.getenv("SAVE_DIR", "/workspace/work/alerts")
os.makedirs(_log_dir, exist_ok=True)
_fh = logging.FileHandler(os.path.join(_log_dir, "qa_trace.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
qa_trace.addHandler(_fh)


@dataclass(frozen=True)
class AnswerResult:
    """Structured result from QAService.answer_question."""
    text: str
    image_path: Optional[str] = None

    def __str__(self) -> str:
        return self.text


IST = ZoneInfo("Asia/Kolkata")

_IMAGE_PATH_RE = re.compile(r"(/\S+?\.(?:jpg|jpeg|png))", re.IGNORECASE)
_UTC_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?")

MAX_AGENT_ITERATIONS = 6

_SYSTEM_PROMPT = """\
You are an agent that answers questions about a home security camera's
object-detection alerts by querying a SQLite database.

You have access to tools for listing tables, reading schemas, and running
SQL queries. Use them as needed.

## Timezone rules
- The `ts` column stores UTC timestamps as plain text (YYYY-MM-DDTHH:MM:SS).
- Current time: {now_utc} UTC = {now_ist} IST.
- The user is in India (IST = UTC+5:30). ALL user time references are in IST.
- You MUST convert IST times to UTC before filtering the `ts` column.
  Formula: UTC = IST − 5:30.
  Examples: midnight IST = previous day 18:30 UTC; 3 AM IST = previous day 21:30 UTC.
- Pre-computed boundaries for "today" (IST): ts >= '{today_utc_start}' AND ts < '{tomorrow_utc_start}'
- Pre-computed boundaries for "yesterday" (IST): ts >= '{yesterday_utc_start}' AND ts < '{today_utc_start}'
- Use plain string comparison with ts (no datetime()/strftime() on the ts column).
- If the user does NOT mention a specific time range, default to TODAY's range.

## Detection & class rules
- Each row in `alerts` is one alert. `count` = number of objects in that alert.
- "How many detections" / "total detections" → use SUM(count).
  "How many alerts" → use COUNT(*).  Default to SUM(count) when ambiguous.
- "Average/mean detections per day" → SUM(count) / number of distinct days, NOT AVG(count).
- Known class names: {class_names}. Only filter by class when the user names one.
- Classes are JSON arrays. Use LIKE '%"classname"%' to filter.
  Always search BOTH trigger_classes AND context_classes.
- For images/photos/screenshots: include image_path, add WHERE image_path IS NOT NULL.

## Analytics views (use these for summary/analytics questions)
- `v_daily_stats`: ist_date, total_detections, alert_count, avg_per_alert
- `v_hourly_stats`: ist_hour (0-23 in IST), total_detections, alert_count
These views already handle UTC→IST conversion — query them directly.

## Answer rules
- After getting query results, give a SHORT conversational answer (1-2 sentences).
- Use plain numbers: "5 dogs detected today", not tables or bullet lists.
- If the user asked for a specific screenshot, just say "Here's the screenshot".
- If no results, say "none found" or similar.
- Do NOT dump raw data or list every row.

## Few-shot examples

Question: "how many detections today"
SQL: SELECT SUM(count) FROM alerts WHERE ts >= '{today_utc_start}' AND ts < '{tomorrow_utc_start}';

Question: "any dogs?"
SQL: SELECT * FROM alerts WHERE (trigger_classes LIKE '%"dog"%' OR context_classes LIKE '%"dog"%') AND ts >= '{today_utc_start}' AND ts < '{tomorrow_utc_start}';

Question: "monthly summary with average per day and busiest hour"
SQL: SELECT SUM(total_detections) AS total, ROUND(1.0*SUM(total_detections)/COUNT(*),1) AS avg_per_day FROM v_daily_stats WHERE ist_date >= '2026-03-01';
Then: SELECT ist_hour, total_detections FROM v_hourly_stats ORDER BY total_detections DESC LIMIT 1;

Question: "show me the 2nd dog screenshot"
SQL: SELECT image_path FROM alerts WHERE (trigger_classes LIKE '%"dog"%' OR context_classes LIKE '%"dog"%') AND image_path IS NOT NULL AND ts >= '{today_utc_start}' AND ts < '{tomorrow_utc_start}' ORDER BY ts LIMIT 1 OFFSET 1;
"""


def _utc_results_to_ist(text: str) -> str:
    """Convert all UTC ISO timestamps in text to IST for the user."""
    offset = timedelta(hours=5, minutes=30)

    def _replace(m: re.Match) -> str:
        try:
            dt = datetime.fromisoformat(m.group())
            return (dt + offset).strftime("%Y-%m-%d %I:%M %p IST")
        except ValueError:
            return m.group()

    return _UTC_TS_RE.sub(_replace, text)


def _extract_image_path(text: str) -> Optional[str]:
    """Pull the first image path from agent output."""
    m = _IMAGE_PATH_RE.search(text)
    return m.group(1) if m else None


@dataclass
class QAService:
    db_path: str
    llm: Optional[BaseChatModel] = None
    class_names: Tuple[str, ...] = ()
    _agent: object = field(default=None, repr=False, init=False)

    def __post_init__(self):
        if self.llm is not None:
            self._agent = self._build_agent()

    def _build_agent(self):
        db = SQLDatabase.from_uri(
            f"sqlite:///{self.db_path}",
            sample_rows_in_table_info=3,
        )
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        tools = toolkit.get_tools()

        query_tool = next(t for t in tools if t.name == "sql_db_query")
        all_tools = [query_tool]
        tool_node = ToolNode(all_tools)

        model_with_tools = self.llm.bind_tools(all_tools)

        def agent_node(state: MessagesState):
            response = model_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "__end__"

        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue)
        builder.add_edge("tools", "agent")

        return builder.compile()

    def _build_system_prompt(self) -> str:
        now_utc = datetime.now(tz=IST).astimezone(ZoneInfo("UTC"))
        now_ist = datetime.now(IST)
        today_ist = now_ist.date()
        _utc_fmt = "%Y-%m-%dT%H:%M:%S"

        def _ist_day_to_utc(d) -> str:
            return datetime(d.year, d.month, d.day, tzinfo=IST).astimezone(
                ZoneInfo("UTC")
            ).strftime(_utc_fmt)

        return _SYSTEM_PROMPT.format(
            now_utc=now_utc.strftime(_utc_fmt),
            now_ist=now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
            today_utc_start=_ist_day_to_utc(today_ist),
            tomorrow_utc_start=_ist_day_to_utc(today_ist + timedelta(days=1)),
            yesterday_utc_start=_ist_day_to_utc(today_ist - timedelta(days=1)),
            class_names=", ".join(self.class_names) if self.class_names else "unknown",
        )

    def answer_question(self, question: str) -> AnswerResult:
        clean_q = question.strip()
        if not clean_q:
            return AnswerResult("Please provide a question.")

        if self._agent is None:
            return AnswerResult(
                "LLM is not configured. Set LLM_MODEL and your provider's "
                "API key in .env to enable Q&A."
            )

        try:
            system_prompt = self._build_system_prompt()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=clean_q),
            ]

            result = self._agent.invoke(
                {"messages": messages},
                {"recursion_limit": MAX_AGENT_ITERATIONS},
            )

            final_messages = result["messages"]
            answer_text = ""
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    answer_text = msg.content.strip()
                    break

            if not answer_text:
                answer_text = "I couldn't find an answer. Please try rephrasing."

            answer_text = _utc_results_to_ist(answer_text)
            image_path = _extract_image_path(
                "\n".join(m.content for m in final_messages if hasattr(m, "content") and m.content)
            )

            qa_trace.debug(
                "Q: %s | System: [%d chars] | Messages: %d | Answer: %s | Image: %s",
                clean_q, len(system_prompt), len(final_messages),
                answer_text, image_path,
            )

            return AnswerResult(text=answer_text, image_path=image_path)

        except Exception:
            logger.exception("QA failed for: %s", clean_q)
            return AnswerResult(
                "Something went wrong while processing your question. Please try again."
            )
