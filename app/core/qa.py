from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

SYSTEM_PREFIX = """\
You answer questions about an object-detection alert history database.

Important context:
- All timestamps (ts column) are stored as ISO-8601 **UTC** strings.
- The user is in India (IST = UTC+5:30).  Always convert and display
  times in IST when responding.
- trigger_classes and context_classes are JSON arrays stored as TEXT,
  e.g. '["person","dog"]'.  Use LIKE '%"car"%' or json_each() to filter.
- Current time: {now_ist}
"""


@dataclass
class QAService:
    db: SQLDatabase
    llm: Optional[BaseChatModel] = None

    def answer_question(self, question: str) -> str:
        clean_q = question.strip()
        if not clean_q:
            return "Please provide a question."

        if self.llm is None:
            return (
                "LLM is not configured.  Set LLM_MODEL and your provider's "
                "API key in .env to enable Q&A."
            )

        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        prefix = SYSTEM_PREFIX.format(now_ist=now_ist)

        try:
            agent = create_sql_agent(
                self.llm,
                db=self.db,
                agent_type="tool-calling",
                prefix=prefix,
                verbose=False,
            )
            result: dict[str, Any] = agent.invoke({"input": clean_q})
            return result.get("output", "No answer was produced.")
        except Exception:
            logger.exception("SQL agent failed for question: %s", clean_q)
            return "Something went wrong while processing your question. Please try again."
