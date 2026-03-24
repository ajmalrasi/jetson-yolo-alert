from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .qa import QAService


@dataclass
class QAChatCommandHandler:
    qa_service: QAService

    def handle_text(self, text: str) -> Optional[str]:
        message = (text or "").strip()
        if not message:
            return None

        lower = message.lower()
        if lower in ("/start", "start", "/help", "help"):
            return (
                "Ask me alert-history questions.\n"
                "Examples:\n"
                "- /ask When was the last alert?\n"
                "- /ask How many people came on 2026-03-19?"
            )

        if lower.startswith("/ask"):
            question = message[4:].strip()
            if not question:
                return "Please add a question after /ask."
            return str(self.qa_service.answer_question(question))

        if lower.startswith("ask "):
            question = message[4:].strip()
            return str(self.qa_service.answer_question(question)) if question else "Please add a question after ask."

        # Keep this strict so generic chatter does not trigger accidental queries.
        return "Use /ask <your question>."
