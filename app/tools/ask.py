from __future__ import annotations

import argparse
import sys

from ..adapters.llm_openai import OpenAILLMClient
from ..core.alert_history import AlertHistoryStore
from ..core.config import Config
from ..core.qa import QAService


def _build_service(cfg: Config) -> QAService:
    try:
        history = AlertHistoryStore(cfg.alert_db_path)
    except PermissionError:
        fallback = "/tmp/alert_history.db"
        history = AlertHistoryStore(fallback)
    llm = None
    if cfg.llm_provider == "openai" and cfg.openai_api_key:
        llm = OpenAILLMClient(
            api_key=cfg.openai_api_key,
            model=cfg.llm_model,
            base_url=cfg.openai_base_url,
        )
    return QAService(history=history, llm=llm)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a natural-language question over alert history.")
    parser.add_argument("question", nargs="*", help="Question text. If empty, reads from stdin.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = sys.stdin.read().strip()

    service = _build_service(Config())
    answer = service.answer_question(question)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
