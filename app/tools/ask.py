from __future__ import annotations

import argparse
import sys

from ..core.config import Config
from ..core.qa_factory import build_qa_service


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a natural-language question over alert history.")
    parser.add_argument("question", nargs="*", help="Question text. If empty, reads from stdin.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = sys.stdin.read().strip()

    cfg = Config()
    try:
        service = build_qa_service(cfg)
    except PermissionError:
        cfg.alert_db_path = "/tmp/alert_history.db"
        service = build_qa_service(cfg)
    answer = service.answer_question(question)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
