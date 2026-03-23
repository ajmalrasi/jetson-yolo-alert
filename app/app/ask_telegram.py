from __future__ import annotations

import logging

from ..adapters.chat_telegram_bot import build_telegram_app
from ..core.config import Config
from ..core.qa_factory import build_qa_service

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = Config()
    if not cfg.tg_token:
        raise SystemExit("TELEGRAM_TOKEN / TG_BOT is required for the Telegram Q&A bot.")

    qa_service = build_qa_service(cfg)
    allowed = cfg.tg_qa_allowed_chat_id or cfg.tg_chat

    logger.info("Starting Telegram Q&A bot (model=%s)", cfg.llm_model)
    app = build_telegram_app(
        token=cfg.tg_token,
        qa_service=qa_service,
        allowed_chat_id=allowed,
    )
    app.run_polling()


if __name__ == "__main__":
    main()
