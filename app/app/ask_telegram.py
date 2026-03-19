from __future__ import annotations

from ..adapters.chat_telegram_polling import TelegramPollingChatAdapter
from ..core.chat_commands import QAChatCommandHandler
from ..core.config import Config
from ..core.qa_factory import build_qa_service


def main() -> None:
    cfg = Config()
    if not cfg.tg_token:
        raise SystemExit("TELEGRAM_TOKEN/TG_BOT is required for Telegram Q&A bot.")

    allowed_chat_id = cfg.tg_qa_allowed_chat_id or cfg.tg_chat
    poll_timeout_sec = cfg.tg_qa_poll_timeout_sec
    idle_sleep_sec = cfg.tg_qa_idle_sleep_sec

    try:
        qa_service = build_qa_service(cfg)
    except PermissionError:
        cfg.alert_db_path = "/tmp/alert_history.db"
        qa_service = build_qa_service(cfg)
    handler = QAChatCommandHandler(qa_service=qa_service)

    bot = TelegramPollingChatAdapter(
        token=cfg.tg_token,
        text_handler=handler.handle_text,
        allowed_chat_id=allowed_chat_id,
        poll_timeout_sec=poll_timeout_sec,
        idle_sleep_sec=idle_sleep_sec,
    )
    bot.run_forever()


if __name__ == "__main__":
    main()
