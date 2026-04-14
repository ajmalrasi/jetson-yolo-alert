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


def _build_video_service(cfg: Config):
    """Build VideoUnderstandingService if VLM_MODEL is configured."""
    vlm = cfg.vlm_model.strip()
    if not vlm or vlm == "none":
        logger.info("VLM_MODEL not set; /describe will be disabled")
        return None

    try:
        from ..core.frame_store import FrameStore
        from ..core.video_understanding import VideoUnderstandingService

        frame_store = FrameStore(cfg.frames_dir)
        all_classes = sorted(cfg.trigger_classes | cfg.draw_classes)

        service = VideoUnderstandingService(
            frame_store=frame_store,
            vlm_model=vlm,
            llm_model=cfg.llm_model,
            class_names=tuple(all_classes),
            vlm_max_frames=cfg.vlm_max_frames,
            vlm_max_width=cfg.vlm_max_width,
        )
        logger.info("Video understanding enabled (vlm=%s, max_frames=%d)", vlm, cfg.vlm_max_frames)
        return service
    except Exception:
        logger.exception("Failed to build VideoUnderstandingService")
        return None


def main() -> None:
    cfg = Config()
    if not cfg.tg_token:
        raise SystemExit("TELEGRAM_TOKEN / TG_BOT is required for the Telegram bot.")

    qa_service = build_qa_service(cfg)
    video_service = _build_video_service(cfg)
    allowed = cfg.tg_qa_allowed_chat_id or cfg.tg_chat

    logger.info(
        "Starting Telegram bot (llm=%s, vlm=%s)",
        cfg.llm_model, cfg.vlm_model,
    )
    app = build_telegram_app(
        token=cfg.tg_token,
        qa_service=qa_service,
        video_service=video_service,
        allowed_chat_id=allowed,
    )
    app.run_polling()


if __name__ == "__main__":
    main()
