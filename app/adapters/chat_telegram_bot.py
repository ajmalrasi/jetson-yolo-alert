from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..core.qa import AnswerResult, QAService

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Commands:\n"
    "/ask <question> -- Ask about alert history (SQL-based)\n"
    "/describe <time reference> -- Describe what happened on camera\n"
    "/describe -- Describe last 5 minutes\n"
    "Send a video -- Describe the video contents\n"
    "\n"
    "Examples:\n"
    "- /ask How many people today?\n"
    "- /describe what happened last night?\n"
    "- /describe last 30 minutes\n"
    "- /describe yesterday afternoon\n"
)


def build_telegram_app(
    token: str,
    qa_service: Optional[QAService] = None,
    video_service=None,
    allowed_chat_id: Optional[str] = None,
):
    """Build an async ``python-telegram-bot`` Application.

    Args:
        token: Telegram bot token.
        qa_service: Optional QAService for /ask (alert history queries).
        video_service: Optional VideoUnderstandingService for /describe.
        allowed_chat_id: Restrict to this chat ID if set.
    """

    chat_filter: filters.BaseFilter = filters.ALL
    if allowed_chat_id:
        try:
            chat_filter = filters.Chat(chat_id=int(allowed_chat_id))
        except (TypeError, ValueError):
            logger.warning("Invalid TG_QA_ALLOWED_CHAT_ID=%s, allowing all chats", allowed_chat_id)

    async def _reply_answer(update: Update, result: AnswerResult) -> None:
        if result.image_path and os.path.isfile(result.image_path):
            caption = result.text[:1024] if result.text else None
            try:
                with open(result.image_path, "rb") as f:
                    await update.message.reply_photo(photo=f, caption=caption)
                return
            except Exception:
                logger.warning("Failed to send photo %s, falling back to text", result.image_path)
        await update.message.reply_text(str(result))

    # ---- /ask handler (existing) ----

    async def _ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if qa_service is None:
            await update.message.reply_text(
                "Q&A is not configured. Set LLM_MODEL in .env to enable /ask."
            )
            return
        question = " ".join(context.args) if context.args else ""
        if not question:
            await update.message.reply_text("Please add a question after /ask.")
            return

        result = await asyncio.to_thread(qa_service.answer_question, question)
        await _reply_answer(update, result)

    # ---- /describe handler (new) ----

    async def _describe_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if video_service is None:
            await update.message.reply_text(
                "Video understanding is not configured. Set VLM_MODEL in .env to enable /describe."
            )
            return

        question = " ".join(context.args) if context.args else ""
        await update.message.reply_chat_action(ChatAction.TYPING)

        if question:
            result = await asyncio.to_thread(video_service.describe_timerange, question)
        else:
            result = await asyncio.to_thread(video_service.describe_recent, 5)

        for chunk in _split_message(result):
            await update.message.reply_text(chunk)

    # ---- Video upload handler (new) ----

    async def _video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if video_service is None:
            await update.message.reply_text(
                "Video understanding is not configured. Set VLM_MODEL in .env."
            )
            return

        video = update.message.video or update.message.video_note
        document = update.message.document
        file_obj = None

        if video:
            file_obj = await context.bot.get_file(video.file_id)
        elif document and document.mime_type and document.mime_type.startswith("video/"):
            file_obj = await context.bot.get_file(document.file_id)

        if file_obj is None:
            return

        await update.message.reply_chat_action(ChatAction.TYPING)
        await update.message.reply_text("Downloading and analyzing video...")

        tmp_dir = tempfile.mkdtemp(prefix="tg_video_")
        tmp_path = os.path.join(tmp_dir, "video.mp4")
        try:
            await file_obj.download_to_drive(tmp_path)
            result = await asyncio.to_thread(video_service.describe_video, tmp_path)
            for chunk in _split_message(result):
                await update.message.reply_text(chunk)
        except Exception:
            logger.exception("Video processing failed")
            await update.message.reply_text("Failed to process the video. Please try again.")
        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

    # ---- Help & plain text ----

    async def _help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(HELP_TEXT)

    async def _plain_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = (update.message.text or "").strip()
        if not text:
            return
        lower = text.lower()
        if lower.startswith("ask ") and qa_service is not None:
            question = text[4:].strip()
            if question:
                result = await asyncio.to_thread(qa_service.answer_question, question)
                await _reply_answer(update, result)
                return
        if lower.startswith("describe ") and video_service is not None:
            question = text[9:].strip()
            if question:
                await update.message.reply_chat_action(ChatAction.TYPING)
                result = await asyncio.to_thread(video_service.describe_timerange, question)
                for chunk in _split_message(result):
                    await update.message.reply_text(chunk)
                return
        await update.message.reply_text("Use /ask <question> or /describe <time reference>.")

    # ---- Build app ----

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("ask", _ask_handler, filters=chat_filter))
    app.add_handler(CommandHandler("describe", _describe_handler, filters=chat_filter))
    app.add_handler(CommandHandler(["help", "start"], _help_handler, filters=chat_filter))
    app.add_handler(MessageHandler(
        (filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO) & chat_filter,
        _video_handler,
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & chat_filter,
        _plain_text_handler,
    ))
    return app


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a long message into Telegram-safe chunks."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at < 0:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
