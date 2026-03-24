from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from telegram import Update
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
    "Ask me alert-history questions.\n"
    "Examples:\n"
    "- /ask When was the last alert?\n"
    "- /ask How many people came on 2026-03-19?"
)


def build_telegram_app(
    token: str,
    qa_service: QAService,
    allowed_chat_id: Optional[str] = None,
):
    """Build an async ``python-telegram-bot`` Application wired to *qa_service*.

    If *allowed_chat_id* is set, only messages from that chat are processed.
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

    async def _ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        question = " ".join(context.args) if context.args else ""
        if not question:
            await update.message.reply_text("Please add a question after /ask.")
            return

        result = await asyncio.to_thread(qa_service.answer_question, question)
        await _reply_answer(update, result)

    async def _help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(HELP_TEXT)

    async def _plain_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = (update.message.text or "").strip()
        if not text:
            return
        lower = text.lower()
        if lower.startswith("ask "):
            question = text[4:].strip()
            if question:
                result = await asyncio.to_thread(qa_service.answer_question, question)
                await _reply_answer(update, result)
                return
        await update.message.reply_text("Use /ask <your question>.")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("ask", _ask_handler, filters=chat_filter))
    app.add_handler(CommandHandler(["help", "start"], _help_handler, filters=chat_filter))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & chat_filter, _plain_text_handler))
    return app
