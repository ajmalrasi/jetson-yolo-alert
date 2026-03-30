from __future__ import annotations

import logging

from ..adapters.llm_litellm import build_chat_llm
from .alert_history import AlertHistoryStore
from .config import Config
from .qa import QAService

logger = logging.getLogger(__name__)


def _ensure_db_exists(db_path: str) -> str:
    """Make sure the SQLite file and its parent directory exist.
    Falls back to /tmp if the configured path is not writable."""
    try:
        AlertHistoryStore(db_path)
        return db_path
    except PermissionError:
        fallback = "/tmp/alert_history.db"
        logger.warning("Cannot write to %s, falling back to %s", db_path, fallback)
        AlertHistoryStore(fallback)
        return fallback


def build_qa_service(cfg: Config) -> QAService:
    db_path = _ensure_db_exists(cfg.alert_db_path)

    llm = None
    model = cfg.llm_model.strip()
    if model and model != "none":
        llm = build_chat_llm(model=model)

    all_classes = sorted(cfg.trigger_classes | cfg.draw_classes)
    return QAService(db_path=db_path, llm=llm, class_names=tuple(all_classes))
