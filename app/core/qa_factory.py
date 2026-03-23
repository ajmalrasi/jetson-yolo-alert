from __future__ import annotations

from ..adapters.llm_openai import OpenAILLMClient
from .alert_history import AlertHistoryStore
from .config import Config
from .qa import QAService


def build_qa_service(cfg: Config) -> QAService:
    history = AlertHistoryStore(cfg.alert_db_path)
    llm = None
    if cfg.llm_provider == "openai" and cfg.openai_api_key:
        llm = OpenAILLMClient(
            api_key=cfg.openai_api_key,
            model=cfg.llm_model,
            base_url=cfg.openai_base_url,
        )
    known_classes = cfg.trigger_classes | cfg.draw_classes
    return QAService(history=history, llm=llm, known_classes=known_classes)
