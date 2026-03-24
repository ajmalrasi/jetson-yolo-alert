from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

_PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
}

_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_chat_llm(model: str, temperature: float = 0) -> BaseChatModel:
    """Build a chat LLM from a litellm-style model string like
    ``openai/gpt-4o-mini``, ``groq/llama-3.1-8b-instant``, or
    ``xai/grok-2-latest``.

    Uses ``ChatOpenAI`` under the hood, pointed at each provider's
    OpenAI-compatible endpoint.  API keys are read from standard env vars.
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
    else:
        provider, model_name = "openai", model

    provider = provider.lower()
    base_url = _PROVIDER_BASE_URLS.get(provider)
    key_env = _PROVIDER_KEY_ENV.get(provider, "OPENAI_API_KEY")
    api_key = os.getenv(key_env, "")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
