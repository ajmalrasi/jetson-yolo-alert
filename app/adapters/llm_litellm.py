from __future__ import annotations

from langchain_community.chat_models import ChatLiteLLM


def build_chat_llm(model: str, temperature: float = 0) -> ChatLiteLLM:
    """Build a ChatLiteLLM instance that routes to any provider based on the
    model string prefix (e.g. ``openai/gpt-4o-mini``, ``ollama/llama3``,
    ``groq/llama3-8b-8192``).  API keys are read from standard env vars
    (OPENAI_API_KEY, GROQ_API_KEY, etc.)."""
    return ChatLiteLLM(model=model, temperature=temperature)
