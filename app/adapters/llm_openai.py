from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import requests


@dataclass
class OpenAILLMClient:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 20.0

    def complete(self, system_prompt: str, user_message: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.2,
            },
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        content: Optional[str] = (
            data.get("choices", [{}])[0].get("message", {}).get("content")
            if isinstance(data, dict)
            else None
        )
        return (content or "").strip()
