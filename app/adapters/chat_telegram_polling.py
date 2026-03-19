from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

import requests


TextHandler = Callable[[str], Optional[str]]


@dataclass
class TelegramPollingChatAdapter:
    token: str
    text_handler: TextHandler
    allowed_chat_id: Optional[str] = None
    poll_timeout_sec: int = 25
    idle_sleep_sec: float = 1.0

    def run_forever(self) -> None:
        offset: Optional[int] = None
        while True:
            try:
                updates = self._get_updates(offset=offset)
            except requests.RequestException:
                time.sleep(self.idle_sleep_sec)
                continue

            for update in updates:
                update_id = int(update.get("update_id", 0))
                offset = update_id + 1

                msg = update.get("message") or update.get("edited_message") or {}
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                text = msg.get("text")
                if not text or not chat_id:
                    continue
                if self.allowed_chat_id and chat_id != self.allowed_chat_id:
                    continue

                try:
                    reply = self.text_handler(text)
                    if reply:
                        self._send_message(chat_id=chat_id, text=reply)
                except Exception:
                    continue

            if not updates:
                time.sleep(self.idle_sleep_sec)

    def _get_updates(self, offset: Optional[int]) -> list[dict]:
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        payload = {"timeout": self.poll_timeout_sec}
        if offset is not None:
            payload["offset"] = offset
        resp = requests.get(url, params=payload, timeout=self.poll_timeout_sec + 5)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("ok"):
            return []
        result = data.get("result")
        return result if isinstance(result, list) else []

    def _send_message(self, chat_id: str, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
