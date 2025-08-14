import os, requests
from typing import Optional
class TelegramSink:
    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        self.token = token
        self.chat = chat_id

    def send(self, text: str, image_path: Optional[str] = None) -> None:
        if not (self.token and self.chat): return
        try:
            if image_path and os.path.exists(image_path):
                requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendPhoto",
                    data={"chat_id": self.chat, "caption": text},
                    files={"photo": open(image_path, "rb")}, timeout=10
                )
            else:
                requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={"chat_id": self.chat, "text": text}, timeout=10
                )
        except requests.RequestException:
            pass
