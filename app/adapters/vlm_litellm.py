from __future__ import annotations

import base64
import logging
import os
from typing import List, Tuple

import litellm

logger = logging.getLogger(__name__)

_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}


def _resolve_api_key(provider: str) -> str | None:
    key_env = _PROVIDER_KEY_ENV.get(provider, "OPENAI_API_KEY")
    return os.getenv(key_env) or None


def describe_frames(
    model: str,
    frames: List[Tuple[str, str, str]],
    system_prompt: str,
) -> str:
    """Call a vision-language model to describe a sequence of frames.

    Args:
        model: litellm model string, e.g. "openai/gpt-4o-mini", "groq/llama-4-scout-17b-16e-instruct".
        frames: list of (timestamp_label, base64_jpeg, yolo_context) tuples.
            - timestamp_label: e.g. "7:15 PM IST"
            - base64_jpeg: base64-encoded JPEG image data
            - yolo_context: e.g. "YOLO detected: 1 person (0.87), 1 dog (0.72)" or ""
        system_prompt: system message for the VLM.

    Returns:
        The VLM's text response describing the video content.
    """
    content_blocks: list[dict] = []
    for ts_label, b64_img, yolo_ctx in frames:
        caption = f"Frame at {ts_label}"
        if yolo_ctx:
            caption += f" -- {yolo_ctx}"
        content_blocks.append({"type": "text", "text": caption})
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_blocks},
    ]

    provider = model.split("/", 1)[0].lower() if "/" in model else "openai"
    api_key = _resolve_api_key(provider)

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=0.3,
        max_tokens=2048,
    )

    return response.choices[0].message.content.strip()


def encode_frame_b64(image, max_width: int = 512, jpeg_quality: int = 80) -> str:
    """Resize and encode a BGR numpy image to base64 JPEG."""
    import cv2

    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (max_width, int(h * scale)))

    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")
