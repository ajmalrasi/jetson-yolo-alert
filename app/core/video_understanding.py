from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import cv2

from ..adapters.vlm_litellm import describe_frames, encode_frame_b64
from .frame_store import FrameRecord, FrameStore

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

_VLM_SYSTEM_PROMPT = """\
You are a security camera analyst. You are given a sequence of frames \
captured from a surveillance camera at the indicated timestamps.

Describe what is happening chronologically. For each distinct event or \
change in the scene, note the approximate time range and describe the \
activity. Be concise but thorough.

Guidelines:
- Group consecutive similar frames into time ranges (e.g., "7:15-7:18 PM").
- Focus on people, animals, vehicles, and any unusual activity.
- If YOLO detection metadata is provided with frames, use it as context \
  but describe what you actually see in the images.
- If nothing notable is happening in a period, briefly note it \
  (e.g., "No activity detected").
- Use IST times in your response.
- Keep the response concise -- aim for 3-8 lines for a typical query.
"""

_TIME_PARSE_SYSTEM = """\
You are a time-range parser for a security camera system. Convert the \
user's natural language time reference into a UTC time range.

Current time: {now_utc} UTC = {now_ist} IST.
The user is in India (IST = UTC+5:30).

IMPORTANT: Output ONLY a JSON object with "start" and "end" keys, \
both as UTC timestamps in format "YYYY-MM-DDTHH:MM:SS". No explanation.

Examples:
- "last night" -> overnight period, roughly 9 PM to 6 AM IST (convert to UTC)
- "this morning" -> 6 AM to 12 PM IST today (convert to UTC)
- "yesterday" -> full day yesterday in IST (convert to UTC)
- "last 30 minutes" -> 30 minutes ago to now in UTC
- "last wednesday" -> full day last Wednesday in IST (convert to UTC)
- "between 3pm and 5pm" -> today 3 PM to 5 PM IST (convert to UTC)
"""


@dataclass
class VideoUnderstandingService:
    frame_store: FrameStore
    vlm_model: str
    llm_model: str
    class_names: Tuple[str, ...] = ()
    vlm_max_frames: int = 15
    vlm_max_width: int = 512

    def describe_timerange(self, question: str) -> str:
        """Parse a natural language time query, load frames, and describe via VLM."""
        if not self.vlm_model or self.vlm_model == "none":
            return "VLM is not configured. Set VLM_MODEL in .env to enable /describe."

        start_utc, end_utc = self._parse_time_range(question)
        if start_utc is None or end_utc is None:
            return "I couldn't understand the time range. Try something like 'last night' or 'yesterday afternoon'."

        records = self.frame_store.query_range(start_utc, end_utc)
        if not records:
            return self._no_frames_message(start_utc, end_utc)

        sampled = self._smart_sample(records)
        frames_for_vlm = self._load_and_encode(sampled)
        if not frames_for_vlm:
            return "Could not load any frames from that time range."

        prompt = _VLM_SYSTEM_PROMPT
        try:
            narrative = describe_frames(self.vlm_model, frames_for_vlm, prompt)
        except Exception:
            logger.exception("VLM call failed")
            return "Something went wrong while analyzing the frames. Please try again."

        count = len(records)
        header = f"Analyzed {count} frames from {self._utc_to_ist_label(start_utc)} to {self._utc_to_ist_label(end_utc)}:\n\n"
        return header + narrative

    def describe_recent(self, minutes: int = 5) -> str:
        """Describe the last N minutes of captured frames."""
        now = datetime.now(UTC)
        start = now - timedelta(minutes=minutes)
        start_utc = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%S")

        records = self.frame_store.query_range(start_utc, end_utc)
        if not records:
            return f"No frames captured in the last {minutes} minutes. Nothing was detected."

        sampled = self._smart_sample(records)
        frames_for_vlm = self._load_and_encode(sampled)
        if not frames_for_vlm:
            return "Could not load frames."

        try:
            narrative = describe_frames(self.vlm_model, frames_for_vlm, _VLM_SYSTEM_PROMPT)
        except Exception:
            logger.exception("VLM call failed")
            return "Something went wrong while analyzing the frames. Please try again."

        return f"Last {minutes} minutes ({len(records)} frames):\n\n{narrative}"

    def describe_video(self, video_path: str) -> str:
        """Describe an uploaded video file."""
        if not self.vlm_model or self.vlm_model == "none":
            return "VLM is not configured. Set VLM_MODEL in .env to enable video description."

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Could not open the video file."

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        interval_sec = max(duration_sec / self.vlm_max_frames, 1.0)

        frames_for_vlm: List[Tuple[str, str, str]] = []
        frame_idx = 0
        next_capture_sec = 0.0

        while True:
            ok, img = cap.read()
            if not ok:
                break
            current_sec = frame_idx / fps if fps > 0 else 0
            if current_sec >= next_capture_sec:
                ts_label = _format_video_timestamp(current_sec)
                b64 = encode_frame_b64(img, max_width=self.vlm_max_width)
                frames_for_vlm.append((ts_label, b64, ""))
                next_capture_sec += interval_sec
                if len(frames_for_vlm) >= self.vlm_max_frames:
                    break
            frame_idx += 1

        cap.release()

        if not frames_for_vlm:
            return "Could not extract any frames from the video."

        dur_label = _format_video_timestamp(duration_sec)
        try:
            narrative = describe_frames(self.vlm_model, frames_for_vlm, _VLM_SYSTEM_PROMPT)
        except Exception:
            logger.exception("VLM call failed for video %s", video_path)
            return "Something went wrong while analyzing the video. Please try again."

        return f"Video analysis ({dur_label}, {len(frames_for_vlm)} frames sampled):\n\n{narrative}"

    # ------------------------------------------------------------------
    # Time-range parsing
    # ------------------------------------------------------------------

    def _parse_time_range(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """Use the text LLM to parse a natural language time reference into UTC range."""
        quick = self._try_quick_parse(question)
        if quick:
            return quick

        if not self.llm_model or self.llm_model == "none":
            return self._fallback_time_range()

        now_utc = datetime.now(UTC)
        now_ist = now_utc.astimezone(IST)
        system_prompt = _TIME_PARSE_SYSTEM.format(
            now_utc=now_utc.strftime("%Y-%m-%dT%H:%M:%S"),
            now_ist=now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        )

        try:
            import litellm
            response = litellm.completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*$", "", text)
            parsed = json.loads(text)
            return parsed.get("start"), parsed.get("end")
        except Exception:
            logger.exception("Time-range parsing failed for: %s", question)
            return self._fallback_time_range()

    def _try_quick_parse(self, question: str) -> Optional[Tuple[str, str]]:
        """Handle common patterns without an LLM call."""
        q = question.lower().strip()
        now = datetime.now(IST)

        match = re.search(r"last\s+(\d+)\s+(minute|min|hour|hr)s?", q)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit in ("hour", "hr"):
                delta = timedelta(hours=amount)
            else:
                delta = timedelta(minutes=amount)
            start = (now - delta).astimezone(UTC)
            end = now.astimezone(UTC)
            return start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S")

        return None

    def _fallback_time_range(self) -> Tuple[str, str]:
        """Default to last 1 hour if LLM is unavailable."""
        now = datetime.now(UTC)
        start = now - timedelta(hours=1)
        return start.strftime("%Y-%m-%dT%H:%M:%S"), now.strftime("%Y-%m-%dT%H:%M:%S")

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def _smart_sample(self, records: List[FrameRecord]) -> List[FrameRecord]:
        """Pick up to vlm_max_frames from records, prioritizing interesting moments."""
        if len(records) <= self.vlm_max_frames:
            return records

        det_records = [r for r in records if r.has_detection]
        no_det_records = [r for r in records if not r.has_detection]

        if len(det_records) <= self.vlm_max_frames:
            remaining = self.vlm_max_frames - len(det_records)
            sampled = det_records + _evenly_sample(no_det_records, remaining)
        else:
            sampled = _evenly_sample(det_records, self.vlm_max_frames)

        sampled.sort(key=lambda r: r.ts)
        return sampled

    def _load_and_encode(
        self, records: List[FrameRecord]
    ) -> List[Tuple[str, str, str]]:
        """Load frames from disk and encode for VLM."""
        result = []
        for rec in records:
            if not os.path.isfile(rec.path):
                continue
            img = cv2.imread(rec.path)
            if img is None:
                continue
            b64 = encode_frame_b64(img, max_width=self.vlm_max_width)
            ts_label = self._utc_to_ist_label(rec.ts)
            yolo_ctx = self._format_yolo_context(rec)
            result.append((ts_label, b64, yolo_ctx))
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_to_ist_label(ts_str: str) -> str:
        try:
            dt = datetime.fromisoformat(ts_str).replace(tzinfo=UTC)
            ist_dt = dt.astimezone(IST)
            return ist_dt.strftime("%I:%M %p IST").lstrip("0")
        except (ValueError, TypeError):
            return ts_str

    @staticmethod
    def _format_yolo_context(rec: FrameRecord) -> str:
        if not rec.has_detection or not rec.detection_classes:
            return ""
        classes = ", ".join(rec.detection_classes)
        return f"YOLO detected: {rec.detection_count} object(s) [{classes}] (best conf: {rec.best_conf:.2f})"

    def _no_frames_message(self, start_utc: str, end_utc: str) -> str:
        start_label = self._utc_to_ist_label(start_utc)
        end_label = self._utc_to_ist_label(end_utc)
        return (
            f"No frames captured between {start_label} and {end_label}. "
            "This means YOLO detected no activity in that period."
        )


def _evenly_sample(items: list, n: int) -> list:
    """Pick n items evenly spaced from a list."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _format_video_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
