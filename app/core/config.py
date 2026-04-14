import os
from dataclasses import dataclass, field
from typing import Optional, Set

def _csv_to_set_int(csv: str) -> Set[int]:
    out = set()
    for part in csv.split(","):
        part = part.strip()
        if part.isdigit(): out.add(int(part))
    return out

def _csv_to_set_str(csv: str) -> Set[str]:
    return {s.strip().lower() for s in csv.split(",") if s.strip()}

@dataclass
class Config:
    # IO
    src: str = os.getenv("SRC", "0")
    engine: str = os.getenv("YOLO_ENGINE", "yolov8n.engine")
    conf_thresh: float = float(os.getenv("CONF_THRESH", "0.80"))
    img_size: int = int(os.getenv("IMG_SIZE", "640"))
    vid_stride: int = int(os.getenv("VID_STRIDE", "6"))
    # FPS policy
    base_fps: float = float(os.getenv("BASE_FPS", "2"))
    high_fps: float = float(os.getenv("HIGH_FPS", "0"))  # 0 = uncapped
    boost_arm_frames: int = int(os.getenv("BOOST_ARM_FRAMES", "3"))
    boost_min_sec: float = float(os.getenv("BOOST_MIN_SEC", "2.0"))
    cooldown_sec: float = float(os.getenv("COOLDOWN_SEC", "5.0"))
    # Presence & alerts
    trigger_classes: Set[str] = field(
        default_factory=lambda: _csv_to_set_str(os.getenv("TRIGGER_CLASSES", "person"))
    )
    draw_classes: Set[str] = field(
        default_factory=lambda: _csv_to_set_str(os.getenv("DRAW_CLASSES", "person,car,dog,cat"))
    )
    min_frames: int = int(os.getenv("MIN_FRAMES", "3"))
    min_persist_sec: float = float(os.getenv("MIN_PERSIST_SEC", "1.0"))
    rearm_sec: float = float(os.getenv("REARM_SEC", "10"))
    rate_window_sec: float = float(os.getenv("RATE_WINDOW_SEC", "5"))
    # Min time between any two alerts (stops same person with new track_id from re-triggering)
    alert_cooldown_sec: float = float(os.getenv("ALERT_COOLDOWN_SEC", "0"))  # 0 = use rate_window_sec only
    # Tracker
    tracker_cfg: Optional[str] = os.getenv("TRACKER", "bytetrack.yaml")
    tracker_on: bool = os.getenv("TRACKER_ON", "1") not in ("0", "false", "False", "")
    # Telegram (TELEGRAM_* or TG_BOT/TG_CHAT from README)
    tg_token: Optional[str] = os.getenv("TELEGRAM_TOKEN") or os.getenv("TG_BOT")
    tg_chat: Optional[str] = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT")
    tg_qa_allowed_chat_id: Optional[str] = os.getenv("TG_QA_ALLOWED_CHAT_ID")
    # Misc
    save_dir: str = os.getenv("SAVE_DIR", "/workspace/work/alerts")
    draw: bool = os.getenv("DRAW", "1") not in ("0","false","False","")
    # Save clean (un-annotated) frames for fine-tuning / labeling
    save_raw_frames: bool = os.getenv("SAVE_RAW_FRAMES", "0") not in ("0", "false", "False", "")
    raw_frames_dir: str = os.getenv(
        "RAW_FRAMES_DIR",
        os.path.join(os.getenv("SAVE_DIR", "/workspace/work/alerts"), "raw_frames"),
    )
    # Alert history / LLM QA
    alert_db_path: str = os.getenv(
        "ALERT_DB_PATH",
        os.path.join(os.getenv("SAVE_DIR", "/workspace/work/alerts"), "alert_history.db"),
    )
    llm_model: str = os.getenv("LLM_MODEL", "none")
    # Video understanding (VLM)
    vlm_model: str = os.getenv("VLM_MODEL", "none")
    vlm_max_frames: int = int(os.getenv("VLM_MAX_FRAMES", "15"))
    vlm_max_width: int = int(os.getenv("VLM_MAX_WIDTH", "512"))
    # Frame capture (for VLM)
    frames_dir: str = os.getenv("FRAMES_DIR", "/workspace/work/frames")
    frames_retention_days: int = int(os.getenv("FRAMES_RETENTION_DAYS", "30"))
    capture_active_fps: float = float(os.getenv("CAPTURE_ACTIVE_FPS", "2.0"))
    capture_cooldown_sec: float = float(os.getenv("CAPTURE_COOLDOWN_SEC", "10.0"))
