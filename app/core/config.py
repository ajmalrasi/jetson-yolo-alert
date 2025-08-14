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
    # Tracker
    tracker_cfg: Optional[str] = os.getenv("TRACKER", "bytetrack.yaml")
    tracker_on: bool = os.getenv("TRACKER_ON", "1") not in ("0", "false", "False", "")
    # Telegram
    tg_token: Optional[str] = os.getenv("TELEGRAM_TOKEN")
    tg_chat: Optional[str]  = os.getenv("TELEGRAM_CHAT_ID")
    # Misc
    save_dir: str = os.getenv("SAVE_DIR", "/workspace/work/alerts")
    draw: bool = os.getenv("DRAW", "1") not in ("0","false","False","")
