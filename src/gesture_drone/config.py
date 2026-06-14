from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "drone.xml"


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    camera_width: int = 960
    camera_height: int = 540
    ui_width: int = 960
    ui_height: int = 700
    sim_hz: float = 100.0
