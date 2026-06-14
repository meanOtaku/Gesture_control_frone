from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloads = [
        (FACE_MODEL, FACE_MODEL_URL),
        (HAND_MODEL, HAND_MODEL_URL),
    ]

    for path, url in downloads:
        if path.exists():
            print(f"Already present: {path}")
            continue
        print(f"Downloading {path.name}...")
        urlretrieve(url, path)
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
