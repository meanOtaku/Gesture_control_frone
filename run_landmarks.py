from pathlib import Path
import os


def _prepare_runtime_cache() -> None:
    cache_dir = Path(__file__).resolve().parent / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


if __name__ == "__main__":
    _prepare_runtime_cache()
    from src.gesture_drone.landmark_viewer import main

    raise SystemExit(main())
