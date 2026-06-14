from pathlib import Path
import os


def _prepare_runtime_cache() -> None:
    cache_dir = Path(__file__).resolve().parent / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


if __name__ == "__main__":
    _prepare_runtime_cache()

    try:
        from src.gesture_drone.app import main
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(f"Missing dependency: {missing}")
        print("Install the project dependencies with:")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        raise SystemExit(1) from exc

    raise SystemExit(main())
