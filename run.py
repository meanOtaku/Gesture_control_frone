if __name__ == "__main__":
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
