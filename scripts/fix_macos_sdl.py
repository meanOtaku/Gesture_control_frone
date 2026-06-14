from pathlib import Path
import sysconfig


HOMEBREW_SDL = Path("/opt/homebrew/opt/sdl2/lib/libSDL2-2.0.0.dylib")


def main() -> int:
    if not HOMEBREW_SDL.exists():
        print(f"Homebrew SDL was not found at: {HOMEBREW_SDL}")
        return 1

    site_packages = Path(sysconfig.get_paths()["purelib"])
    matches = sorted(site_packages.glob("cv2/.dylibs/libSDL2-2.0.0.dylib"))
    matches.extend(sorted(site_packages.glob("cv2/.dylibs/libSDL2-2.0.0.dylib.disabled")))

    if not matches:
        print("No OpenCV SDL dylib found, nothing to change.")
        return 0

    for path in matches:
        target_path = path.with_name("libSDL2-2.0.0.dylib")
        backup_path = target_path.with_suffix(target_path.suffix + ".bundled")

        if target_path.is_symlink() and target_path.resolve() == HOMEBREW_SDL:
            print(f"Already linked to Homebrew SDL: {target_path}")
            continue

        if path.name.endswith(".disabled"):
            if not backup_path.exists():
                path.rename(backup_path)
            else:
                path.unlink()
        elif path.exists() and not backup_path.exists():
            path.rename(backup_path)
        elif path.exists():
            path.unlink()

        target_path.symlink_to(HOMEBREW_SDL)
        print(f"Linked OpenCV SDL to Homebrew SDL: {target_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
