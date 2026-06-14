# Face and Gesture Controlled Drone in MuJoCo

This project runs a small MuJoCo drone simulation while MediaPipe reads your face from a webcam. MuJoCo owns the simulation and 3D render window. Pygame owns a separate camera UI that shows the MediaPipe face frame and the current control values.

## What It Does

- Head left/right controls drone roll.
- Head up/down controls drone pitch.
- Mouth opening controls throttle.
- A face smile-like signal controls yaw.
- Losing the face returns the drone gently toward hover.

This is a starter control rig, not a physically exact quadrotor autopilot. The MuJoCo model is intentionally simple so the face-control pipeline is easy to understand and extend.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python scripts/fix_macos_sdl.py
```

On macOS, allow camera access for the terminal or app that launches Python.

## Run

Landmark-only viewer:

```bash
python run_landmarks.py
```

Full MuJoCo drone demo:

```bash
python run.py
```

For the full demo, two windows should appear:

- MuJoCo viewer: drone simulation and render.
- Pygame face UI: webcam frame, MediaPipe face mesh, and control bars.

Press `Esc` or close the Pygame window to stop.

On macOS, `python scripts/fix_macos_sdl.py` points OpenCV at the same SDL library Pygame uses. If you previously installed `opencv-python` directly, remove it so the app does not load an extra OpenCV wheel:

```bash
pip uninstall opencv-python
```

## Tuning

Most control sensitivity lives in `src/gesture_drone/face_control.py` and the drone stabilization gains live in `src/gesture_drone/simulator.py`.
