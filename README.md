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
```

On macOS, allow camera access for the terminal or app that launches Python.

## Run

```bash
python run.py
```

Two windows should appear:

- MuJoCo viewer: drone simulation and render.
- Pygame face UI: webcam frame, MediaPipe face mesh, and control bars.

Press `Esc` or close the Pygame window to stop.

## Tuning

Most control sensitivity lives in `src/gesture_drone/face_control.py` and the drone stabilization gains live in `src/gesture_drone/simulator.py`.
