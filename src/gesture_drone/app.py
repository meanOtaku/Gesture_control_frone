from __future__ import annotations

import sys

import cv2

from .config import AppConfig, MODEL_PATH
from .face_control import FaceController, FaceCommand
from .simulator import DroneSimulator
from .ui import FaceUi


def main() -> int:
    config = AppConfig()

    capture = cv2.VideoCapture(config.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

    if not capture.isOpened():
        print("Could not open webcam. Check camera permissions and camera_index.")
        return 1

    face = FaceController()
    ui = FaceUi(config.ui_width, config.ui_height)
    sim = DroneSimulator(MODEL_PATH, config.sim_hz)
    sim.start_viewer()

    try:
        running = True
        latest_command = FaceCommand()
        while running and sim.is_running():
            running = ui.handle_events()

            ok, frame = capture.read()
            if ok:
                frame = cv2.flip(frame, 1)
                annotated, latest_command = face.process(frame)
                ui.draw(annotated, latest_command)

            sim.set_command(latest_command)
            sim.step_until_now()
    finally:
        capture.release()
        face.close()
        sim.close()
        ui.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
