from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class FaceCommand:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    face_visible: bool = False


class FaceController:
    """Turns MediaPipe face landmarks into normalized drone commands."""

    _CONNECTIONS = (
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
        (154, 155), (155, 133), (362, 382), (382, 381), (381, 380), (380, 374),
        (374, 373), (373, 390), (390, 249), (249, 263), (61, 146), (146, 91),
        (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321),
        (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
        (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    )

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe face model: {model_path}\n"
                "Download it with:\n"
                "  python scripts/download_models.py"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._smoothed = FaceCommand()
        self._timestamp_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, FaceCommand]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        results = self._landmarker.detect_for_video(image, self._timestamp_ms)
        self._timestamp_ms += 33

        annotated = frame_rgb.copy()
        command = FaceCommand()

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            command = self._command_from_landmarks(landmarks)
            self._draw_landmarks(annotated, landmarks)

        self._smoothed = self._smooth(command)
        return annotated, self._smoothed

    def _command_from_landmarks(self, landmarks) -> FaceCommand:
        nose = landmarks[1]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        face_width = max(abs(right_cheek.x - left_cheek.x), 0.001)
        face_height = max(abs(bottom_lip.y - nose.y), 0.001)

        face_center_x = (left_cheek.x + right_cheek.x) * 0.5
        nose_offset_x = (nose.x - face_center_x) / face_width
        nose_offset_y = (nose.y - 0.5) * 2.0

        mouth_open = abs(bottom_lip.y - top_lip.y) / face_height
        mouth_width = abs(right_mouth.x - left_mouth.x) / face_width

        return FaceCommand(
            roll=self._clamp(nose_offset_x * 2.2),
            pitch=self._clamp(-nose_offset_y * 1.2),
            yaw=self._clamp((mouth_width - 0.8) * 2.0),
            throttle=self._clamp((mouth_open - 0.16) * 3.0),
            face_visible=True,
        )

    def _smooth(self, command: FaceCommand) -> FaceCommand:
        alpha = 0.22 if command.face_visible else 0.08
        return FaceCommand(
            roll=self._lerp(self._smoothed.roll, command.roll, alpha),
            pitch=self._lerp(self._smoothed.pitch, command.pitch, alpha),
            yaw=self._lerp(self._smoothed.yaw, command.yaw, alpha),
            throttle=self._lerp(self._smoothed.throttle, command.throttle, alpha),
            face_visible=command.face_visible,
        )

    def _draw_landmarks(self, frame_rgb: np.ndarray, landmarks) -> None:
        height, width = frame_rgb.shape[:2]
        points = [(int(point.x * width), int(point.y * height)) for point in landmarks]

        for start, end in self._CONNECTIONS:
            self._draw_line(frame_rgb, points[start], points[end], (80, 220, 255))

        for index in (1, 13, 14, 33, 61, 133, 234, 263, 291, 362, 454):
            self._draw_square(frame_rgb, points[index], (255, 245, 120))

    def _draw_line(self, frame_rgb: np.ndarray, start: tuple[int, int], end: tuple[int, int], color) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1).astype(int)
        ys = np.linspace(y0, y1, steps + 1).astype(int)
        valid = (0 <= xs) & (xs < frame_rgb.shape[1]) & (0 <= ys) & (ys < frame_rgb.shape[0])
        frame_rgb[ys[valid], xs[valid]] = color

    def _draw_square(self, frame_rgb: np.ndarray, center: tuple[int, int], color) -> None:
        x, y = center
        x0 = max(0, x - 2)
        x1 = min(frame_rgb.shape[1], x + 3)
        y0 = max(0, y - 2)
        y1 = min(frame_rgb.shape[0], y + 3)
        frame_rgb[y0:y1, x0:x1] = color

    @staticmethod
    def _clamp(value: float) -> float:
        return float(max(-1.0, min(1.0, value)))

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t
