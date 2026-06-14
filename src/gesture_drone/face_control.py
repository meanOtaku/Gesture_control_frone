from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceCommand:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    face_visible: bool = False


class FaceController:
    """Turns MediaPipe face landmarks into normalized drone commands."""

    def __init__(self) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smoothed = FaceCommand()

    def close(self) -> None:
        self._mesh.close()

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, FaceCommand]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        annotated = frame_bgr.copy()
        command = FaceCommand()

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            command = self._command_from_landmarks(face.landmark)
            self._draw_landmarks(annotated, face)

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

    def _draw_landmarks(self, frame_bgr: np.ndarray, face_landmarks) -> None:
        self._drawing.draw_landmarks(
            image=frame_bgr,
            landmark_list=face_landmarks,
            connections=self._mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._styles.get_default_face_mesh_tesselation_style(),
        )
        self._drawing.draw_landmarks(
            image=frame_bgr,
            landmark_list=face_landmarks,
            connections=self._mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._styles.get_default_face_mesh_contours_style(),
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return float(max(-1.0, min(1.0, value)))

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t
