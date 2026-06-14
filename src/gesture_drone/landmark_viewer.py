from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import pygame

from .config import AppConfig, FACE_LANDMARKER_PATH, HAND_LANDMARKER_PATH


FACE_CONNECTIONS = (
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

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class LandmarkFrame:
    face_landmarks: list
    hand_landmarks: list
    handedness: list


@dataclass
class DroneGesture:
    vertical: str = "hover"
    motion: str = "hold"
    roll: float = 0.0
    pitch: float = 0.0
    has_right_hand: bool = False
    has_face: bool = False


class MediaPipeLandmarker:
    def __init__(self, face_model: Path, hand_model: Path) -> None:
        missing = [str(path) for path in (face_model, hand_model) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing MediaPipe model files:\n"
                + "\n".join(f"  {path}" for path in missing)
                + "\nRun: python scripts/download_models.py"
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base = python.BaseOptions
        self._mp = mp
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(face_model), delegate=base.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(hand_model), delegate=base.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._timestamp_ms = 0

    def close(self) -> None:
        self._face.close()
        self._hands.close()

    def detect(self, frame_rgb: np.ndarray) -> LandmarkFrame:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        timestamp = self._timestamp_ms
        self._timestamp_ms += 33
        face_result = self._face.detect_for_video(image, timestamp)
        hand_result = self._hands.detect_for_video(image, timestamp)
        return LandmarkFrame(
            face_landmarks=list(face_result.face_landmarks),
            hand_landmarks=list(hand_result.hand_landmarks),
            handedness=list(hand_result.handedness),
        )


class GestureController:
    """Classifies right-hand gestures and face direction into drone commands."""

    def __init__(self) -> None:
        self._smoothed_roll = 0.0
        self._smoothed_pitch = 0.0

    def classify(self, landmarks: LandmarkFrame) -> DroneGesture:
        gesture = DroneGesture()
        right_hand = self._right_hand(landmarks)

        if right_hand is not None:
            gesture.has_right_hand = True
            gesture.vertical, gesture.motion = self._classify_right_hand(right_hand)

        if landmarks.face_landmarks:
            gesture.has_face = True
            roll, pitch = self._face_direction(landmarks.face_landmarks[0])
            self._smoothed_roll = self._smooth(self._smoothed_roll, roll, 0.24)
            self._smoothed_pitch = self._smooth(self._smoothed_pitch, pitch, 0.24)
            gesture.roll = self._smoothed_roll
            gesture.pitch = self._smoothed_pitch
        else:
            self._smoothed_roll = self._smooth(self._smoothed_roll, 0.0, 0.12)
            self._smoothed_pitch = self._smooth(self._smoothed_pitch, 0.0, 0.12)

        return gesture

    def _right_hand(self, landmarks: LandmarkFrame):
        for hand, handedness in zip(landmarks.hand_landmarks, landmarks.handedness):
            if handedness and handedness[0].category_name == "Right":
                return hand

        if landmarks.hand_landmarks:
            return landmarks.hand_landmarks[0]
        return None

    def _classify_right_hand(self, hand) -> tuple[str, str]:
        thumb_up_score = hand[2].y - hand[4].y
        thumb_down_score = hand[4].y - hand[2].y
        thumb_is_vertical = abs(hand[4].y - hand[2].y) > abs(hand[4].x - hand[2].x) * 1.35
        folded_fingers = sum(not self._finger_extended(hand, tip, pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))

        if thumb_is_vertical and folded_fingers >= 2 and thumb_up_score > 0.08:
            return "up", "hold"
        if thumb_is_vertical and folded_fingers >= 2 and thumb_down_score > 0.08:
            return "down", "hold"

        open_fingers = sum(self._finger_extended(hand, tip, pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
        if open_fingers >= 3:
            return "hover", "forward" if self._palm_facing_camera(hand) else "back"

        return "hover", "hold"

    def _face_direction(self, face) -> tuple[float, float]:
        nose = face[1]
        left_cheek = face[234]
        right_cheek = face[454]
        forehead = face[10]
        chin = face[152]

        face_width = max(abs(right_cheek.x - left_cheek.x), 0.001)
        face_center_x = (left_cheek.x + right_cheek.x) * 0.5
        face_center_y = (forehead.y + chin.y) * 0.5

        roll = self._clamp((nose.x - face_center_x) / face_width * 2.4)
        pitch = self._clamp((face_center_y - nose.y) * 3.2)
        return roll, pitch

    def _finger_extended(self, hand, tip: int, pip: int) -> bool:
        return hand[tip].y < hand[pip].y - 0.025

    def _palm_facing_camera(self, hand) -> bool:
        index_mcp = hand[5]
        pinky_mcp = hand[17]
        thumb_tip = hand[4]
        palm_center_x = (index_mcp.x + pinky_mcp.x) * 0.5
        return thumb_tip.x < palm_center_x

    @staticmethod
    def _smooth(current: float, target: float, alpha: float) -> float:
        return current + (target - current) * alpha

    @staticmethod
    def _clamp(value: float) -> float:
        return float(max(-1.0, min(1.0, value)))


class LandmarkViewer:
    def __init__(self, width: int, height: int) -> None:
        pygame.init()
        pygame.display.set_caption("MediaPipe Face and Hand Landmarks")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height

    def close(self) -> None:
        pygame.quit()

    def should_continue(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def draw(self, frame_rgb: np.ndarray, landmarks: LandmarkFrame, gesture: DroneGesture, fps: float) -> None:
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        frame_surface = pygame.transform.smoothscale(frame_surface, (self.width, self.height))
        self.screen.blit(frame_surface, (0, 0))

        for face in landmarks.face_landmarks:
            self._draw_landmark_set(face, FACE_CONNECTIONS, (77, 217, 255), 1, 2)

        for hand in landmarks.hand_landmarks:
            self._draw_landmark_set(hand, HAND_CONNECTIONS, (124, 255, 156), 3, 5)

        self._status_dots(len(landmarks.face_landmarks), len(landmarks.hand_landmarks), fps)
        self._gesture_panel(gesture)

        pygame.display.flip()
        self.clock.tick(30)

    def _draw_landmark_set(self, landmark_set, connections, color, radius: int, width: int) -> None:
        points = [(int(point.x * self.width), int(point.y * self.height)) for point in landmark_set]

        for start, end in connections:
            if start < len(points) and end < len(points):
                pygame.draw.line(self.screen, color, points[start], points[end], width)

        for x, y in points:
            pygame.draw.circle(self.screen, color, (x, y), radius)

    def _status_dots(self, face_count: int, hand_count: int, fps: float) -> None:
        panel = pygame.Surface((168, 48), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 110))
        self.screen.blit(panel, (12, 12))

        face_color = (77, 217, 255) if face_count else (95, 100, 108)
        hand_color = (124, 255, 156) if hand_count else (95, 100, 108)
        fps_width = max(8, min(90, int(fps * 2)))

        pygame.draw.circle(self.screen, face_color, (32, 36), 9)
        pygame.draw.circle(self.screen, hand_color, (62, 36), 9)
        pygame.draw.rect(self.screen, (95, 100, 108), (88, 29, 92, 14), border_radius=4)
        pygame.draw.rect(self.screen, (245, 200, 88), (89, 30, fps_width, 12), border_radius=4)

    def _gesture_panel(self, gesture: DroneGesture) -> None:
        panel_w = 190
        panel_h = 176
        x = self.width - panel_w - 14
        y = 14
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 126))
        self.screen.blit(panel, (x, y))

        self._command_tile(x + 16, y + 16, gesture.vertical == "up", (92, 192, 255), "up")
        self._command_tile(x + 16, y + 72, gesture.vertical == "down", (255, 120, 112), "down")
        self._command_tile(x + 72, y + 44, gesture.motion == "forward", (124, 255, 156), "forward")
        self._command_tile(x + 128, y + 44, gesture.motion == "back", (255, 190, 86), "back")

        center = (x + panel_w // 2, y + 138)
        pygame.draw.circle(self.screen, (42, 47, 56), center, 25)
        pygame.draw.circle(self.screen, (77, 217, 255) if gesture.has_face else (95, 100, 108), center, 25, 2)
        self._draw_face_joystick(center, gesture.roll, gesture.pitch, gesture.has_face)

    def _command_tile(self, x: int, y: int, active: bool, color, direction: str) -> None:
        fill = color if active else (52, 57, 66)
        outline = (250, 252, 255) if active else (95, 100, 108)
        pygame.draw.rect(self.screen, fill, (x, y, 44, 44), border_radius=7)
        pygame.draw.rect(self.screen, outline, (x, y, 44, 44), 2, border_radius=7)

        cx = x + 22
        cy = y + 22
        if direction == "up":
            points = [(cx, cy - 14), (cx - 11, cy + 3), (cx - 4, cy + 3), (cx - 4, cy + 14), (cx + 4, cy + 14), (cx + 4, cy + 3), (cx + 11, cy + 3)]
        elif direction == "down":
            points = [(cx, cy + 14), (cx - 11, cy - 3), (cx - 4, cy - 3), (cx - 4, cy - 14), (cx + 4, cy - 14), (cx + 4, cy - 3), (cx + 11, cy - 3)]
        elif direction == "forward":
            points = [(cx, cy - 14), (cx - 13, cy + 10), (cx, cy + 4), (cx + 13, cy + 10)]
        else:
            points = [(cx, cy + 14), (cx - 13, cy - 10), (cx, cy - 4), (cx + 13, cy - 10)]
        pygame.draw.polygon(self.screen, (18, 20, 24), points)

    def _draw_face_joystick(self, center: tuple[int, int], roll: float, pitch: float, active: bool) -> None:
        color = (77, 217, 255) if active else (95, 100, 108)
        cx, cy = center
        radius = 25
        knob_x = int(cx + max(-1.0, min(1.0, roll)) * (radius - 7))
        knob_y = int(cy - max(-1.0, min(1.0, pitch)) * (radius - 7))

        pygame.draw.line(self.screen, (72, 78, 88), (cx - radius + 5, cy), (cx + radius - 5, cy), 1)
        pygame.draw.line(self.screen, (72, 78, 88), (cx, cy - radius + 5), (cx, cy + radius - 5), 1)
        pygame.draw.line(self.screen, color, (cx, cy), (knob_x, knob_y), 3)
        pygame.draw.circle(self.screen, color, (knob_x, knob_y), 7)


def main() -> int:
    config = AppConfig()
    viewer = LandmarkViewer(config.ui_width, config.ui_height)
    capture = cv2.VideoCapture(config.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

    if not capture.isOpened():
        print("Could not open webcam. Check macOS Camera permission for your terminal or editor.")
        viewer.close()
        return 1

    landmarker = MediaPipeLandmarker(FACE_LANDMARKER_PATH, HAND_LANDMARKER_PATH)
    gestures = GestureController()

    try:
        previous = time.perf_counter()
        fps = 0.0
        while viewer.should_continue():
            ok, frame_bgr = capture.read()
            if not ok:
                continue

            frame_rgb = cv2.flip(frame_bgr, 1)[:, :, ::-1]
            detections = landmarker.detect(frame_rgb)
            gesture = gestures.classify(detections)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous, 0.001)
            previous = now
            fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1

            viewer.draw(frame_rgb, detections, gesture, fps)
    finally:
        capture.release()
        landmarker.close()
        viewer.close()

    return 0
