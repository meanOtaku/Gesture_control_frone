from __future__ import annotations

import cv2
import pygame
import numpy as np

from .face_control import FaceCommand


class FaceUi:
    def __init__(self, width: int, height: int) -> None:
        pygame.init()
        pygame.display.set_caption("MediaPipe Face Control")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Helvetica", 22)
        self.small_font = pygame.font.SysFont("Helvetica", 16)
        self.width = width
        self.height = height

    def close(self) -> None:
        pygame.quit()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def draw(self, frame_bgr: np.ndarray, command: FaceCommand) -> None:
        self.screen.fill((18, 20, 24))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        surface = pygame.surfarray.make_surface(frame_rgb)
        surface = pygame.transform.smoothscale(surface, (self.width, int(self.height * 0.78)))
        self.screen.blit(surface, (0, 0))

        panel_top = int(self.height * 0.78)
        pygame.draw.rect(self.screen, (28, 31, 36), (0, panel_top, self.width, self.height - panel_top))

        status = "FACE LOCKED" if command.face_visible else "SEARCHING"
        status_color = (92, 220, 156) if command.face_visible else (238, 183, 77)
        self._text(status, 24, panel_top + 18, status_color, self.font)

        bars = [
            ("Roll", command.roll, (95, 179, 255)),
            ("Pitch", command.pitch, (112, 220, 170)),
            ("Yaw", command.yaw, (245, 196, 99)),
            ("Throttle", command.throttle, (240, 118, 118)),
        ]

        start_x = 190
        for index, (label, value, color) in enumerate(bars):
            y = panel_top + 18 + index * 31
            self._bar(label, value, start_x, y, self.width - start_x - 32, color)

        self._text("Esc exits", 24, self.height - 30, (150, 158, 170), self.small_font)
        pygame.display.flip()
        self.clock.tick(30)

    def _bar(self, label: str, value: float, x: int, y: int, width: int, color) -> None:
        self._text(label, x, y - 2, (218, 222, 228), self.small_font)
        bar_x = x + 92
        bar_width = width - 92
        center = bar_x + bar_width // 2
        pygame.draw.rect(self.screen, (51, 56, 64), (bar_x, y, bar_width, 16), border_radius=4)
        pygame.draw.line(self.screen, (125, 132, 145), (center, y - 2), (center, y + 18), 2)
        fill = int((bar_width // 2) * max(-1.0, min(1.0, value)))
        rect = (center, y, fill, 16) if fill >= 0 else (center + fill, y, -fill, 16)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        self._text(f"{value:+.2f}", bar_x + bar_width + 10, y - 3, (218, 222, 228), self.small_font)

    def _text(self, text: str, x: int, y: int, color, font) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))
