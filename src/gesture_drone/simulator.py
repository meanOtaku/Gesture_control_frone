from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .face_control import FaceCommand


class DroneSimulator:
    """MuJoCo simulation wrapper that applies face commands as body forces."""

    def __init__(self, model_path: Path, sim_hz: float) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.dt = 1.0 / sim_hz
        self.command = FaceCommand()
        self._viewer = None
        self._last_step = time.perf_counter()

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def start_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.cam.distance = 5.0
        self._viewer.cam.elevation = -25
        self._viewer.cam.azimuth = 135

    def is_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

    def set_command(self, command: FaceCommand) -> None:
        self.command = command

    def step_until_now(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_step
        steps = max(1, min(8, int(elapsed / self.dt)))

        for _ in range(steps):
            self._apply_control()
            mujoco.mj_step(self.model, self.data)
            self._last_step += self.dt

        if self._viewer is not None:
            self._viewer.sync()

    def _apply_control(self) -> None:
        self.data.xfrc_applied[:, :] = 0.0

        mass = self.model.body_mass[self.body_id]
        gravity = abs(float(self.model.opt.gravity[2]))

        quat = self.data.xquat[self.body_id].copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)

        world_up = np.array([0.0, 0.0, 1.0])
        body_up = rot[:, 2]
        body_forward = rot[:, 0]
        body_right = rot[:, 1]

        target_roll = -self.command.roll * math.radians(22)
        target_pitch = self.command.pitch * math.radians(18)
        target_yaw_rate = self.command.yaw * math.radians(55)

        roll_error = float(np.dot(body_up, body_right)) + target_roll
        pitch_error = float(np.dot(body_up, body_forward)) + target_pitch

        angular_velocity = self.data.cvel[self.body_id, 0:3]
        linear_velocity = self.data.cvel[self.body_id, 3:6]

        thrust = mass * (gravity + self.command.throttle * 5.5)
        altitude_hold = (1.4 - self.data.xpos[self.body_id, 2]) * 2.0
        vertical_damping = -linear_velocity[2] * 1.4
        force_world = world_up * (thrust + mass * (altitude_hold + vertical_damping))

        torque_world = (
            body_right * (-pitch_error * 3.8)
            + body_forward * (roll_error * 3.8)
            + world_up * (target_yaw_rate - angular_velocity[2]) * 1.2
            - angular_velocity * 0.35
        )

        self.data.xfrc_applied[self.body_id, 0:3] = force_world
        self.data.xfrc_applied[self.body_id, 3:6] = torque_world
