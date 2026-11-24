"""First-person camera controlled by keyboard and mouse."""

from __future__ import annotations

import math
import numpy as np

from ssv_engine.graphics.matrices import look_at


class Camera:
    def __init__(self, position=(3.8, 2.9, 5.8), yaw=-125.0, pitch=-18.0):
        self.position = np.array(position, dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.speed = 4.0
        self.sensitivity = 0.12
        self._update_vectors()

    def reset(self) -> None:
        self.position = np.array([3.8, 2.9, 5.8], dtype=np.float32)
        self.yaw = -125.0
        self.pitch = -18.0
        self._update_vectors()

    def _update_vectors(self) -> None:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        direction = np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.sin(yaw) * math.cos(pitch),
        ], dtype=np.float32)
        self.front = direction / np.linalg.norm(direction)
        self.right = np.cross(self.front, self.world_up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)

    def rotate(self, x_offset: float, y_offset: float) -> None:
        self.yaw += x_offset * self.sensitivity
        self.pitch = float(np.clip(self.pitch + y_offset * self.sensitivity, -88.0, 88.0))
        self._update_vectors()

    def move(self, keys: set[int], delta_time: float, glfw_module) -> None:
        velocity = self.speed * min(delta_time, 0.05)
        if glfw_module.KEY_W in keys:
            self.position += self.front * velocity
        if glfw_module.KEY_S in keys:
            self.position -= self.front * velocity
        if glfw_module.KEY_A in keys:
            self.position -= self.right * velocity
        if glfw_module.KEY_D in keys:
            self.position += self.right * velocity
        if glfw_module.KEY_SPACE in keys:
            self.position += self.world_up * velocity
        if glfw_module.KEY_LEFT_SHIFT in keys:
            self.position -= self.world_up * velocity

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.position + self.front, self.up)
