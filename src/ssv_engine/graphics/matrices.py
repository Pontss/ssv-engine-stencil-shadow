"""Small matrix helpers; matrices are returned in NumPy row-major form."""

from __future__ import annotations

import math
import numpy as np


def perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    if aspect <= 0 or near <= 0 or far <= near:
        raise ValueError("Invalid perspective projection parameters")
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = (2.0 * far * near) / (near - far)
    result[3, 2] = -1.0
    return result


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    side = np.cross(forward, up)
    side = side / np.linalg.norm(side)
    corrected_up = np.cross(side, forward)
    result = np.identity(4, dtype=np.float32)
    result[0, :3] = side
    result[1, :3] = corrected_up
    result[2, :3] = -forward
    result[0, 3] = -np.dot(side, eye)
    result[1, 3] = -np.dot(corrected_up, eye)
    result[2, 3] = np.dot(forward, eye)
    return result


def translation(x: float, y: float, z: float) -> np.ndarray:
    result = np.identity(4, dtype=np.float32)
    result[:3, 3] = (x, y, z)
    return result


def scale(x: float, y: float, z: float) -> np.ndarray:
    return np.diag([x, y, z, 1.0]).astype(np.float32)


def infinite_perspective(fov_degrees: float, aspect: float, near: float) -> np.ndarray:
    """OpenGL right-handed projection with an infinite far plane."""
    if aspect <= 0 or near <= 0:
        raise ValueError("Invalid infinite perspective projection parameters")
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = -1.0
    result[2, 3] = -2.0 * near
    result[3, 2] = -1.0
    return result
