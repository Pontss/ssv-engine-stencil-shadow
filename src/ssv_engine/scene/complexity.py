"""Extra shadow-casting crates used to scale geometric complexity (no OpenGL needed)."""

from math import cos, sin

import numpy as np

from ssv_engine.graphics.matrices import translation, scale

MAX_COMPLEXITY_LEVEL = 5
CUBE_TRIANGLES = 12


def complexity_objects(level: int) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    """Deterministic extra crates arranged in rings around the plaza.

    Level 0 adds nothing; each further level adds one ring (16, 22, 28, ...
    crates). All crates stay below the minimum light height (2.45), so the
    point light is never inside an occluder.
    """
    objects = []
    for ring in range(1, max(0, min(int(level), MAX_COMPLEXITY_LEVEL)) + 1):
        radius = 6.2 + 1.5 * (ring - 1)
        count = 10 + 6 * ring
        for i in range(count):
            angle = 2.0 * np.pi * (i + 0.5 * (ring % 2)) / count
            width = 0.45 + 0.25 * ((i * 37 + ring * 11) % 5) / 4.0
            height = 0.45 + 0.55 * ((i * 53 + ring * 7) % 4) / 3.0
            spin = angle + 0.35 * ((i * 17) % 3)
            rotation = np.identity(4, dtype=np.float32)
            rotation[0, 0], rotation[0, 2] = cos(spin), sin(spin)
            rotation[2, 0], rotation[2, 2] = -sin(spin), cos(spin)
            model = (translation(radius * cos(angle), height / 2.0, radius * sin(angle))
                     @ rotation @ scale(width, height, width))
            shade = 0.82 + 0.18 * ((i * 29 + ring) % 3) / 2.0
            objects.append((model.astype(np.float32), (shade, shade * 0.9, shade * 0.76)))
    return objects
