"""Closed ornamental fountain mesh for the Greek courtyard."""
from __future__ import annotations
import numpy as np
from ssv_engine.scene.mesh import TriangleMesh
from ssv_engine.scene.sundial import _cylinder, sundial_render_vertices


def fountain_mesh() -> TriangleMesh:
    # A stepped limestone basin, inner water surface and central water pedestal.
    parts = [
        _cylinder(1.18, 0.08, 0.24, 64),
        _cylinder(1.28, 0.24, 0.38, 64),
        _cylinder(1.12, 0.38, 0.48, 64),
        _cylinder(0.96, 0.30, 0.34, 64),
        _cylinder(0.30, 0.34, 0.76, 32),
        _cylinder(0.40, 0.76, 0.88, 32),
    ]
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    mesh = TriangleMesh(np.vstack(vertices), np.vstack(faces))
    mesh.validate_closed_manifold()
    return mesh


def fountain_water_mesh() -> TriangleMesh:
    # A shallow, closed disk, visually distinct from the warm stone basin.
    v, f = _cylinder(1.02, 0.345, 0.365, 64)
    mesh = TriangleMesh(v, f)
    mesh.validate_closed_manifold()
    return mesh
