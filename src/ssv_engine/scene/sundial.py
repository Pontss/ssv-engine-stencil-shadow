"""Closed procedural sundial mesh shared by the visible pass and Z-Fail."""
from __future__ import annotations
import numpy as np
from ssv_engine.scene.mesh import TriangleMesh


def _cylinder(radius: float, y0: float, y1: float, segments: int = 48, center=(0.0, 0.0)):
    cx, cz = center
    vertices = [(cx, y0, cz), (cx, y1, cz)]
    for y in (y0, y1):
        for i in range(segments):
            a = 2.0 * np.pi * i / segments
            vertices.append((cx + radius*np.cos(a), y, cz + radius*np.sin(a)))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = 2+i, 2+j
        t0, t1 = 2+segments+i, 2+segments+j
        faces.extend(((0, b1, b0), (1, t0, t1), (b0, b1, t1), (b0, t1, t0)))
    # The original fan/side order is consistently inward; reverse every
    # triangle so the closed cylinder has outward-facing normals.
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)[:, [0, 2, 1]]


def _gnomon():
    # A tapered triangular pointer, tilted toward the north side of the dial.
    # The prism has a broad foot and a narrow raised tip.
    vertices = np.asarray([
        (-0.10, 0.58, -0.30), (0.10, 0.58, -0.30),
        (0.035, 2.28, 0.48), (-0.035, 2.28, 0.48),
        (-0.10, 0.58, -0.22), (0.10, 0.58, -0.22),
        (0.035, 2.28, 0.56), (-0.035, 2.28, 0.56),
    ], dtype=np.float32)
    faces = np.asarray([
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ], dtype=np.int32)
    # Orient every face away from the prism centroid.
    center = vertices.mean(axis=0)
    for i, face in enumerate(faces):
        a, b, c = vertices[face]
        if np.dot(np.cross(b-a, c-a), (a+b+c)/3-center) < 0:
            faces[i] = face[[0, 2, 1]]
    return vertices, faces


def sundial_mesh() -> TriangleMesh:
    parts = [
        _cylinder(1.22, 0.18, 0.34, 64),
        _cylinder(1.14, 0.34, 0.43, 64),
        _cylinder(0.98, 0.43, 0.49, 64),
        _cylinder(0.20, 0.49, 0.61, 32),
        _gnomon(),
    ]
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    mesh = TriangleMesh(np.vstack(vertices), np.vstack(faces))
    mesh.validate_closed_manifold()
    return mesh


def sundial_render_vertices(mesh: TriangleMesh) -> np.ndarray:
    rows = []
    for face in mesh.faces:
        points = mesh.vertices[face]
        normal = np.cross(points[1]-points[0], points[2]-points[0])
        normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
        for point in points:
            rows.extend((*point, *normal, float(point[0]*0.5+0.5), float(point[2]*0.5+0.5)))
    return np.asarray(rows, dtype=np.float32)
