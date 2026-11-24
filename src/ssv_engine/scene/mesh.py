"""Indexed, consistently wound triangle meshes for shadow-volume topology."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray  # (N, 3), object/local-space positions
    faces: np.ndarray     # (M, 3), counter-clockwise viewed from outside

    def validate_closed_manifold(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Mesh vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Mesh faces must have shape (M, 3)")
        if not np.issubdtype(self.faces.dtype, np.integer):
            raise ValueError("Mesh face indices must use an integer dtype")
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("Mesh cannot be empty")
        if not np.isfinite(self.vertices).all():
            raise ValueError("Mesh contains non-finite vertices")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("Mesh face references an invalid vertex")
        edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for face in self.faces:
            a, b, c = (int(i) for i in face)
            if len({a, b, c}) != 3:
                raise ValueError("Degenerate triangle has repeated vertex indices")
            p0, p1, p2 = self.vertices[[a, b, c]]
            if float(np.linalg.norm(np.cross(p1 - p0, p2 - p0))) <= 1e-10:
                raise ValueError("Degenerate triangle has zero area")
            for start, end in ((a, b), (b, c), (c, a)):
                edge_uses.setdefault(tuple(sorted((start, end))), []).append((start, end))
        for edge, uses in edge_uses.items():
            if len(uses) != 2:
                raise ValueError(f"Mesh is not closed/manifold at edge {edge}: expected 2 faces, got {len(uses)}")
            if uses[0] != uses[1][::-1]:
                raise ValueError(f"Mesh has inconsistent face winding at edge {edge}")


def cube_mesh() -> TriangleMesh:
    """Unit cube centered at origin with shared vertices and outward CCW faces."""
    vertices = np.asarray([
        (-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
        (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5),
    ], dtype=np.float32)
    faces = np.asarray([
        (0, 2, 1), (0, 3, 2),       # -Z
        (4, 5, 6), (4, 6, 7),       # +Z
        (0, 4, 7), (0, 7, 3),       # -X
        (1, 2, 6), (1, 6, 5),       # +X
        (0, 1, 5), (0, 5, 4),       # -Y
        (3, 7, 6), (3, 6, 2),       # +Y
    ], dtype=np.int32)
    mesh = TriangleMesh(vertices, faces)
    mesh.validate_closed_manifold()
    return mesh
