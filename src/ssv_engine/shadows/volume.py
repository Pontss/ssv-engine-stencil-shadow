"""CPU construction of closed point-light shadow volumes for Z-Fail.

The output uses homogeneous coordinates: original cap vertices have w=1,
while vertices extruded to infinity have w=0. A rendering pass must use a
projection matrix compatible with homogeneous infinite points and must not
apply ordinary perspective division on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ssv_engine.scene.mesh import TriangleMesh




def _point_mesh_relation(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray, epsilon: float) -> str:
    """Classify a point against a closed oriented triangle mesh.

    Uses triangle-plane/barycentric checks for the boundary and the signed
    solid-angle sum for the interior.
    """
    solid_angle = 0.0
    for face in faces:
        a, b, c = vertices[face] - point
        la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        if min(la, lb, lc) <= epsilon:
            return "surface"

        edge0 = vertices[face[1]] - vertices[face[0]]
        edge1 = vertices[face[2]] - vertices[face[0]]
        normal = np.cross(edge0, edge1)
        normal_length = np.linalg.norm(normal)
        point_offset = point - vertices[face[0]]
        plane_distance = abs(float(np.dot(point_offset, normal))) / normal_length
        if plane_distance <= epsilon:
            projected = point - (np.dot(point_offset, normal) / np.dot(normal, normal)) * normal
            v0 = edge0
            v1 = edge1
            v2 = projected - vertices[face[0]]
            d00 = float(np.dot(v0, v0))
            d01 = float(np.dot(v0, v1))
            d11 = float(np.dot(v1, v1))
            d20 = float(np.dot(v2, v0))
            d21 = float(np.dot(v2, v1))
            denominator = d00 * d11 - d01 * d01
            if abs(denominator) > epsilon:
                v = (d11 * d20 - d01 * d21) / denominator
                w = (d00 * d21 - d01 * d20) / denominator
                u = 1.0 - v - w
                if min(u, v, w) >= -epsilon:
                    return "surface"

        numerator = float(np.dot(a, np.cross(b, c)))
        denominator = (la * lb * lc + float(np.dot(a, b)) * lc
                       + float(np.dot(b, c)) * la + float(np.dot(c, a)) * lb)
        solid_angle += 2.0 * np.arctan2(numerator, denominator)

    return "inside" if abs(solid_angle) > 2.0 * np.pi else "outside"


@dataclass(frozen=True)
class ShadowVolume:
    triangles: np.ndarray       # (T, 3, 4), homogeneous positions
    front_face_count: int
    back_face_count: int
    silhouette_edge_count: int

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.triangles.shape[0] * 3)


def build_shadow_volume(
    mesh: TriangleMesh,
    light_position: np.ndarray,
    model_matrix: np.ndarray | None = None,
    epsilon: float = 1e-7,
) -> ShadowVolume:
    """Build front cap, silhouette sides and infinite back cap for a point light.

    `light_position` is in world space. `model_matrix` transforms mesh-local
    vertices to world space. The mesh must be a closed, consistently wound,
    two-manifold triangle mesh, and the point light must be outside it. Exact
    edge-on faces use a deterministic infinitesimal tie-break to keep the
    generated volume topologically closed.
    """
    mesh.validate_closed_manifold()
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")
    light = np.asarray(light_position, dtype=np.float64)
    if light.shape != (3,) or not np.isfinite(light).all():
        raise ValueError("light_position must be a finite 3D vector")

    model = np.identity(4, dtype=np.float64) if model_matrix is None else np.asarray(model_matrix, dtype=np.float64)
    if model.shape != (4, 4) or not np.isfinite(model).all():
        raise ValueError("model_matrix must be a finite 4x4 matrix")
    if not np.allclose(model[3], (0.0, 0.0, 0.0, 1.0), atol=epsilon):
        raise ValueError("model_matrix must be affine (last row [0, 0, 0, 1])")
    linear_determinant = float(np.linalg.det(model[:3, :3]))
    if abs(linear_determinant) <= epsilon:
        raise ValueError("model_matrix must not collapse mesh dimensions")

    local_h = np.column_stack((mesh.vertices.astype(np.float64), np.ones(len(mesh.vertices))))
    world_h = (model @ local_h.T).T
    if np.any(np.abs(world_h[:, 3]) < epsilon):
        raise ValueError("Model transform produced a vertex at infinity")
    world = world_h[:, :3] / world_h[:, 3, None]

    triangles = mesh.faces.astype(np.int64)
    relation = _point_mesh_relation(light, world, triangles, epsilon)
    if relation == "surface":
        raise ValueError("Point light must not lie on the occluder surface")
    if relation == "inside":
        raise ValueError("Point light must be outside the closed occluder mesh")

    # A reflection changes handedness. Reverse face winding so the transformed
    # closed surface remains consistently outward-oriented in world space.
    if linear_determinant < 0:
        triangles = triangles[:, [0, 2, 1]]
    p0, p1, p2 = (world[triangles[:, i]] for i in range(3))
    normals = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0
    facing = np.einsum("ij,ij->i", normals, light - centroids)
    front = facing > epsilon
    back = facing < -epsilon
    coplanar = ~(front | back)
    if np.any(coplanar):
        # Resolve exact/near edge-on faces as the limit of an infinitesimal,
        # deterministic light displacement. Leaving them unclassified can
        # open the volume because front/back silhouette edges then disappear.
        tie_direction = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
        tie_values = normals[coplanar] @ tie_direction
        unresolved = np.abs(tie_values) <= epsilon
        if np.any(unresolved):
            tie_normals = normals[coplanar][unresolved]
            for row_index, normal in enumerate(tie_normals):
                nonzero = np.flatnonzero(np.abs(normal) > epsilon)
                tie_values[np.flatnonzero(unresolved)[row_index]] = normal[nonzero[0]]
        front[coplanar] = tie_values > 0.0
        back[coplanar] = tie_values < 0.0

    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(triangles):
        a, b, c = (int(v) for v in face)
        for start, end in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault(tuple(sorted((start, end))), []).append((face_index, start, end))

    output: list[np.ndarray] = []

    # Front cap: original light-facing triangles retain their outward winding.
    for face in triangles[front]:
        output.append(np.column_stack((world[face], np.ones(3))))

    # Side walls: one quad for every edge separating front and back faces.
    silhouette_edges = 0
    for adjacent in edge_faces.values():
        if len(adjacent) != 2:
            raise ValueError("Shadow-volume construction requires exactly two faces per edge")
        first, second = adjacent
        first_front = bool(front[first[0]])
        second_front = bool(front[second[0]])
        if first_front == second_front:
            continue
        silhouette_edges += 1
        # Use the directed edge as it appears on the light-facing triangle.
        start, end = (first[1], first[2]) if first_front else (second[1], second[2])
        near_start, near_end = world[start], world[end]
        far_start = near_start - light
        far_end = near_end - light
        # The side's outward winding is opposite the front-face boundary.
        quad = np.asarray([
            (*near_end, 1.0), (*near_start, 1.0),
            (*far_start, 0.0), (*far_end, 0.0),
        ], dtype=np.float64)
        output.append(quad[[0, 1, 2]])
        output.append(quad[[0, 2, 3]])

    # Infinite back cap: extrude the light-away faces. Their original winding
    # closes the side-wall boundary in homogeneous space.
    for face in triangles[back]:
        extruded = world[face] - light
        far_h = np.column_stack((extruded, np.zeros(3)))
        output.append(far_h[[0, 1, 2]])

    packed = np.asarray(output, dtype=np.float64).reshape((-1, 3, 4)) if output else np.empty((0, 3, 4), dtype=np.float64)
    return ShadowVolume(
        triangles=packed.astype(np.float32),
        front_face_count=int(np.count_nonzero(front)),
        back_face_count=int(np.count_nonzero(back)),
        silhouette_edge_count=silhouette_edges,
    )
