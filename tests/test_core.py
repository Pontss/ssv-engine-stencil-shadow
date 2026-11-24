"""Headless tests for mathematical and geometry foundations."""

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ssv_engine.graphics.matrices import look_at, perspective, infinite_perspective
from ssv_engine.scene.primitives import cube_vertices, plane_vertices, sphere_vertices
from ssv_engine.camera.controller import Camera
from ssv_engine.scene.mesh import cube_mesh
from ssv_engine.shadows.volume import build_shadow_volume
from ssv_engine.metrics.benchmark import BenchmarkRecorder, ComparisonBenchmark
from ssv_engine.scene.materials import make_material_pixels


class MatrixTests(unittest.TestCase):
    def test_perspective_has_expected_shape_and_values(self):
        matrix = perspective(60.0, 16 / 9, 0.1, 100.0)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertAlmostEqual(float(matrix[3, 2]), -1.0)
        self.assertGreater(float(matrix[0, 0]), 0.0)

    def test_infinite_perspective_maps_homogeneous_infinity_to_far_clip_plane(self):
        matrix = infinite_perspective(55.0, 16 / 9, 0.1)
        # A direction pointing down camera -Z has clip z == clip w at infinity.
        clip = matrix @ np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(float(clip[2]), float(clip[3]), places=6)
        self.assertGreater(float(clip[3]), 0.0)

    def test_look_at_is_finite(self):
        matrix = look_at(
            np.array([0.0, 0.0, 3.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.isfinite(matrix).all())


class GeometryTests(unittest.TestCase):
    def test_cube_has_36_vertices_with_position_normal_and_uv(self):
        vertices = cube_vertices()
        self.assertEqual(vertices.shape, (36 * 8,))
        self.assertTrue(np.isfinite(vertices).all())
        uv = vertices.reshape(-1, 8)[:, 6:8]
        self.assertTrue(((uv >= 0.0) & (uv <= 1.0)).all())

    def test_sphere_has_position_normal_and_uv(self):
        vertices = sphere_vertices(8, 12).reshape(-1, 8)
        self.assertEqual(vertices.shape, (8 * 12 * 6, 8))
        self.assertTrue(np.isfinite(vertices).all())
        np.testing.assert_allclose(np.linalg.norm(vertices[:, 3:6], axis=1), 1.0, atol=1e-5)

    def test_plane_has_two_triangles(self):
        vertices = plane_vertices()
        self.assertEqual(vertices.shape, (6 * 8,))
        uv = vertices.reshape(-1, 8)[:, 6:8]
        self.assertGreater(float(uv.max()), 1.0)  # repeated floor texture


class MaterialTests(unittest.TestCase):
    def test_procedural_materials_are_deterministic_rgba_maps(self):
        for kind in ("floor", "crate", "stone", "wood"):
            first = make_material_pixels(kind, 64)
            second = make_material_pixels(kind, 64)
            self.assertEqual(first.shape, (64, 64, 4))
            self.assertEqual(first.dtype, np.uint8)
            np.testing.assert_array_equal(first, second)
            self.assertGreater(int(np.unique(first[:, :, :3].reshape(-1, 3), axis=0).shape[0]), 40)
        floor = make_material_pixels("floor", 64)
        crate = make_material_pixels("crate", 64)
        stone = make_material_pixels("stone", 64)
        wood = make_material_pixels("wood", 64)
        np.testing.assert_array_equal(floor[0, 0, :3], [105, 87, 66])
        np.testing.assert_array_equal(crate[0, 0, :3], [27, 38, 50])
        np.testing.assert_array_equal(crate[0, 15, :3], [205, 119, 43])
        self.assertEqual(stone.shape, (64, 64, 4))
        self.assertEqual(wood.shape, (64, 64, 4))

    def test_unknown_material_kind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Material kind"):
            make_material_pixels("metal", 64)

    def test_invalid_material_texture_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 16"):
            make_material_pixels("floor", 12)


class ShadowVolumeTests(unittest.TestCase):
    def test_fountain_meshes_are_closed_and_cast_shadow_volumes(self):
        from ssv_engine.scene.fountain import fountain_mesh, fountain_water_mesh
        mesh = fountain_mesh()
        water = fountain_water_mesh()
        mesh.validate_closed_manifold()
        water.validate_closed_manifold()
        model = np.identity(4, dtype=np.float64)
        model[0, 3] = -3.0
        model[2, 3] = -1.0
        volume = build_shadow_volume(mesh, np.array([0.0, 4.8, 0.0]), model)
        self.assertGreater(volume.triangle_count, 0)
        self.assertGreater(volume.silhouette_edge_count, 0)

    def test_sundial_is_closed_and_builds_shadow_volumes(self):
        from ssv_engine.scene.sundial import sundial_mesh
        mesh = sundial_mesh()
        mesh.validate_closed_manifold()
        for light in (np.array([4.8, 4.8, 0.0]), np.array([0.0, 5.0, 4.8]), np.array([-4.8, 4.8, 0.0])):
            volume = build_shadow_volume(mesh, light)
            self.assertGreater(volume.triangle_count, 0)
            self.assertGreater(volume.silhouette_edge_count, 0)

    def test_cube_is_closed_and_consistently_wound(self):
        mesh = cube_mesh()
        mesh.validate_closed_manifold()
        self.assertEqual(mesh.vertices.shape, (8, 3))
        self.assertEqual(mesh.faces.shape, (12, 3))
        for face in mesh.faces:
            a, b, c = mesh.vertices[face]
            normal = np.cross(b - a, c - a)
            centroid = (a + b + c) / 3.0
            self.assertGreater(float(np.dot(normal, centroid)), 0.0)

    def test_point_light_builds_front_back_caps_and_silhouette_sides(self):
        volume = build_shadow_volume(cube_mesh(), np.array([2.0, 4.0, 3.0]))
        self.assertGreater(volume.front_face_count, 0)
        self.assertGreater(volume.back_face_count, 0)
        self.assertGreater(volume.silhouette_edge_count, 0)
        self.assertEqual(volume.triangle_count, volume.front_face_count + volume.back_face_count + 2 * volume.silhouette_edge_count)
        self.assertEqual(volume.triangles.shape[1:], (3, 4))
        self.assertTrue(np.isfinite(volume.triangles).all())

    def test_shadow_volume_uses_homogeneous_infinite_vertices(self):
        volume = build_shadow_volume(cube_mesh(), np.array([3.0, 5.0, 2.0]))
        w_values = set(np.unique(volume.triangles[:, :, 3]).tolist())
        self.assertEqual(w_values, {0.0, 1.0})

    def test_generated_volume_is_closed_with_consistent_winding(self):
        volume = build_shadow_volume(cube_mesh(), np.array([2.0, 4.0, 3.0]))
        edge_uses = {}
        for triangle in volume.triangles:
            for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                a = tuple(np.round(start, 6))
                b = tuple(np.round(end, 6))
                edge_uses.setdefault(tuple(sorted((a, b))), []).append((a, b))
        self.assertTrue(edge_uses)
        for uses in edge_uses.values():
            self.assertEqual(len(uses), 2)
            self.assertEqual(uses[0], uses[1][::-1])

    def test_model_transform_is_applied_before_silhouette_detection(self):
        model = np.identity(4, dtype=np.float64)
        model[0, 3] = 3.0
        volume = build_shadow_volume(cube_mesh(), np.array([3.0, 4.0, 4.0]), model)
        finite_points = volume.triangles[volume.triangles[:, :, 3] == 1.0]
        self.assertAlmostEqual(float(finite_points[:, 0].mean()), 3.0, places=5)

    def test_all_scene_cube_transforms_produce_closed_shadow_volumes(self):
        from ssv_engine.graphics.matrices import translation, scale
        light = np.array([0.0, 4.8, 0.0], dtype=np.float64)
        # Keep this regression test synchronized with BasicScene.objects.
        transforms = (
            translation(0.0, 0.82, -3.6) @ scale(1.64, 1.64, 1.64),
            translation(-2.7, 0.48, -1.3) @ scale(0.96, 0.96, 0.96),
            translation(2.35, 0.62, -1.9) @ scale(1.24, 1.24, 1.24),
            translation(-4.0, 1.25, -2.7) @ scale(0.58, 2.5, 0.58),
            translation(4.0, 1.25, -2.7) @ scale(0.58, 2.5, 0.58),
        )
        for model in transforms:
            volume = build_shadow_volume(cube_mesh(), light, model)
            self.assertGreater(volume.triangle_count, 0)
            edge_uses = {}
            for triangle in volume.triangles:
                for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                    a = tuple(np.round(start, 6))
                    b = tuple(np.round(end, 6))
                    edge_uses.setdefault(tuple(sorted((a, b))), []).append((a, b))
            self.assertTrue(all(len(uses) == 2 and uses[0] == uses[1][::-1] for uses in edge_uses.values()))

    def test_reflected_model_transform_preserves_closed_volume_winding(self):
        model = np.identity(4, dtype=np.float64)
        model[0, 0] = -1.0
        model[0, 3] = 1.5
        volume = build_shadow_volume(cube_mesh(), np.array([3.0, 4.0, 2.0]), model)
        edge_uses = {}
        for triangle in volume.triangles:
            for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                a = tuple(np.round(start, 6))
                b = tuple(np.round(end, 6))
                edge_uses.setdefault(tuple(sorted((a, b))), []).append((a, b))
        self.assertTrue(all(len(uses) == 2 and uses[0] == uses[1][::-1] for uses in edge_uses.values()))

    def test_point_light_inside_occluder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the closed occluder"):
            build_shadow_volume(cube_mesh(), np.array([0.0, 0.0, 0.0]))

    def test_point_light_on_occluder_surface_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not lie on the occluder surface"):
            build_shadow_volume(cube_mesh(), np.array([0.5, 0.0, 0.0]))

    def test_singular_model_transform_is_rejected(self):
        model = np.identity(4, dtype=np.float64)
        model[1, 1] = 0.0
        with self.assertRaisesRegex(ValueError, "collapse mesh dimensions"):
            build_shadow_volume(cube_mesh(), np.array([0.0, 2.0, 3.0]), model)

    def test_non_affine_model_transform_is_rejected(self):
        model = np.identity(4, dtype=np.float64)
        model[3, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "must be affine"):
            build_shadow_volume(cube_mesh(), np.array([0.0, 2.0, 3.0]), model)

    def test_invalid_epsilon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "epsilon"):
            build_shadow_volume(cube_mesh(), np.array([0.0, 2.0, 3.0]), epsilon=0.0)

    def test_shadow_volume_remains_closed_for_lights_around_cube(self):
        positions = (
            (3.0, 2.0, 1.0), (-3.0, 2.0, 1.0), (1.0, 2.0, 3.0),
            (1.0, 2.0, -3.0), (3.0, 0.2, -2.0), (-2.0, 4.0, -3.0),
            (2.0, 0.5, 0.0), (0.0, 2.0, 0.5), (2.0, 0.0, 0.5),
        )
        for position in positions:
            volume = build_shadow_volume(cube_mesh(), np.asarray(position, dtype=np.float64))
            self.assertEqual(volume.front_face_count + volume.back_face_count, 12)
            edge_uses = {}
            for triangle in volume.triangles:
                for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                    a = tuple(np.round(start, 6))
                    b = tuple(np.round(end, 6))
                    edge_uses.setdefault(tuple(sorted((a, b))), []).append((a, b))
            self.assertTrue(all(len(uses) == 2 and uses[0] == uses[1][::-1] for uses in edge_uses.values()), position)

    def test_complexity_crates_build_closed_volumes_below_light(self):
        from ssv_engine.scene.complexity import complexity_objects, MAX_COMPLEXITY_LEVEL
        self.assertEqual(complexity_objects(0), [])
        crates = complexity_objects(MAX_COMPLEXITY_LEVEL)
        self.assertEqual(len(crates), sum(10 + 6 * r for r in range(1, MAX_COMPLEXITY_LEVEL + 1)))
        mesh = cube_mesh()
        for model, _color in crates[::7]:
            top = (model @ np.array([0.0, 0.5, 0.0, 1.0]))[1]
            self.assertLess(top, 2.45)
            volume = build_shadow_volume(mesh, np.array([0.0, 2.45, 0.0]), model)
            self.assertGreater(volume.silhouette_edge_count, 0)

    def test_degenerate_mesh_face_is_rejected(self):
        from ssv_engine.scene.mesh import TriangleMesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
        mesh = TriangleMesh(vertices, faces)
        with self.assertRaisesRegex(ValueError, "zero area"):
            mesh.validate_closed_manifold()

    def test_invalid_open_mesh_is_rejected(self):
        from ssv_engine.scene.mesh import TriangleMesh
        mesh = TriangleMesh(np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32), np.array([[0,1,2]], dtype=np.int32))
        with self.assertRaises(ValueError):
            build_shadow_volume(mesh, np.array([0.0, 0.0, 2.0]))


class CameraTests(unittest.TestCase):
    def test_camera_rotation_keeps_pitch_bounded(self):
        camera = Camera()
        camera.rotate(0.0, 10000.0)
        self.assertLessEqual(camera.pitch, 88.0)
        camera.rotate(0.0, -20000.0)
        self.assertGreaterEqual(camera.pitch, -88.0)

    def test_camera_view_matrix_is_finite(self):
        camera = Camera()
        self.assertTrue(np.isfinite(camera.view_matrix()).all())

    def test_camera_reset_restores_initial_pose(self):
        camera = Camera()
        camera.rotate(150.0, 80.0)
        camera.position += np.array([2.0, -1.0, 3.0], dtype=np.float32)
        camera.reset()
        np.testing.assert_allclose(camera.position, [3.8, 2.9, 5.8])
        self.assertAlmostEqual(camera.yaw, -125.0)
        self.assertAlmostEqual(camera.pitch, -18.0)


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_writes_csv_and_returns_summary(self):
        import tempfile
        from pathlib import Path

        recorder = BenchmarkRecorder()
        recorder.start()
        recorder.record(1 / 60, True, 1280, 720, (2.0, 4.0, 1.5), (4.5, 3.6, 6.5))
        recorder.record(1 / 50, False, 1280, 720, (2.0, 4.0, 1.5), (4.5, 3.6, 6.5))
        with tempfile.TemporaryDirectory() as directory:
            path, summary = recorder.stop(directory)
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("shadows_enabled", content)
            self.assertEqual(summary["frames"], 2)
            self.assertGreater(summary["average_fps"], 0)

    def test_comparison_benchmark_runs_both_35_second_conditions(self):
        comparison = ComparisonBenchmark()
        comparison.start()
        self.assertEqual(comparison.phase, "warmup_on")
        for _ in range(30):
            comparison.update(0.1)
        self.assertEqual(comparison.phase, "measure_on")
        self.assertTrue(comparison.shadows_enabled)
        for _ in range(351):
            comparison.record(1 / 60, 1280, 720, (0, 4.8, 0), (0, 2, 8))
            comparison.update(0.1)
        self.assertEqual(comparison.phase, "warmup_off")
        self.assertFalse(comparison.shadows_enabled)
        for _ in range(30):
            comparison.update(0.1)
        self.assertEqual(comparison.phase, "measure_off")
        for _ in range(351):
            comparison.record(1 / 75, 1280, 720, (0, 4.8, 0), (0, 2, 8))
            comparison.update(0.1)
        self.assertEqual(comparison.phase, "complete")
        self.assertFalse(comparison.active)
        self.assertGreaterEqual(comparison.results["ON"]["frames"], 349)
        self.assertGreaterEqual(comparison.results["OFF"]["frames"], 349)
        self.assertGreater(comparison.results["OFF"]["average_fps"], comparison.results["ON"]["average_fps"])

    def test_comparison_benchmark_saves_readable_summary(self):
        import tempfile
        comparison = ComparisonBenchmark()
        comparison.start()
        comparison.phase = "complete"
        comparison.active = False
        comparison.rows = [
            {"condition": "ON", "frame_time_ms": 10.0, "fps_instant": 100.0, "width": 1280, "height": 720,
             "light_x": 0.0, "light_y": 4.8, "light_z": 0.0, "camera_x": 0.0, "camera_y": 2.0, "camera_z": 8.0},
            {"condition": "OFF", "frame_time_ms": 8.0, "fps_instant": 125.0, "width": 1280, "height": 720,
             "light_x": 0.0, "light_y": 4.8, "light_z": 0.0, "camera_x": 0.0, "camera_y": 2.0, "camera_z": 8.0},
        ]
        comparison._summarize()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = comparison.save(directory)
            self.assertTrue(csv_path.exists())
            self.assertTrue(csv_path.with_suffix(".txt").exists())
            self.assertIn("Sombras ligadas", csv_path.with_suffix(".txt").read_text(encoding="utf-8"))

    def test_complexity_sweep_measures_every_level(self):
        import tempfile
        comparison = ComparisonBenchmark()
        comparison.start(levels=[0, 1, 2], warmup_seconds=0.2, measure_seconds=0.5)
        self.assertTrue(comparison.sweep)
        seen_levels = set()
        for _ in range(1000):
            if not comparison.active:
                break
            seen_levels.add(comparison.current_level)
            frame = 1 / 100 if comparison.shadows_enabled else 1 / 120
            comparison.record(frame, 1280, 720, (0, 4.8, 0), (0, 2, 8),
                              {"objects": 10 + comparison.current_level, "scene_triangles": 100, "volume_triangles": 50})
            comparison.update(0.05)
        self.assertEqual(comparison.phase, "complete")
        self.assertEqual(seen_levels, {0, 1, 2})
        for level in (0, 1, 2):
            self.assertGreater(comparison.results_by_level[level]["ON"]["frames"], 0)
            self.assertGreater(comparison.results_by_level[level]["OFF"]["average_fps"],
                               comparison.results_by_level[level]["ON"]["average_fps"])
            self.assertEqual(comparison.scene_info_by_level[level]["objects"], 10 + level)
        with tempfile.TemporaryDirectory() as directory:
            path = comparison.save(directory, machine_info={"GPU": "test"})
            text = path.with_suffix(".txt").read_text(encoding="utf-8")
            self.assertIn("Nível 2", text)
            self.assertIn("GPU: test", text)

    def test_cancelled_benchmark_returns_to_idle(self):
        comparison = ComparisonBenchmark()
        comparison.start()
        comparison.cancel()
        self.assertFalse(comparison.active)
        self.assertEqual(comparison.phase, "idle")

    def test_stopping_inactive_benchmark_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "No benchmark recording"):
            BenchmarkRecorder().stop()

    def test_starting_an_active_benchmark_is_rejected_without_discarding_samples(self):
        recorder = BenchmarkRecorder()
        recorder.start()
        recorder.record(1 / 60, True, 1280, 720, (2.0, 4.0, 1.5), (4.5, 3.6, 6.5))
        with self.assertRaisesRegex(RuntimeError, "already active"):
            recorder.start()
        self.assertEqual(len(recorder._rows), 1)


if __name__ == "__main__":
    unittest.main()
