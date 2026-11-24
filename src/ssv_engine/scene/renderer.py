"""Lit scene renderer with a two-sided stencil Z-Fail shadow-volume pass."""

from pathlib import Path
from time import perf_counter

import moderngl
import numpy as np
from OpenGL import GL

from ssv_engine.graphics.matrices import infinite_perspective, translation, scale
from ssv_engine.scene.primitives import cube_vertices, plane_vertices, sphere_vertices
from ssv_engine.scene.mesh import cube_mesh
from ssv_engine.scene.materials import make_material_texture
from ssv_engine.scene.sundial import sundial_mesh, sundial_render_vertices
from ssv_engine.shadows.volume import build_shadow_volume
from ssv_engine.metrics.gpu_timer import GpuTimer
from ssv_engine.scene.complexity import CUBE_TRIANGLES, MAX_COMPLEXITY_LEVEL, complexity_objects

SHADER_DIR = Path(__file__).resolve().parents[1] / "shaders"


class BasicScene:
    def __init__(self, context: moderngl.Context):
        vertex_shader = (SHADER_DIR / "basic.vert").read_text(encoding="utf-8")
        fragment_shader = (SHADER_DIR / "basic.frag").read_text(encoding="utf-8")
        self.program = context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.cube_buffer = context.buffer(cube_vertices().tobytes())
        self.floor_buffer = context.buffer(plane_vertices(48.0, 24.0).tobytes())
        self.cube_vao = context.vertex_array(self.program, [(self.cube_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")])
        sun_vertex = (SHADER_DIR / "sun.vert").read_text(encoding="utf-8")
        sun_fragment = (SHADER_DIR / "sun.frag").read_text(encoding="utf-8")
        self.sun_program = context.program(vertex_shader=sun_vertex, fragment_shader=sun_fragment)
        sun_quad = np.array([(-1,-1), (1,-1), (1,1), (-1,-1), (1,1), (-1,1)], dtype="f4")
        self.sun_buffer = context.buffer(sun_quad.tobytes())
        self.sun_vao = context.vertex_array(self.sun_program, [(self.sun_buffer, "2f", "in_corner")])
        self.floor_vao = context.vertex_array(self.program, [(self.floor_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")])
        self.floor_texture = make_material_texture(context, "floor", 512)
        self.stone_texture = make_material_texture(context, "stone", 512)
        self.wood_texture = make_material_texture(context, "wood", 512)
        self.sundial_mesh = sundial_mesh()
        self.sundial_buffer = context.buffer(sundial_render_vertices(self.sundial_mesh).tobytes())
        self.sundial_vao = context.vertex_array(self.program, [(self.sundial_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")])
        self.sundial_model = np.identity(4, dtype=np.float32)
        # A luz permanece fixa: o operador controla sua posição para manter
        # condições reproduzíveis nos experimentos com B (sombras ON/OFF).
        self.sun_orbit_enabled = False
        self.sun_orbit_time = 0.0
        self.crate_texture = make_material_texture(context, "crate")
        sky_vertex = (SHADER_DIR / "sky.vert").read_text(encoding="utf-8")
        sky_fragment = (SHADER_DIR / "sky.frag").read_text(encoding="utf-8")
        self.sky_program = context.program(vertex_shader=sky_vertex, fragment_shader=sky_fragment)
        self.sky_vao = context.vertex_array(self.sky_program, [])
        self.light_position = np.array([0.0, 4.8, 0.0], dtype=np.float32)

        volume_vertex = (SHADER_DIR / "shadow_volume.vert").read_text(encoding="utf-8")
        volume_fragment = (SHADER_DIR / "shadow_volume.frag").read_text(encoding="utf-8")
        self.volume_program = context.program(vertex_shader=volume_vertex, fragment_shader=volume_fragment)
        overlay_vertex = (SHADER_DIR / "shadow_overlay.vert").read_text(encoding="utf-8")
        overlay_fragment = (SHADER_DIR / "shadow_overlay.frag").read_text(encoding="utf-8")
        self.overlay_program = context.program(vertex_shader=overlay_vertex, fragment_shader=overlay_fragment)
        self.overlay_vao = context.vertex_array(self.overlay_program, [])

        debug_vertex = (SHADER_DIR / "volume_debug.vert").read_text(encoding="utf-8")
        debug_fragment = (SHADER_DIR / "volume_debug.frag").read_text(encoding="utf-8")
        self.debug_program = context.program(vertex_shader=debug_vertex, fragment_shader=debug_fragment)
        self.shadow_mesh = cube_mesh()
        self.method = "Z-Fail"  # "Z-Fail" (robusto) ou "Z-Pass" (comparação)
        self.complexity_level = 0
        self._volume_stats: dict[int, tuple[int, int]] = {}
        self._sundial_stats = (0, 0)
        self.last_volume_build_ms = 0.0
        self._build_accumulator_ms = 0.0
        self.scene_gpu_timer = GpuTimer()
        self.shadow_gpu_timer = GpuTimer()
        self._sundial_triangles = len(sundial_render_vertices(self.sundial_mesh)) // 3
        self.show_volume = False
        self.shadows_enabled = True
        self._volume_resources = {}
        self._debug_resources = {}
        # Sunset garden plaza: a central paved processional path, paired lanterns,
        # an architectural portal, low benches and stylized Mediterranean trees.
        limestone = (1.00, 0.94, 0.82)
        carved = (1.08, 1.00, 0.86)
        bronze = (0.48, 0.27, 0.12)
        self.objects = [
            # Rear monumental portal / torii-like Greek garden gate.
            (translation(-2.65, 1.55, -5.35) @ scale(0.34, 3.10, 0.38), limestone),
            (translation(2.65, 1.55, -5.35) @ scale(0.34, 3.10, 0.38), limestone),
            (translation(0.0, 3.18, -5.35) @ scale(5.75, 0.38, 0.58), carved),
            (translation(0.0, 2.82, -5.35) @ scale(4.95, 0.16, 0.48), bronze),
            # Lantern posts along both sides of the path; all posts cast Z-Fail shadows.
            *[(translation(x, 0.92, z) @ scale(0.16, 1.84, 0.16), bronze)
              for x in (-3.15, 3.15) for z in (-4.1, -1.3, 1.5, 4.0)],
            *[(translation(x, 1.88, z) @ scale(0.52, 0.10, 0.52), bronze)
              for x in (-3.15, 3.15) for z in (-4.1, -1.3, 1.5, 4.0)],
            *[(translation(x, 1.56, z) @ scale(0.38, 0.48, 0.38), bronze)
              for x in (-3.15, 3.15) for z in (-4.1, -1.3, 1.5, 4.0)],
            # Long low garden borders and stone benches in the foreground.
            (translation(-4.25, 0.22, -0.1) @ scale(0.34, 0.44, 9.8), limestone),
            (translation(4.25, 0.22, -0.1) @ scale(0.34, 0.44, 9.8), limestone),
            (translation(0.0, 0.30, 4.35) @ scale(3.4, 0.48, 0.82), carved),
            (translation(0.0, 0.58, 4.35) @ scale(3.55, 0.12, 0.92), limestone),
            # Restore a restrained classical colonnade without bringing back
            # the oversized trees: three limestone columns on each side.
            *[(translation(x, 1.38, z) @ scale(0.30, 2.76, 0.30), limestone)
              for x in (-4.15, 4.15) for z in (-3.9, -0.3, 3.3)],
            *[(translation(x, 0.18, z) @ scale(0.52, 0.36, 0.52), carved)
              for x in (-4.15, 4.15) for z in (-3.9, -0.3, 3.3)],
            *[(translation(x, 2.78, z) @ scale(0.48, 0.24, 0.48), carved)
              for x in (-4.15, 4.15) for z in (-3.9, -0.3, 3.3)],
        ]
        self.base_objects = list(self.objects)
        self.lantern_glass = [translation(x, 1.60, z) @ scale(0.24, 0.30, 0.24)
                              for x in (-3.15, 3.15) for z in (-4.1, -1.3, 1.5, 4.0)]

    def update(self, delta_time: float) -> None:
        if self.sun_orbit_enabled:
            self.sun_orbit_time += min(float(delta_time), 0.05) * 0.24
            angle = self.sun_orbit_time
            self.light_position[:] = (5.2 * np.cos(angle), 5.0, 5.2 * np.sin(angle))

    def toggle_sun_orbit(self) -> bool:
        self.sun_orbit_enabled = not self.sun_orbit_enabled
        return self.sun_orbit_enabled

    def move_light(self, axis: int, amount: float) -> None:
        self.sun_orbit_enabled = False
        self.light_position[axis] += amount
        # Keep the point light above the highest scene occluder; Z-Fail
        # volumes are undefined when the point source is inside its caster.
        self.light_position[1] = max(2.45, float(self.light_position[1]))

    def reset_light(self) -> None:
        self.light_position[:] = (0.0, 4.8, 0.0)
        self.sun_orbit_enabled = False
        self.sun_orbit_time = 0.0

    def toggle_volume_debug(self) -> bool:
        self.show_volume = not self.show_volume
        return self.show_volume

    def toggle_shadows(self) -> bool:
        self.shadows_enabled = not self.shadows_enabled
        return self.shadows_enabled

    def toggle_method(self) -> str:
        self.method = "Z-Pass" if self.method == "Z-Fail" else "Z-Fail"
        return self.method

    def set_complexity(self, level: int) -> int:
        level = max(0, min(int(level), MAX_COMPLEXITY_LEVEL))
        if level == self.complexity_level:
            return level
        self.complexity_level = level
        self.objects = self.base_objects + complexity_objects(level)
        # Release cached volumes of objects that no longer exist.
        for resources in (self._volume_resources, self._debug_resources):
            for index in [i for i in resources if i >= len(self.objects)]:
                _key, vao, buffer = resources.pop(index)
                if vao is not None:
                    vao.release()
                    buffer.release()
        for index in [i for i in self._volume_stats if i >= len(self.objects)]:
            del self._volume_stats[index]
        return level

    def stats(self) -> dict[str, int]:
        """Geometry counts of what is currently drawn (for HUD and CSV)."""
        occluders = len(self.objects) + 1  # cubes + sundial
        scene_triangles = (2 + CUBE_TRIANGLES * (len(self.objects) + len(self.lantern_glass))
                           + self._sundial_triangles)
        volume_triangles = sum(t for t, _ in self._volume_stats.values()) + self._sundial_stats[0]
        silhouette_edges = sum(e for _, e in self._volume_stats.values()) + self._sundial_stats[1]
        return {
            "objects": occluders + len(self.lantern_glass) + 1,  # + lantern glass + floor
            "occluders": occluders,
            "scene_triangles": scene_triangles,
            "volume_triangles": volume_triangles if self.shadows_enabled else 0,
            "volume_triangles_total": volume_triangles,
            "silhouette_edges": silhouette_edges,
            "complexity_level": self.complexity_level,
        }

    def _resource_key(self, index: int, model: np.ndarray):
        return (index, tuple(float(v) for v in self.light_position), tuple(float(v) for v in model.reshape(-1)))

    def _get_volume_resource(self, context: moderngl.Context, index: int, model: np.ndarray):
        key = self._resource_key(index, model)
        cached = self._volume_resources.get(index)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        started = perf_counter()
        volume = build_shadow_volume(self.shadow_mesh, self.light_position, model)
        self._build_accumulator_ms += (perf_counter() - started) * 1000.0
        self._volume_stats[index] = (volume.triangle_count, volume.silhouette_edge_count)
        packed = volume.triangles.reshape(-1, 4).astype("f4")
        vao = buffer = None
        if len(packed):
            buffer = context.buffer(packed.tobytes())
            vao = context.vertex_array(self.volume_program, [(buffer, "4f", "in_position")])
        if cached is not None:
            if cached[1] is not None:
                cached[1].release()
                cached[2].release()
        self._volume_resources[index] = (key, vao, buffer)
        return vao, buffer

    def _get_debug_resource(self, context: moderngl.Context, index: int, model: np.ndarray):
        key = self._resource_key(index, model)
        cached = self._debug_resources.get(index)
        if cached is not None and cached[0] == key:
            return cached[1]
        volume = build_shadow_volume(self.shadow_mesh, self.light_position, model)
        vao = buffer = None
        if volume.triangle_count:
            triangles = volume.triangles
            lines = np.stack((triangles[:, 0], triangles[:, 1], triangles[:, 1], triangles[:, 2],
                              triangles[:, 2], triangles[:, 0]), axis=1).reshape(-1, 4)
            buffer = context.buffer(lines.astype("f4").tobytes())
            vao = context.vertex_array(self.debug_program, [(buffer, "4f", "in_position")])
        if cached is not None and cached[1] is not None:
            cached[1].release()
            cached[2].release()
        self._debug_resources[index] = (key, vao, buffer)
        return vao

    def _get_sundial_volume(self, context: moderngl.Context):
        key = (tuple(float(v) for v in self.light_position),)
        cached = getattr(self, "_sundial_volume_resource", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        started = perf_counter()
        volume = build_shadow_volume(self.sundial_mesh, self.light_position, self.sundial_model)
        self._build_accumulator_ms += (perf_counter() - started) * 1000.0
        self._sundial_stats = (volume.triangle_count, volume.silhouette_edge_count)
        packed = volume.triangles.reshape(-1, 4).astype("f4")
        buffer = context.buffer(packed.tobytes()) if len(packed) else None
        vao = context.vertex_array(self.volume_program, [(buffer, "4f", "in_position")]) if buffer else None
        if cached is not None and cached[1] is not None:
            cached[1].release(); cached[2].release()
        self._sundial_volume_resource = (key, vao, buffer)
        return vao, buffer

    def _render_z_fail(self, context: moderngl.Context, view_projection: np.ndarray) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glFrontFace(GL.GL_CCW)
        # Prevent near/far clipping from opening the otherwise closed volume.
        GL.glEnable(GL.GL_DEPTH_CLAMP)
        GL.glEnable(GL.GL_STENCIL_TEST)
        GL.glStencilMask(0xFF)
        GL.glStencilFunc(GL.GL_ALWAYS, 0, 0xFF)
        if self.method == "Z-Pass":
            # Z-Pass (Heidmann): conta as faces A FRENTE da geometria visivel.
            # Falha quando a camera esta dentro do volume ou o near plane o corta.
            GL.glStencilOpSeparate(GL.GL_FRONT, GL.GL_KEEP, GL.GL_KEEP, GL.GL_INCR_WRAP)
            GL.glStencilOpSeparate(GL.GL_BACK, GL.GL_KEEP, GL.GL_KEEP, GL.GL_DECR_WRAP)
        else:
            # Z-Fail (Carmack / Everitt-Kilgard): conta as faces ATRAS da geometria.
            GL.glStencilOpSeparate(GL.GL_FRONT, GL.GL_KEEP, GL.GL_DECR_WRAP, GL.GL_KEEP)
            GL.glStencilOpSeparate(GL.GL_BACK, GL.GL_KEEP, GL.GL_INCR_WRAP, GL.GL_KEEP)

        for index, (model, _color) in enumerate(self.objects):
            vao, _buffer = self._get_volume_resource(context, index, model)
            if vao is not None:
                self.volume_program["u_view_projection"].write(view_projection.T.astype("f4").tobytes())
                vao.render(mode=moderngl.TRIANGLES)
        sundial_vao, _sundial_buffer = self._get_sundial_volume(context)
        if sundial_vao is not None:
            self.volume_program["u_view_projection"].write(view_projection.T.astype("f4").tobytes())
            sundial_vao.render(mode=moderngl.TRIANGLES)

        # Composite a translucent dark layer only where the accumulated
        # stencil count from all occluders is non-zero.
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glStencilMask(0x00)
        GL.glStencilFunc(GL.GL_NOTEQUAL, 0, 0xFF)
        GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_KEEP)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.overlay_vao.render(mode=moderngl.TRIANGLES, vertices=3)

        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_CLAMP)
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glStencilMask(0xFF)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

    def render(self, context: moderngl.Context, width: int, height: int, camera) -> None:
        projection = infinite_perspective(48.0, width / max(height, 1), 0.1)
        view = camera.view_matrix()
        view_projection = projection @ view
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        self.sky_program["u_viewport_height"].value = float(height)
        self.sky_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.scene_gpu_timer.begin()
        self.program["u_light_position"].value = tuple(float(v) for v in self.light_position)
        self.program["u_camera_position"].value = tuple(float(v) for v in camera.position)
        self.program["u_material"].value = 0

        floor_model = np.identity(4, dtype=np.float32)
        self.floor_texture.use(location=0)
        self.program["u_roughness"].value = 0.92
        self._draw(self.floor_vao, floor_model, view_projection @ floor_model, (1.0, 0.96, 0.88))
        self.stone_texture.use(location=0)
        self.program["u_roughness"].value = 0.86
        for index, (model, color) in enumerate(self.objects):
            if index >= len(self.base_objects):
                self.crate_texture.use(location=0)
                self.program["u_roughness"].value = 0.82
            elif index in (30, 31):
                self.wood_texture.use(location=0)
                self.program["u_roughness"].value = 0.78
            else:
                self.stone_texture.use(location=0)
                self.program["u_roughness"].value = 0.90
            self._draw(self.cube_vao, model, view_projection @ model, color)
        self.stone_texture.use(location=0)
        self.program["u_roughness"].value = 0.88
        self._draw(self.sundial_vao, self.sundial_model, view_projection @ self.sundial_model, (1.28, 1.20, 1.05))
        # Warm lantern glass is emissive; this is a visual glow, not a second
        # shadow-casting light source, so the Z-Fail experiment stays controlled.
        self.program["u_emissive"].value = 0.78
        for model in self.lantern_glass:
            self._draw(self.cube_vao, model, view_projection @ model, (1.0, 0.36, 0.055))
        self.program["u_emissive"].value = 0.0
        self.stone_texture.use(location=0)
        self.program["u_roughness"].value = 0.88

        # Stencil must be cleared after opaque depth has been populated.
        GL.glClearStencil(0)
        GL.glStencilMask(0xFF)
        GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
        self.scene_gpu_timer.end()
        self._build_accumulator_ms = 0.0
        if self.shadows_enabled:
            self.shadow_gpu_timer.begin()
            self._render_z_fail(context, view_projection)
            self.shadow_gpu_timer.end()
        if self._build_accumulator_ms > 0.0:
            self.last_volume_build_ms = self._build_accumulator_ms

        if self.show_volume:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_DEPTH_CLAMP)
            for index, (model, _color) in enumerate(self.objects):
                debug_vao = self._get_debug_resource(context, index, model)
                if debug_vao is not None:
                    self.debug_program["u_view_projection"].write(view_projection.T.astype("f4").tobytes())
                    debug_vao.render(mode=moderngl.LINES)
            volume = build_shadow_volume(self.sundial_mesh, self.light_position, self.sundial_model)
            triangles = volume.triangles
            lines = np.stack((triangles[:, 0], triangles[:, 1], triangles[:, 1], triangles[:, 2], triangles[:, 2], triangles[:, 0]), axis=1).reshape(-1, 4)
            debug_buffer = context.buffer(lines.astype("f4").tobytes())
            debug_vao = context.vertex_array(self.debug_program, [(debug_buffer, "4f", "in_position")])
            self.debug_program["u_view_projection"].write(view_projection.T.astype("f4").tobytes())
            debug_vao.render(mode=moderngl.LINES)
            debug_vao.release(); debug_buffer.release()
            GL.glEnable(GL.GL_DEPTH_TEST)

        # Camera-facing sun disc with a radial corona. It is a visual marker only;
        # the point-light position used by the Z-Fail calculation is unchanged.
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        self.sun_program["u_view_projection"].write(view_projection.T.astype("f4").tobytes())
        self.sun_program["u_center"].value = tuple(float(v) for v in self.light_position)
        self.sun_program["u_camera_right"].value = tuple(float(v) for v in camera.right)
        self.sun_program["u_camera_up"].value = tuple(float(v) for v in camera.up)
        self.sun_program["u_radius"].value = 0.72
        self.sun_vao.render(mode=moderngl.TRIANGLES)
        # Small amber halos around the lantern glass add a warm evening cue.
        self.sun_program["u_radius"].value = 0.24
        for model in self.lantern_glass:
            center = (model @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))[:3]
            self.sun_program["u_center"].value = tuple(float(v) for v in center)
            self.sun_vao.render(mode=moderngl.TRIANGLES)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.program["u_emissive"].value = 0.0

    def _draw(self, vao, model, mvp, color) -> None:
        self.program["u_model"].write(model.T.astype("f4").tobytes())
        self.program["u_mvp"].write(mvp.T.astype("f4").tobytes())
        self.program["u_color"].value = color
        vao.render(mode=moderngl.TRIANGLES)

    def release(self) -> None:
        self.scene_gpu_timer.release()
        self.shadow_gpu_timer.release()
        for resources in (self._volume_resources, self._debug_resources):
            for _key, vao, buffer in resources.values():
                if vao is not None:
                    vao.release()
                    buffer.release()
        sundial_resource = getattr(self, "_sundial_volume_resource", None)
        if sundial_resource and sundial_resource[1] is not None:
            sundial_resource[1].release(); sundial_resource[2].release()
        self.sundial_vao.release()
        self.sundial_buffer.release()
        self.stone_texture.release()
        self.wood_texture.release()
        self.floor_texture.release()
        self.crate_texture.release()
        self.sky_vao.release()
        self.sky_program.release()
        self.overlay_vao.release()
        self.overlay_program.release()
        self.volume_program.release()
        self.debug_program.release()
        self.sun_vao.release()
        self.sun_buffer.release()
        self.sun_program.release()
        self.cube_vao.release()
        self.floor_vao.release()
        self.cube_buffer.release()
        self.floor_buffer.release()
        self.program.release()
