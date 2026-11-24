"""SSV Engine application bootstrap and interactive scene."""

from __future__ import annotations

from collections import deque

import glfw
import moderngl
from OpenGL import GL

from ssv_engine.camera.controller import Camera
from ssv_engine.metrics.frame_timer import FrameTimer
from ssv_engine.metrics.benchmark import ComparisonBenchmark
from ssv_engine.metrics.hud import BenchmarkHud, HudState
from ssv_engine.scene.complexity import MAX_COMPLEXITY_LEVEL
from ssv_engine.scene.renderer import BasicScene

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "SSV ENGINE  //  SHADOW LAB  //  Z-FAIL"


def create_window():
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed. Check graphics drivers and display availability.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.STENCIL_BITS, 8)
    glfw.window_hint(glfw.DEPTH_BITS, 24)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 3.3 window with a stencil buffer.")

    glfw.make_context_current(window)
    # Disable V-Sync so the FPS counter exposes the real ON/OFF shadow cost.
    # Otherwise both modes can be pinned to the monitor refresh (e.g. 120 FPS).
    glfw.swap_interval(0)
    try:
        context = moderngl.create_context(require=330)
        # ModernGL exposes the actual default framebuffer bit depths.
        # This is reliable on Windows and avoids calling the OpenGL 1.1
        # opengl32.dll export directly (which can bypass the active ICD).
        framebuffer_bits = context.screen.bits
        stencil_bits = int(framebuffer_bits["stencil"])
        depth_bits = int(framebuffer_bits["depth"])
        if stencil_bits < 8:
            raise RuntimeError(f"Framebuffer has {stencil_bits} stencil bits; at least 8 are required.")
        if depth_bits < 16:
            raise RuntimeError(f"Framebuffer has only {depth_bits} depth bits.")
        print(f"OpenGL: {context.info.get('GL_VERSION', 'unknown')}")
        print(f"Stencil bits: {stencil_bits} | Depth bits: {depth_bits}")
        return window, context
    except Exception:
        glfw.destroy_window(window)
        glfw.terminate()
        raise


def main() -> None:
    window, context = create_window()
    scene = None
    context.enable(moderngl.DEPTH_TEST)
    camera = Camera()
    timer = FrameTimer()
    benchmark = ComparisonBenchmark()
    last_benchmark_phase = "idle"
    hud = None
    hud_state = HudState()
    shadow_history: deque[bool] = deque(maxlen=timer.history_ms.maxlen)
    user_complexity = {"level": 0}
    gl_info = {"renderer": str(context.info.get("GL_RENDERER", "")),
               "version": str(context.info.get("GL_VERSION", ""))}
    pressed_keys: set[int] = set()
    mouse_state = {"first": True, "x": WINDOW_WIDTH / 2, "y": WINDOW_HEIGHT / 2, "captured": True}
    window_state = {"fullscreen": False, "windowed": None}

    def toggle_fullscreen():
        if not window_state["fullscreen"]:
            window_state["windowed"] = (*glfw.get_window_pos(window), *glfw.get_window_size(window))
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
            window_state["fullscreen"] = True
            print("Fullscreen: ON (F11 / Alt+Enter)")
        else:
            x, y, width, height = window_state["windowed"]
            glfw.set_window_monitor(window, None, x, y, width, height, 0)
            window_state["fullscreen"] = False
            print("Fullscreen: OFF")

    def start_benchmark(sweep: bool) -> None:
        try:
            if sweep:
                user_complexity["level"] = scene.complexity_level
                benchmark.start(levels=list(range(MAX_COMPLEXITY_LEVEL + 1)))
            else:
                benchmark.start(current_level=scene.complexity_level)
        except RuntimeError as error:
            print(f"Benchmark could not be started: {error}")
            return
        pressed_keys.clear()
        scene.shadows_enabled = True
        hud_state.last_report = ""
        hud.show()
        if sweep:
            print(f"Varredura de complexidade iniciada: níveis 0..{MAX_COMPLEXITY_LEVEL}, "
                  f"{benchmark.warmup_seconds:g}s aquecimento + {benchmark.measure_seconds:g}s ON e OFF por nível "
                  f"(~{benchmark.total_duration:.0f}s).")
        else:
            print("Comparação automática iniciada: 3s aquecimento + 35s medição com sombras ON, depois 3s + 35s com sombras OFF.")

    def on_key(_window, key, _scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_P:
                visible = hud.toggle()
                print(f"Painel de benchmark: {'ON' if visible else 'OFF'}")
                return
            if benchmark.active:
                if key in (glfw.KEY_K, glfw.KEY_F5, glfw.KEY_J, glfw.KEY_F7):
                    benchmark.cancel()
                    scene.set_complexity(user_complexity["level"])
                    print("Benchmark cancelado.")
                    return
                if key not in (glfw.KEY_ESCAPE, glfw.KEY_F11, glfw.KEY_ENTER):
                    return
            pressed_keys.add(key)
            if key == glfw.KEY_V:
                enabled = scene.toggle_volume_debug()
                print(f"Shadow Volume wireframe debug: {'ON' if enabled else 'OFF'}")
            if key == glfw.KEY_B:
                enabled = scene.toggle_shadows()
                print(f"Z-Fail shadows: {'ON' if enabled else 'OFF'}")
            if key == glfw.KEY_M:
                method = scene.toggle_method()
                print(f"Método de contagem do stencil: {method}")
            if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                step = 1 if key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD) else -1
                level = scene.set_complexity(scene.complexity_level + step)
                stats = scene.stats()
                print(f"Complexidade: nível {level} | {stats['objects']} objetos | {stats['scene_triangles']} triângulos")
            if key == glfw.KEY_O:
                enabled = scene.toggle_sun_orbit()
                print(f"Sun orbit: {'ON' if enabled else 'OFF'}")
            if key == glfw.KEY_R:
                camera.reset()
                scene.reset_light()
                print("Camera and light reset to default positions.")
            if key == glfw.KEY_F11 or (key == glfw.KEY_ENTER and (mods & glfw.MOD_ALT)):
                toggle_fullscreen()
            if key in (glfw.KEY_F5, glfw.KEY_K):
                start_benchmark(sweep=False)
            if key in (glfw.KEY_F7, glfw.KEY_J):
                start_benchmark(sweep=True)
            if key == glfw.KEY_ESCAPE:
                mouse_state["captured"] = not mouse_state["captured"]
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED if mouse_state["captured"] else glfw.CURSOR_NORMAL)
                mouse_state["first"] = True
        elif action == glfw.RELEASE:
            pressed_keys.discard(key)

    def on_mouse(_window, xpos, ypos):
        if not mouse_state["captured"] or benchmark.active:
            return
        if mouse_state["first"]:
            mouse_state["x"], mouse_state["y"] = xpos, ypos
            mouse_state["first"] = False
            return
        x_offset = xpos - mouse_state["x"]
        y_offset = mouse_state["y"] - ypos
        mouse_state["x"], mouse_state["y"] = xpos, ypos
        camera.rotate(x_offset, y_offset)

    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, on_mouse)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    try:
        scene = BasicScene(context)
        hud = BenchmarkHud(context)
        print("SSV ENGINE // SUNDIAL ENVIRONMENT — Z-Fail stencil shadow-volume pass enabled.")
        print("WASD: move | Mouse: look | Space/Shift: up/down | Arrows/Z/X: move light | O: sun orbit | B: shadows | V: volumes | R: reset")
        print("P: painel de benchmark | M: Z-Fail/Z-Pass | +/-: complexidade | K/F5: comparação ON/OFF (35s cada) | J/F7: varredura de complexidade")
        print("V-Sync: OFF (uncapped FPS) | F11 or Alt+Enter: fullscreen | ESC: cursor")
        while not glfw.window_should_close(window):
            timer.tick()
            if benchmark.active:
                benchmark.update(timer.delta_time)
                scene.shadows_enabled = benchmark.shadows_enabled
                if benchmark.active and benchmark.sweep:
                    scene.set_complexity(benchmark.current_level)
                if benchmark.phase != last_benchmark_phase:
                    print(benchmark.phase_label)
                    last_benchmark_phase = benchmark.phase
                if benchmark.phase == "complete":
                    if benchmark.sweep:
                        scene.set_complexity(user_complexity["level"])
                    scene.shadows_enabled = True
                    width, height = glfw.get_framebuffer_size(window)
                    result_path = benchmark.save(machine_info={
                        "GPU": gl_info["renderer"], "OpenGL": gl_info["version"],
                        "Resolução": f"{width}x{height}", "Método": scene.method,
                    })
                    hud_state.last_report = str(result_path)
                    print("\n========== RESULTADO DO BENCHMARK ==========")
                    for level in benchmark.levels:
                        level_results = benchmark.results_by_level[level]
                        prefix = f"Nível {level} | " if benchmark.sweep else ""
                        print(f"{prefix}Sombras ON : {level_results['ON']['average_fps']:.2f} FPS | {level_results['ON']['average_frame_time_ms']:.3f} ms")
                        print(f"{prefix}Sombras OFF: {level_results['OFF']['average_fps']:.2f} FPS | {level_results['OFF']['average_frame_time_ms']:.3f} ms")
                    print(benchmark.result_label)
                    print(f"Relatório: {result_path} (CSV + TXT)")
                    print("============================================\n")
            else:
                camera.move(pressed_keys, timer.delta_time, glfw)
                scene.update(timer.delta_time)
                light_step = 2.0 * min(timer.delta_time, 0.05)
                if glfw.KEY_LEFT in pressed_keys:
                    scene.move_light(0, -light_step)
                if glfw.KEY_RIGHT in pressed_keys:
                    scene.move_light(0, light_step)
                if glfw.KEY_UP in pressed_keys:
                    scene.move_light(2, -light_step)
                if glfw.KEY_DOWN in pressed_keys:
                    scene.move_light(2, light_step)
                if glfw.KEY_Z in pressed_keys:
                    scene.move_light(1, light_step)
                if glfw.KEY_X in pressed_keys:
                    scene.move_light(1, -light_step)

            width, height = glfw.get_framebuffer_size(window)
            if width > 0 and height > 0:
                context.viewport = (0, 0, width, height)
                context.clear(0.055, 0.065, 0.085, 1.0, depth=1.0)
                scene.render(context, width, height, camera)
                scene_stats = scene.stats()
                benchmark.record(
                    timer.delta_time, width, height,
                    scene.light_position, camera.position, scene_stats,
                )
                shadow_history.append(scene.shadows_enabled)
                if hud.visible:
                    hud_state.fps = timer.fps
                    hud_state.frame_time_ms = timer.frame_time_ms
                    hud_state.history_ms = list(timer.history_ms)
                    hud_state.history_shadow = list(shadow_history)
                    hud_state.shadows_enabled = scene.shadows_enabled
                    hud_state.method = scene.method
                    hud_state.stats = scene_stats
                    hud_state.gpu_scene_ms = scene.scene_gpu_timer.average_ms
                    hud_state.gpu_shadow_ms = scene.shadow_gpu_timer.average_ms
                    hud_state.gpu_timer_available = scene.shadow_gpu_timer.available
                    hud_state.volume_build_ms = scene.last_volume_build_ms
                    hud_state.width, hud_state.height = width, height
                    hud_state.renderer_name = gl_info["renderer"]
                    hud.render(context, width, height, hud_state, benchmark)
                if benchmark.active and not hud.visible:
                    # Simple in-scene progress bar; phase and countdown are in the title.
                    GL.glDisable(GL.GL_DEPTH_TEST)
                    GL.glEnable(GL.GL_SCISSOR_TEST)
                    bar_x, bar_y, bar_w, bar_h = 24, 24, max(1, width - 48), 12
                    GL.glScissor(bar_x, bar_y, bar_w, bar_h)
                    GL.glClearColor(0.10, 0.12, 0.15, 1.0)
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                    fill_w = int(bar_w * benchmark.progress)
                    if fill_w > 0:
                        GL.glScissor(bar_x, bar_y, fill_w, bar_h)
                        GL.glClearColor(0.95, 0.62, 0.22, 1.0)
                        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                    GL.glDisable(GL.GL_SCISSOR_TEST)
                    GL.glEnable(GL.GL_DEPTH_TEST)
            mode_label = " | FULLSCREEN" if window_state["fullscreen"] else ""
            if benchmark.active:
                remaining = max(0, int(benchmark.phase_duration - benchmark.phase_elapsed + 0.999))
                status = f" | {benchmark.phase_label} | {remaining}s"
            elif benchmark.phase == "complete":
                status = f" | {benchmark.result_label}"
            else:
                status = ""
            glfw.set_window_title(window, f"{WINDOW_TITLE}{status}{mode_label} | {timer.fps:.0f} FPS | {timer.frame_time_ms:.2f} ms")
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        if hud is not None:
            hud.release()
        if scene is not None:
            scene.release()
        context.release()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
