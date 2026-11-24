"""On-screen benchmark panel (toggled with P).

The panel is drawn on the CPU with Pillow a few times per second, uploaded
as an RGBA texture and composited over the scene with a single textured quad.
Redrawing at a fixed low rate keeps the overhead small and identical for the
shadow ON and OFF conditions, so it does not bias the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ssv_engine.metrics.benchmark import ComparisonBenchmark, summarize_frame_times

PANEL_WIDTH = 540
PANEL_MAX_HEIGHT = 1100
REDRAW_INTERVAL = 0.20

BG = (9, 12, 17, 222)
LINE = (255, 255, 255, 28)
TEXT = (226, 232, 240, 255)
DIM = (140, 152, 168, 255)
ACCENT = (242, 158, 56, 255)      # sombras ON
COOL = (96, 170, 255, 255)        # sombras OFF
GOOD = (110, 214, 140, 255)
BAD = (240, 110, 100, 255)

FONT_CANDIDATES = (
    "consola.ttf", "C:/Windows/Fonts/consola.ttf",
    "DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "Menlo.ttc", "/System/Library/Fonts/Menlo.ttc",
)
BOLD_CANDIDATES = (
    "consolab.ttf", "C:/Windows/Fonts/consolab.ttf",
    "DejaVuSansMono-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
)


def _load_font(candidates, size: int):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


VERTEX_SHADER = """
#version 330 core
uniform vec4 u_rect;   // x0, y0, x1, y1 in NDC
uniform vec2 u_uv_max;
out vec2 v_uv;
void main() {
    vec2 corner = vec2((gl_VertexID == 1 || gl_VertexID == 2 || gl_VertexID == 4) ? 1.0 : 0.0,
                       (gl_VertexID == 2 || gl_VertexID == 4 || gl_VertexID == 5) ? 1.0 : 0.0);
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, corner), 0.0, 1.0);
    v_uv = vec2(corner.x * u_uv_max.x, (1.0 - corner.y) * u_uv_max.y);
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = texture(u_texture, v_uv);
}
"""


@dataclass
class HudState:
    """Everything the panel shows; filled by the application every frame."""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    history_ms: list[float] = field(default_factory=list)
    history_shadow: list[bool] = field(default_factory=list)
    shadows_enabled: bool = True
    method: str = "Z-Fail"
    stats: dict = field(default_factory=dict)
    gpu_scene_ms: float = 0.0
    gpu_shadow_ms: float = 0.0
    gpu_timer_available: bool = False
    volume_build_ms: float = 0.0
    width: int = 0
    height: int = 0
    renderer_name: str = ""
    gl_version: str = ""
    last_report: str = ""


class BenchmarkHud:
    def __init__(self, context: moderngl.Context) -> None:
        self.visible = False
        self.font = _load_font(FONT_CANDIDATES, 14)
        self.font_small = _load_font(FONT_CANDIDATES, 12)
        self.font_bold = _load_font(BOLD_CANDIDATES, 15)
        self.font_big = _load_font(BOLD_CANDIDATES, 34)
        self.program = context.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.vao = context.vertex_array(self.program, [])
        self.texture = context.texture((PANEL_WIDTH, PANEL_MAX_HEIGHT), 4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._used_height = 1
        self._last_redraw = -1.0

    def toggle(self) -> bool:
        self.visible = not self.visible
        self._last_redraw = -1.0
        return self.visible

    def show(self) -> None:
        if not self.visible:
            self.toggle()

    # -------------------------------------------------------------- drawing
    def _text(self, draw, xy, text, fill=TEXT, font=None):
        draw.text(xy, text, fill=fill, font=font or self.font)

    def _section(self, draw, y, title):
        draw.line((16, y, PANEL_WIDTH - 16, y), fill=LINE, width=1)
        self._text(draw, (16, y + 8), title, DIM, self.font_small)
        return y + 28

    def _bar(self, draw, x, y, w, h, fraction, color):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=(255, 255, 255, 20))
        filled = int(w * max(0.0, min(1.0, fraction)))
        if filled > 2:
            draw.rounded_rectangle((x, y, x + filled, y + h), radius=h // 2, fill=color)

    def _graph(self, draw, y, state: HudState) -> int:
        x0, x1, height = 16, PANEL_WIDTH - 16, 96
        draw.rectangle((x0, y, x1, y + height), fill=(255, 255, 255, 10))
        history = state.history_ms[-(x1 - x0):]
        if history:
            ordered = sorted(history)
            top = max(16.7, ordered[int(len(ordered) * 0.98) - 1 if len(ordered) > 1 else 0] * 1.25)
            for ms, label in ((8.33, "120"), (16.67, "60"), (33.3, "30")):
                if ms < top:
                    gy = y + height - int(ms / top * height)
                    draw.line((x0, gy, x1, gy), fill=(255, 255, 255, 34))
                    self._text(draw, (x1 - 58, max(y + 2, gy - 14)), f"{label} fps", DIM, self.font_small)
            flags = state.history_shadow[-len(history):]
            start = x1 - len(history)
            for i, ms in enumerate(history):
                bar = min(height, max(1, int(ms / top * height)))
                color = ACCENT if (flags[i] if i < len(flags) else True) else COOL
                draw.line((start + i, y + height, start + i, y + height - bar), fill=color[:3] + (200,))
            self._text(draw, (x0 + 6, y + 4), f"frame time (ms)  escala {top:.1f} ms", DIM, self.font_small)
        return y + height + 8

    def _compare_rows(self, draw, y, on: dict, off: dict) -> int:
        best = max(float(on["average_fps"]), float(off["average_fps"]), 1.0)
        for label, result, color in (("ON ", on, ACCENT), ("OFF", off, COOL)):
            self._text(draw, (16, y), label, color, self.font_bold)
            self._bar(draw, 56, y + 3, 170, 11, float(result["average_fps"]) / best, color)
            self._text(draw, (236, y), f"{result['average_fps']:7.1f} FPS {result['average_frame_time_ms']:6.2f} ms")
            y += 18
            self._text(draw, (56, y), f"P95 {result['p95_frame_time_ms']:.2f} ms · 1% low {result['low_1_fps']:.0f} FPS · "
                                      f"{result['frames']} frames", DIM, self.font_small)
            y += 20
        cost = ComparisonBenchmark.cost_percent(float(on["average_fps"]), float(off["average_fps"]))
        delta_ms = float(on["average_frame_time_ms"]) - float(off["average_frame_time_ms"])
        self._text(draw, (16, y), f"Custo das sombras: {cost:.1f}%  ({delta_ms:+.2f} ms/frame)",
                   BAD if cost > 0 else GOOD, self.font_bold)
        return y + 24

    def _render_panel(self, state: HudState, benchmark: ComparisonBenchmark) -> Image.Image:
        image = Image.new("RGBA", (PANEL_WIDTH, PANEL_MAX_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        y = 14
        self._text(draw, (16, y), "SSV ENGINE · BENCHMARK", ACCENT, self.font_bold)
        self._text(draw, (PANEL_WIDTH - 90, y + 1), "[P] fechar", DIM, self.font_small)
        y += 26

        # ------------------------------------------------- live performance
        fps_color = GOOD if state.fps >= 60 else (ACCENT if state.fps >= 30 else BAD)
        self._text(draw, (16, y), f"{state.fps:.0f}", fps_color, self.font_big)
        self._text(draw, (16 + 22 * max(2, len(f"{state.fps:.0f}")), y + 16), "FPS", DIM, self.font_bold)
        window = summarize_frame_times(list(state.history_ms))
        self._text(draw, (200, y + 2), f"frame time  {state.frame_time_ms:6.2f} ms")
        self._text(draw, (200, y + 20), f"P95 {window['p95_frame_time_ms']:.2f} ms · máx {window['max_frame_time_ms']:.1f}",
                   DIM, self.font_small)
        self._text(draw, (200, y + 36), f"1% low {window['low_1_fps']:.0f} FPS (últimos {window['frames']} frames)",
                   DIM, self.font_small)
        y += 56
        y = self._graph(draw, y, state)
        if state.gpu_timer_available:
            total = state.gpu_scene_ms + (state.gpu_shadow_ms if state.shadows_enabled else 0.0)
            shadow = f"{state.gpu_shadow_ms:.3f} ms" if state.shadows_enabled else "—"
            self._text(draw, (16, y), f"GPU  cena {state.gpu_scene_ms:.3f} ms · passe stencil {shadow} · total {total:.3f} ms")
        else:
            self._text(draw, (16, y), "GPU timer query indisponível neste driver", DIM)
        y += 18
        self._text(draw, (16, y), f"CPU  construção dos volumes (última): {state.volume_build_ms:.2f} ms", DIM, self.font_small)
        y += 22

        # ------------------------------------------------------------ scene
        y = self._section(draw, y, "CENA")
        stats = state.stats
        shadow_text, shadow_color = ("LIGADAS", ACCENT) if state.shadows_enabled else ("DESLIGADAS", COOL)
        self._text(draw, (16, y), "Sombras [B]:", DIM)
        self._text(draw, (130, y), shadow_text, shadow_color, self.font_bold)
        self._text(draw, (250, y), "Método [M]:", DIM)
        self._text(draw, (352, y), state.method, GOOD if state.method == "Z-Fail" else BAD, self.font_bold)
        y += 20
        self._text(draw, (16, y), f"Complexidade [+/-]: nível {stats.get('complexity_level', 0)}  ·  "
                                  f"{stats.get('objects', 0)} objetos ({stats.get('occluders', 0)} oclusores)")
        y += 18
        self._text(draw, (16, y), f"Triângulos cena: {stats.get('scene_triangles', 0):,}".replace(",", ".")
                   + f"  ·  volumes: {stats.get('volume_triangles_total', 0):,}".replace(",", "."))
        y += 18
        self._text(draw, (16, y), f"Arestas de silhueta: {stats.get('silhouette_edges', 0)}  ·  "
                                  f"{state.width}x{state.height}  ·  V-Sync OFF", DIM)
        y += 18
        if state.renderer_name:
            self._text(draw, (16, y), f"{state.renderer_name[:52]}", DIM, self.font_small)
            y += 16
        y += 4

        # -------------------------------------------------------- benchmark
        y = self._section(draw, y, "BENCHMARK  ·  [K] ON/OFF 35s   [J] varredura de complexidade")
        if benchmark.active:
            color = ACCENT if benchmark.shadows_enabled else COOL
            self._text(draw, (16, y), benchmark.phase_label, color, self.font_bold)
            y += 20
            remaining = max(0, int(benchmark.phase_duration - benchmark.phase_elapsed + 0.999))
            self._text(draw, (16, y), f"fase: {remaining}s restantes", DIM, self.font_small)
            self._bar(draw, 140, y + 3, PANEL_WIDTH - 156, 9, benchmark.progress, color)
            y += 18
            self._text(draw, (16, y), f"total: {int(benchmark.remaining_seconds + 0.999)}s", DIM, self.font_small)
            self._bar(draw, 140, y + 3, PANEL_WIDTH - 156, 9, benchmark.total_progress, (200, 200, 210, 200))
            y += 20
            live = benchmark.live_samples()
            if live:
                current = summarize_frame_times(live)
                self._text(draw, (16, y), f"parcial: {current['average_fps']:.1f} FPS · {current['average_frame_time_ms']:.2f} ms "
                                          f"· {current['frames']} frames")
                y += 18
            self._text(draw, (16, y), "Câmera e luz bloqueadas · [K]/[J] cancela", DIM, self.font_small)
            y += 20
        elif benchmark.phase == "complete" and benchmark.results:
            if benchmark.sweep:
                y = self._sweep_table(draw, y, benchmark)
            else:
                y = self._compare_rows(draw, y, benchmark.results["ON"], benchmark.results["OFF"])
            if state.last_report:
                self._text(draw, (16, y), f"Salvo: {Path(state.last_report).name} (+ .txt)", DIM, self.font_small)
                y += 18
        else:
            self._text(draw, (16, y), "Posicione câmera e luz e pressione K (comparação ON/OFF)", DIM)
            y += 18
            self._text(draw, (16, y), "ou J (níveis de complexidade 0→5, 8s por condição).", DIM)
            y += 20

        y += 8
        image = image.crop((0, 0, PANEL_WIDTH, min(PANEL_MAX_HEIGHT, y)))
        # Background panel behind everything that was drawn.
        background = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ImageDraw.Draw(background).rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=12, fill=BG,
                                                     outline=(255, 255, 255, 36))
        return Image.alpha_composite(background, image)

    def _sweep_table(self, draw, y, benchmark: ComparisonBenchmark) -> int:
        self._text(draw, (16, y), "Nív  Objs  Tri.vol   ON fps  OFF fps  custo", DIM, self.font_small)
        y += 18
        costs = []
        for level in benchmark.levels:
            result = benchmark.results_by_level[level]
            info = benchmark.scene_info_by_level.get(level, {})
            on, off = float(result["ON"]["average_fps"]), float(result["OFF"]["average_fps"])
            cost = ComparisonBenchmark.cost_percent(on, off)
            costs.append(cost)
            self._text(draw, (16, y), f"{level:>3}  {info.get('objects', 0):>4}  {info.get('volume_triangles', 0):>7}  "
                                      f"{on:>7.1f}  {off:>7.1f}  {cost:>5.1f}%")
            y += 17
        # Mini chart: shadow cost per complexity level.
        y += 4
        chart_h, x0 = 60, 16
        width = PANEL_WIDTH - 32
        draw.rectangle((x0, y, x0 + width, y + chart_h), fill=(255, 255, 255, 10))
        top = max(max(costs, default=1.0), 1.0)
        slot = width / max(1, len(costs))
        for i, cost in enumerate(costs):
            h = int(max(0.0, cost) / top * (chart_h - 16))
            bx = int(x0 + i * slot + slot * 0.2)
            draw.rectangle((bx, y + chart_h - h, int(bx + slot * 0.6), y + chart_h), fill=ACCENT)
            self._text(draw, (bx, y + chart_h - h - 14), f"{cost:.0f}%", TEXT, self.font_small)
        y += chart_h + 4
        self._text(draw, (16, y), "custo das sombras (%) por nível de complexidade", DIM, self.font_small)
        return y + 20

    # ------------------------------------------------------------- display
    def render(self, context: moderngl.Context, width: int, height: int,
               state: HudState, benchmark: ComparisonBenchmark) -> None:
        if not self.visible or width <= 0 or height <= 0:
            return
        now = perf_counter()
        if now - self._last_redraw >= REDRAW_INTERVAL:
            panel = self._render_panel(state, benchmark)
            self._used_height = panel.height
            full = Image.new("RGBA", (PANEL_WIDTH, PANEL_MAX_HEIGHT), (0, 0, 0, 0))
            full.paste(panel, (0, 0))
            self.texture.write(np.asarray(full, dtype=np.uint8).tobytes())
            self._last_redraw = now

        margin = 16
        scale = min(1.0, (height - 2 * margin) / self._used_height, (width - 2 * margin) / PANEL_WIDTH)
        panel_w, panel_h = PANEL_WIDTH * scale, self._used_height * scale
        x0 = -1.0 + 2.0 * margin / width
        y1 = 1.0 - 2.0 * margin / height
        x1 = x0 + 2.0 * panel_w / width
        y0 = y1 - 2.0 * panel_h / height
        self.program["u_rect"].value = (x0, y0, x1, y1)
        self.program["u_uv_max"].value = (1.0, self._used_height / PANEL_MAX_HEIGHT)
        self.program["u_texture"].value = 0
        self.texture.use(location=0)
        context.disable(moderngl.DEPTH_TEST)
        context.enable(moderngl.BLEND)
        context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(mode=moderngl.TRIANGLES, vertices=6)
        context.disable(moderngl.BLEND)
        context.enable(moderngl.DEPTH_TEST)

    def release(self) -> None:
        self.texture.release()
        self.vao.release()
        self.program.release()
