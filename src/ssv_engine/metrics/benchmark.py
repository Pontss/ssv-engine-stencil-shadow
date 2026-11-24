"""Frame-by-frame benchmark capture for reproducible shadow comparisons."""

from __future__ import annotations

import csv
from datetime import datetime
from math import ceil
from pathlib import Path
from time import perf_counter


def summarize_frame_times(samples: list[float]) -> dict[str, float | int]:
    """Mean/P95/1%-low statistics for a list of frame times in milliseconds."""
    if not samples:
        return {"frames": 0, "average_fps": 0.0, "average_frame_time_ms": 0.0,
                "p95_frame_time_ms": 0.0, "p99_frame_time_ms": 0.0, "low_1_fps": 0.0,
                "min_frame_time_ms": 0.0, "max_frame_time_ms": 0.0}
    ordered = sorted(samples)
    mean_ms = sum(samples) / len(samples)
    p95 = ordered[min(len(ordered) - 1, ceil(0.95 * len(ordered)) - 1)]
    p99 = ordered[min(len(ordered) - 1, ceil(0.99 * len(ordered)) - 1)]
    worst = ordered[int(len(ordered) * 0.99):] or ordered[-1:]
    worst_mean = sum(worst) / len(worst)
    return {
        "frames": len(samples),
        "average_fps": 1000.0 / mean_ms if mean_ms else 0.0,
        "average_frame_time_ms": mean_ms,
        "p95_frame_time_ms": p95,
        "p99_frame_time_ms": p99,
        "low_1_fps": 1000.0 / worst_mean if worst_mean else 0.0,
        "min_frame_time_ms": ordered[0],
        "max_frame_time_ms": ordered[-1],
    }


class ComparisonBenchmark:
    """Automatic paired shadow ON/OFF comparison with warmup.

    By default it measures the current scene for 35 seconds per condition.
    `start(levels=[...])` turns it into a geometric-complexity sweep: the ON/OFF
    pair is repeated for every complexity level (shorter phases by default).
    """

    WARMUP_SECONDS = 3.0
    MEASURE_SECONDS = 35.0
    SWEEP_WARMUP_SECONDS = 2.0
    SWEEP_MEASURE_SECONDS = 8.0
    PHASES = ("warmup_on", "measure_on", "warmup_off", "measure_off")

    def __init__(self) -> None:
        self.active = False
        self.phase = "idle"
        self.phase_elapsed = 0.0
        self.rows: list[dict[str, float | int | bool | str]] = []
        self.results: dict[str, dict[str, float | int]] = {}
        self.results_by_level: dict[int, dict[str, dict[str, float | int]]] = {}
        self.scene_info_by_level: dict[int, dict[str, int]] = {}
        self.output_path: Path | None = None
        self.result_label = ""
        self.levels: list[int] = [0]
        self.level_index = 0
        self.sweep = False
        self.warmup_seconds = self.WARMUP_SECONDS
        self.measure_seconds = self.MEASURE_SECONDS
        self.total_elapsed = 0.0

    # ------------------------------------------------------------------ state
    @property
    def recording(self) -> bool:
        return self.active and self.phase.startswith("measure_")

    @property
    def shadows_enabled(self) -> bool:
        return self.phase in ("warmup_on", "measure_on")

    @property
    def current_level(self) -> int:
        return self.levels[min(self.level_index, len(self.levels) - 1)]

    @property
    def phase_duration(self) -> float:
        return self.measure_seconds if self.phase.startswith("measure_") else self.warmup_seconds

    @property
    def total_duration(self) -> float:
        return len(self.levels) * 2 * (self.warmup_seconds + self.measure_seconds)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.total_duration - self.total_elapsed)

    @property
    def phase_label(self) -> str:
        label = {
            "warmup_on": "Aquecendo — sombras LIGADAS",
            "measure_on": "Medindo — sombras LIGADAS",
            "warmup_off": "Aquecendo — sombras DESLIGADAS",
            "measure_off": "Medindo — sombras DESLIGADAS",
            "complete": "Comparação concluída",
        }.get(self.phase, "Pronto")
        if self.sweep and self.active:
            label += f" — nível {self.current_level} ({self.level_index + 1}/{len(self.levels)})"
        return label

    @property
    def progress(self) -> float:
        """Progress of the current phase (0..1)."""
        if not self.active:
            return 1.0 if self.phase == "complete" else 0.0
        return min(1.0, self.phase_elapsed / self.phase_duration)

    @property
    def total_progress(self) -> float:
        if not self.active:
            return 1.0 if self.phase == "complete" else 0.0
        return min(1.0, self.total_elapsed / self.total_duration)

    def live_samples(self) -> list[float]:
        """Frame times (ms) collected so far in the current measured phase."""
        if not self.recording:
            return []
        condition = "ON" if self.phase == "measure_on" else "OFF"
        level = self.current_level
        return [float(r["frame_time_ms"]) for r in self.rows
                if r["condition"] == condition and r["level"] == level]

    # ---------------------------------------------------------------- control
    def start(self, levels: list[int] | None = None,
              warmup_seconds: float | None = None,
              measure_seconds: float | None = None,
              current_level: int = 0) -> None:
        if self.active:
            raise RuntimeError("A comparison benchmark is already active")
        self.sweep = levels is not None and len(levels) > 1
        self.levels = list(levels) if levels else [int(current_level)]
        default_warmup = self.SWEEP_WARMUP_SECONDS if self.sweep else self.WARMUP_SECONDS
        default_measure = self.SWEEP_MEASURE_SECONDS if self.sweep else self.MEASURE_SECONDS
        self.warmup_seconds = float(warmup_seconds if warmup_seconds is not None else default_warmup)
        self.measure_seconds = float(measure_seconds if measure_seconds is not None else default_measure)
        self.level_index = 0
        self.active = True
        self.phase = "warmup_on"
        self.phase_elapsed = 0.0
        self.total_elapsed = 0.0
        self.rows.clear()
        self.results.clear()
        self.results_by_level.clear()
        self.scene_info_by_level.clear()
        self.output_path = None
        self.result_label = ""

    def cancel(self) -> None:
        self.active = False
        self.phase = "idle"
        self.phase_elapsed = 0.0
        self.rows.clear()

    def update(self, delta_time: float) -> bool:
        """Advance the state machine; return True when the run has completed."""
        if not self.active:
            return False
        step = max(0.0, min(float(delta_time), 0.1))
        self.phase_elapsed += step
        self.total_elapsed += step
        if self.phase_elapsed < self.phase_duration:
            return False
        transitions = {
            "warmup_on": "measure_on",
            "measure_on": "warmup_off",
            "warmup_off": "measure_off",
            "measure_off": "complete",
        }
        self.phase = transitions[self.phase]
        self.phase_elapsed = 0.0
        if self.phase == "complete" and self.level_index + 1 < len(self.levels):
            self.level_index += 1
            self.phase = "warmup_on"
        if self.phase == "complete":
            self.active = False
            self._summarize()
            return True
        return False

    def record(self, delta_time: float, width: int, height: int, light_position, camera_position,
               scene_info: dict[str, int] | None = None) -> None:
        if not self.recording or delta_time <= 0:
            return
        light = tuple(float(v) for v in light_position)
        camera = tuple(float(v) for v in camera_position)
        info = scene_info or {}
        level = self.current_level
        if info:
            self.scene_info_by_level.setdefault(level, dict(info))
        self.rows.append({
            "level": level,
            "condition": "ON" if self.phase == "measure_on" else "OFF",
            "frame_time_ms": delta_time * 1000.0,
            "fps_instant": 1.0 / delta_time,
            "width": int(width), "height": int(height),
            "objects": int(info.get("objects", 0)),
            "scene_triangles": int(info.get("scene_triangles", 0)),
            "volume_triangles": int(info.get("volume_triangles", 0)),
            "light_x": light[0], "light_y": light[1], "light_z": light[2],
            "camera_x": camera[0], "camera_y": camera[1], "camera_z": camera[2],
        })

    # ---------------------------------------------------------------- results
    @staticmethod
    def cost_percent(on_fps: float, off_fps: float) -> float:
        return ((off_fps - on_fps) / off_fps * 100.0) if off_fps > 0 else 0.0

    def _summarize(self) -> None:
        for level in self.levels:
            per_level = {}
            for condition in ("ON", "OFF"):
                samples = [float(row["frame_time_ms"]) for row in self.rows
                           if row["condition"] == condition and row.get("level", level) == level]
                per_level[condition] = summarize_frame_times(samples)
            self.results_by_level[level] = per_level
        self.results = self.results_by_level[self.levels[0]]
        on = float(self.results["ON"]["average_fps"])
        off = float(self.results["OFF"]["average_fps"])
        if self.sweep:
            last = self.results_by_level[self.levels[-1]]
            last_cost = self.cost_percent(float(last["ON"]["average_fps"]), float(last["OFF"]["average_fps"]))
            self.result_label = (f"Varredura {len(self.levels)} níveis | nível {self.levels[0]}: "
                                 f"custo {self.cost_percent(on, off):.1f}% | nível {self.levels[-1]}: custo {last_cost:.1f}%")
        else:
            self.result_label = f"ON {on:.1f} FPS | OFF {off:.1f} FPS | custo {self.cost_percent(on, off):.1f}%"

    def save(self, output_directory: str | Path | None = None, machine_info: dict[str, str] | None = None) -> Path:
        if self.active or self.phase != "complete":
            raise RuntimeError("Comparison benchmark is not complete")
        output = (Path(output_directory) if output_directory is not None
                  else Path(__file__).resolve().parents[3] / "results" / "benchmarks")
        output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = "ssv_sweep" if self.sweep else "ssv_comparison"
        path = output / f"{prefix}_{timestamp}.csv"
        fields = ["level", "condition", "frame_time_ms", "fps_instant", "width", "height",
                  "objects", "scene_triangles", "volume_triangles",
                  "light_x", "light_y", "light_z", "camera_x", "camera_y", "camera_z"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", restval=0)
            writer.writeheader()
            writer.writerows(self.rows)
        summary_path = path.with_suffix(".txt")
        with summary_path.open("w", encoding="utf-8") as stream:
            title = "VARREDURA DE COMPLEXIDADE" if self.sweep else "COMPARAÇÃO AUTOMÁTICA"
            stream.write(f"SSV ENGINE — {title} (sombras ON x OFF)\n")
            stream.write(f"{self.measure_seconds:g} segundos medidos por condição; "
                         f"aquecimento de {self.warmup_seconds:g} segundos por condição.\n")
            for key, value in (machine_info or {}).items():
                stream.write(f"{key}: {value}\n")
            stream.write("\n")
            for level in self.levels:
                info = self.scene_info_by_level.get(level, {})
                if self.sweep or info:
                    stream.write(f"Nível {level}: {info.get('objects', 0)} objetos | "
                                 f"{info.get('scene_triangles', 0)} triângulos de cena | "
                                 f"{info.get('volume_triangles', 0)} triângulos de volume\n")
                for condition, label in (("ON", "Sombras ligadas"), ("OFF", "Sombras desligadas")):
                    result = self.results_by_level[level][condition]
                    stream.write(f"  {label}: {result['average_fps']:.2f} FPS | {result['average_frame_time_ms']:.3f} ms | "
                                 f"P95 {result['p95_frame_time_ms']:.3f} ms | 1% low {result['low_1_fps']:.1f} FPS | "
                                 f"{result['frames']} frames\n")
                on = self.results_by_level[level]
                stream.write(f"  Custo das sombras: {self.cost_percent(float(on['ON']['average_fps']), float(on['OFF']['average_fps'])):.1f}%\n\n")
            stream.write(f"{self.result_label}\n")
        self.output_path = path
        return path


class BenchmarkRecorder:
    def __init__(self) -> None:
        self.active = False
        self._started_at = 0.0
        self._rows: list[dict[str, float | int | bool]] = []

    def start(self) -> None:
        if self.active:
            raise RuntimeError("A benchmark recording is already active")
        self._rows.clear()
        self._started_at = perf_counter()
        self.active = True

    def record(
        self,
        delta_time: float,
        shadows_enabled: bool,
        width: int,
        height: int,
        light_position,
        camera_position,
    ) -> None:
        if not self.active or delta_time <= 0:
            return
        light = tuple(float(v) for v in light_position)
        camera = tuple(float(v) for v in camera_position)
        self._rows.append({
            "frame": len(self._rows) + 1,
            "elapsed_seconds": perf_counter() - self._started_at,
            "frame_time_ms": delta_time * 1000.0,
            "fps_instant": 1.0 / delta_time,
            "shadows_enabled": bool(shadows_enabled),
            "width": int(width),
            "height": int(height),
            "light_x": light[0],
            "light_y": light[1],
            "light_z": light[2],
            "camera_x": camera[0],
            "camera_y": camera[1],
            "camera_z": camera[2],
        })

    def stop(self, output_directory: str | Path | None = None) -> tuple[Path, dict[str, float | int]]:
        if not self.active:
            raise RuntimeError("No benchmark recording is active")
        self.active = False
        if not self._rows:
            raise RuntimeError("Benchmark recording contains no frame samples")

        output = (Path(output_directory) if output_directory is not None
                  else Path(__file__).resolve().parents[3] / "results" / "benchmarks")
        output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = output / f"ssv_benchmark_{timestamp}.csv"
        fields = list(self._rows[0])
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._rows)

        frame_times = sorted(float(row["frame_time_ms"]) for row in self._rows)
        mean_ms = sum(frame_times) / len(frame_times)
        p95_index = min(len(frame_times) - 1, max(0, ceil(0.95 * len(frame_times)) - 1))
        summary = {
            "frames": len(frame_times),
            "duration_seconds": float(self._rows[-1]["elapsed_seconds"]),
            "average_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "average_frame_time_ms": mean_ms,
            "p95_frame_time_ms": frame_times[p95_index],
        }
        self._rows.clear()
        return path, summary
