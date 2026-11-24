"""Run the source-level checks that can be performed without a GPU/display."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "src" / "ssv_engine" / "shaders"
REQUIRED_SHADERS = (
    "basic.vert", "basic.frag", "shadow_volume.vert", "shadow_volume.frag",
    "shadow_overlay.vert", "shadow_overlay.frag", "volume_debug.vert", "volume_debug.frag",
    "sky.vert", "sky.frag", "sun.vert", "sun.frag",
)


def run(label: str, command: list[str]) -> None:
    print(f"\n[CHECK] {label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    missing = [name for name in REQUIRED_SHADERS if not (SHADERS / name).is_file()]
    if missing:
        print("Missing shader files:", ", ".join(missing))
        return 1
    print(f"[OK] All {len(REQUIRED_SHADERS)} GLSL shader files are present.")

    run("Unit tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run("Python syntax", [sys.executable, "-m", "compileall", "-q", "src", "main.py", "scripts"])
    run("Export manifest (dry run)", [sys.executable, "scripts/export_project.py", "--dry-run"])

    print("\nSource-level checks passed.")
    print("GPU rendering, stencil output and visual correctness still require the Windows/OpenGL runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
