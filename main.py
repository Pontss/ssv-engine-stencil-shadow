"""Entry point for the SSV Engine experimental renderer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ssv_engine.app import main  # noqa: E402


if __name__ == "__main__":
    main()
