"""Create a clean source archive without virtual environments or caches."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "ssv-engine-source.zip"
EXCLUDED_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", "build", "dist", "results"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_FILES = {".DS_Store", ".coverage"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list files without creating the ZIP")
    args = parser.parse_args()
    included = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name in EXCLUDED_FILES or path.name.startswith(".env"):
            continue
        included.append((path, relative))

    print(f"Files to include: {len(included)}")
    for _path, relative in sorted(included, key=lambda item: str(item[1])):
        print(f"  {relative}")
    if args.dry_run:
        print("Dry run only; no archive was created.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in sorted(included, key=lambda item: str(item[1])):
            archive.write(path, Path("ssv-engine") / relative)

    print(f"Export created: {OUTPUT}")
    print("The archive excludes .venv, caches, build artifacts and Git metadata.")


if __name__ == "__main__":
    main()
