"""Generate or repair normalized [R, G, B, NDVI, NDRE] NPY files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from multispectral import DEFAULT_MULTISPECTRAL_DIR, ensure_5ch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_MULTISPECTRAL_DIR, help="Directory containing JPG files")
    parser.add_argument("--tif-dir", type=Path, default=DEFAULT_MULTISPECTRAL_DIR, help="Directory containing TIF bands")
    parser.add_argument("--output", type=Path, default=None, help="NPY output directory; defaults to --source")
    parser.add_argument("--repair", action="store_true", help="Validate existing files and rebuild invalid ones")
    parser.add_argument("--force", action="store_true", help="Rebuild every NPY file")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images")
    return parser.parse_args()


def process_directory(
    source: Path,
    tif_dir: Path,
    output: Path,
    *,
    repair: bool,
    force: bool,
    limit: int = 0,
) -> Counter:
    images = sorted(path for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})
    if limit > 0:
        images = images[:limit]
    counts: Counter = Counter()
    failures: list[tuple[Path, str]] = []

    for index, jpg_path in enumerate(images, 1):
        npy_path = output / f"{jpg_path.stem}_5ch.npy"
        try:
            _, status = ensure_5ch(
                jpg_path,
                tif_dir,
                npy_path,
                force=force,
                repair=repair or force,
            )
            counts[status] += 1
        except Exception as exc:
            counts["failed"] += 1
            failures.append((jpg_path, str(exc)))
        if index % 25 == 0 or index == len(images):
            print(f"[5ch] {index}/{len(images)} {dict(counts)}")

    if failures:
        print("\nFailures:")
        for path, error in failures[:20]:
            print(f"  {path.name}: {error}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    return counts


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    tif_dir = args.tif_dir.resolve()
    output = (args.output or args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if not tif_dir.is_dir():
        raise SystemExit(f"TIF directory does not exist: {tif_dir}")
    output.mkdir(parents=True, exist_ok=True)

    counts = process_directory(
        source,
        tif_dir,
        output,
        repair=args.repair,
        force=args.force,
        limit=args.limit,
    )
    print(f"\nCompleted: {dict(counts)}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
