"""Read-only coverage and alignment audit for per-image DEM/DSM/DTM rasters."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from multispectral import PROJECT_ROOT, read_image
from prepare_terrain_multichannel import find_elevation_path
from terrain_factors import compute_terrain_factors, load_elevation


def audit_terrain_sources(
    dataset: Path,
    terrain_root: Path,
    *,
    splits: tuple[str, ...] = ("train", "valid"),
    pixel_size_m: float,
) -> tuple[dict, list[dict[str, object]]]:
    """Audit source coverage without resizing, generating features, or writing labels."""
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive and finite, got {pixel_size_m}")
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    split_counts: dict[str, int] = {}
    for split in splits:
        images = sorted((dataset / split / "images").glob("*.JPG"))
        split_counts[split] = len(images)
        for image_path in images:
            row: dict[str, object] = {"split": split, "stem": image_path.stem, "image": str(image_path)}
            try:
                terrain_path = find_elevation_path(terrain_root, split, image_path.stem)
                image_hw = tuple(read_image(image_path).shape[:2])
                elevation = load_elevation(terrain_path)
                row.update(
                    terrain=str(terrain_path),
                    image_height=image_hw[0],
                    image_width=image_hw[1],
                    terrain_height=elevation.shape[0],
                    terrain_width=elevation.shape[1],
                    elevation_min=float(elevation.min()),
                    elevation_max=float(elevation.max()),
                    elevation_std=float(elevation.std()),
                )
                if elevation.shape != image_hw:
                    row["status"] = "shape_mismatch"
                    row["error"] = f"terrain={elevation.shape}, image={image_hw}"
                    counts["shape_mismatch"] += 1
                else:
                    factors = compute_terrain_factors(elevation, pixel_size_m=pixel_size_m)
                    row.update(
                        status="ready",
                        slope_mean=float(factors["SLOPE"].mean()),
                        slope_p95=float(np.percentile(factors["SLOPE"], 95)),
                        tri_mean=float(factors["TRI"].mean()),
                        relief_mean=float(factors["RELIEF5"].mean()),
                    )
                    counts["ready"] += 1
            except FileNotFoundError as exc:
                row.update(status="missing", error=str(exc))
                counts["missing"] += 1
            except Exception as exc:
                row.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
                counts["invalid"] += 1
            rows.append(row)
    total = len(rows)
    summary = {
        "version": "terrain_source_audit_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dataset": str(dataset.resolve()),
        "terrain_root": str(terrain_root.resolve()),
        "splits": list(splits),
        "held_out_test_used": False,
        "pixel_size_m": float(pixel_size_m),
        "split_counts": split_counts,
        "status_counts": dict(counts),
        "total": total,
        "ready": counts["ready"],
        "coverage": float(counts["ready"] / total) if total else 0.0,
        "can_prepare_candidates": total > 0 and counts["ready"] == total,
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--terrain-root", type=Path, required=True)
    parser.add_argument("--pixel-size-m", type=float, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid"), default=("train", "valid"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "Data_v6_terrain_source_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, rows = audit_terrain_sources(
        args.dataset.resolve(),
        args.terrain_root.resolve(),
        splits=tuple(args.splits),
        pixel_size_m=args.pixel_size_m,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "items.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    return 0 if summary["can_prepare_candidates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

