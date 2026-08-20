"""Build train/valid multichannel candidates from multispectral data and aligned DEM/DSM rasters."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from multispectral import DEFAULT_MULTISPECTRAL_DIR, PROJECT_ROOT, build_multichannel, read_image, save_multichannel_uint8
from terrain_factors import TERRAIN_CHANNELS, compute_terrain_factors, load_elevation


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_v6_terrain_channels.json"


def find_elevation_path(terrain_root: Path, split: str, stem: str) -> Path:
    candidates = [terrain_root / split / f"{stem}{suffix}" for suffix in (".npy", ".tif", ".tiff")]
    candidates.extend(terrain_root / f"{stem}{suffix}" for suffix in (".npy", ".tif", ".tiff"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing aligned elevation for {split}/{stem}; expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--tif-dir", type=Path, default=DEFAULT_MULTISPECTRAL_DIR)
    parser.add_argument("--terrain-root", type=Path, required=True, help="Aligned elevation rasters named by image stem.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v6_terrain_candidates")
    parser.add_argument("--splits", nargs="+", choices=("train", "valid"), default=("train", "valid"))
    parser.add_argument("--pixel-size-m", type=float, default=None, help="Ground sampling distance; required unless set in config.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _valid_existing(path: Path, expected_hw: tuple[int, int], channels: int) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return array.dtype == np.uint8 and array.shape == (*expected_hw, channels)
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    channels = tuple(config["channels"])
    terrain_channels = tuple(name for name in channels if name in TERRAIN_CHANNELS)
    spectral_channels = tuple(name for name in channels if name not in TERRAIN_CHANNELS)
    if channels[:3] != ("R", "G", "B"):
        raise SystemExit("Terrain candidate channels must start with R,G,B")
    if not terrain_channels:
        raise SystemExit("Config contains no terrain channels")
    terrain_config = config.get("terrain", {})
    pixel_size_m = args.pixel_size_m if args.pixel_size_m is not None else terrain_config.get("pixel_size_m")
    if pixel_size_m is None:
        raise SystemExit("Ground sampling distance is unknown; pass --pixel-size-m or set terrain.pixel_size_m")
    if float(pixel_size_m) <= 0:
        raise SystemExit("--pixel-size-m must be positive")
    scales = terrain_config.get("encoding_scales_m", {})

    dataset = args.dataset.resolve()
    terrain_root = args.terrain_root.resolve()
    output = args.output.resolve()
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    sums = np.zeros(len(channels), dtype=np.float64)
    sums_sq = np.zeros(len(channels), dtype=np.float64)
    minima = np.full(len(channels), 255, dtype=np.uint8)
    maxima = np.zeros(len(channels), dtype=np.uint8)
    pixels = 0
    split_counts: dict[str, int] = {}

    for split in args.splits:
        images = sorted((dataset / split / "images").glob("*.JPG"))
        if args.limit > 0:
            images = images[: args.limit]
        split_counts[split] = len(images)
        feature_dir = output / split / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(images, 1):
            target = feature_dir / f"{image_path.stem}.npy"
            try:
                image = read_image(image_path)
                expected_hw = image.shape[:2]
                existed = target.is_file()
                if not args.force and _valid_existing(target, expected_hw, len(channels)):
                    stored = np.load(target, allow_pickle=False)
                    counts["valid"] += 1
                else:
                    elevation_path = find_elevation_path(terrain_root, split, image_path.stem)
                    elevation = load_elevation(elevation_path)
                    if elevation.shape != expected_hw:
                        raise ValueError(
                            f"Elevation must already be pixel-aligned with the JPG; "
                            f"got {elevation.shape}, expected {expected_hw}: {elevation_path}"
                        )
                    factors = compute_terrain_factors(
                        elevation,
                        pixel_size_m=float(pixel_size_m),
                        tri_scale_m=float(scales.get("TRI", 2.0)),
                        tpi_scale_m=float(scales.get("TPI", 2.0)),
                        relief_scale_m=float(scales.get("RELIEF5", 4.0)),
                    )
                    spectral = build_multichannel(image_path, spectral_channels, args.tif_dir)
                    by_name = {name: spectral[..., position] for position, name in enumerate(spectral_channels)}
                    by_name.update(factors)
                    combined = np.dstack([by_name[name] for name in channels]).astype(np.float32)
                    save_multichannel_uint8(target, combined, channels)
                    stored = np.load(target, allow_pickle=False)
                    counts["repaired" if existed else "generated"] += 1
                for channel_index in range(len(channels)):
                    values = stored[..., channel_index]
                    sums[channel_index] += values.sum(dtype=np.float64)
                    sums_sq[channel_index] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
                minima = np.minimum(minima, stored.min(axis=(0, 1)))
                maxima = np.maximum(maxima, stored.max(axis=(0, 1)))
                pixels += stored.shape[0] * stored.shape[1]
            except Exception as exc:
                counts["failed"] += 1
                failures.append({"split": split, "image": image_path.name, "error": str(exc)})
            if index % 25 == 0 or index == len(images):
                print(f"[{split}] {index}/{len(images)} {dict(counts)}")

    means = sums / max(pixels, 1)
    variances = np.maximum(sums_sq / max(pixels, 1) - means * means, 0.0)
    manifest = {
        "version": config["version"],
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_dataset": str(dataset),
        "multispectral_tif_dir": str(args.tif_dir.resolve()),
        "terrain_root": str(terrain_root),
        "pixel_size_m": float(pixel_size_m),
        "channels": list(channels),
        "terrain_channels": list(terrain_channels),
        "selection_split": list(args.splits),
        "held_out_split": "test",
        "held_out_split_generated": False,
        "storage": config["storage"],
        "feature_policy": config.get("feature_policy", {}),
        "split_counts": split_counts,
        "status_counts": dict(counts),
        "pixels": pixels,
        "channel_stats_encoded_0_255": [
            {
                "name": name,
                "min": int(minima[i]),
                "max": int(maxima[i]),
                "mean": float(means[i]),
                "std": float(np.sqrt(variances[i])),
            }
            for i, name in enumerate(channels)
        ],
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Manifest: {manifest_path}")
    print(f"Completed: {dict(counts)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
