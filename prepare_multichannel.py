"""Generate compact manifest-ordered multichannel candidate feature arrays."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from multispectral import (
    DEFAULT_MULTISPECTRAL_DIR,
    PROJECT_ROOT,
    build_multichannel,
    read_image,
    save_multichannel_uint8,
)
from spectral_indices import registry_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--tif-dir", type=Path, default=DEFAULT_MULTISPECTRAL_DIR)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data_v4_channels.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v4_candidates")
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=("train", "valid", "test"))
    parser.add_argument("--limit", type=int, default=0, help="Per-split image limit for smoke tests.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def valid_existing(path: Path, expected_hw: tuple[int, int], channels: int) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return array.dtype == np.uint8 and array.shape == (*expected_hw, channels)
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(2)
    dataset = args.dataset.resolve()
    tif_dir = args.tif_dir.resolve()
    output = args.output.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    channels = tuple(config["channels"])
    if config.get("storage", {}).get("dtype") != "uint8":
        raise SystemExit("Candidate storage must be uint8")

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
                image = read_image(image_path, cv2.IMREAD_COLOR)
                expected_hw = image.shape[:2]
                existed = target.is_file()
                if not args.force and target.is_file() and valid_existing(target, expected_hw, len(channels)):
                    stored = np.load(target, allow_pickle=False)
                    counts["valid"] += 1
                else:
                    array = build_multichannel(image_path, channels, tif_dir)
                    save_multichannel_uint8(target, array, channels)
                    stored = np.load(target, allow_pickle=False)
                    counts["repaired" if existed else "generated"] += 1
                for channel_index in range(len(channels)):
                    channel = stored[..., channel_index]
                    sums[channel_index] += channel.sum(dtype=np.float64)
                    sums_sq[channel_index] += np.square(channel, dtype=np.float64).sum(dtype=np.float64)
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
        "tif_dir": str(tif_dir),
        "storage": config["storage"],
        "config": str(args.config.resolve()),
        "sensor": config.get("sensor"),
        "source_bands": config.get("source_bands", []),
        "selection_split": config.get("selection_split", ["train", "valid"]),
        "held_out_split": config.get("held_out_split", "test"),
        "feature_policy": config.get("feature_policy", {}),
        "channels": list(channels),
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
        "index_registry": registry_manifest(),
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
