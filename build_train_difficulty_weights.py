"""Build deterministic train-only sampling weights from labels and RGB scene structure."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from build_sparse_mixed_difficult_review import percentile_rank, scene_features
from multispectral import PROJECT_ROOT


DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed" / "train"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "generated" / "data_v8_train_difficulty_weights_h1.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def require_train_directory(path: Path, kind: str) -> None:
    if path.name.lower() != kind or path.parent.name.lower() != "train":
        raise ValueError(f"{kind} must be the direct {kind} directory under a train split: {path}")
    lowered_parts = {part.lower() for part in path.parts}
    if "valid" in lowered_parts or "val" in lowered_parts or "test" in lowered_parts:
        raise ValueError(f"Refusing non-train path: {path}")


def labels_sha256(label_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in label_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_weights(images_dir: Path, labels_dir: Path, strength: float) -> dict[str, object]:
    require_train_directory(images_dir, "images")
    require_train_directory(labels_dir, "labels")
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"--strength must be in (0, 1], got {strength}")

    image_paths = sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not image_paths:
        raise FileNotFoundError(f"No train images found in {images_dir}")
    if len({path.name for path in image_paths}) != len(image_paths):
        raise ValueError("Train image names must be unique")
    label_paths = [labels_dir / f"{path.stem}.txt" for path in image_paths]
    missing_labels = [path for path in label_paths if not path.is_file()]
    if missing_labels:
        raise FileNotFoundError(f"Missing {len(missing_labels)} train labels, e.g. {missing_labels[:5]}")
    extra_labels = sorted(path.stem for path in labels_dir.glob("*.txt") if path.stem not in {p.stem for p in image_paths})
    if extra_labels:
        raise ValueError(f"Found {len(extra_labels)} labels without train images, e.g. {extra_labels[:5]}")

    records: list[dict[str, object]] = []
    for image_path, label_path in zip(image_paths, label_paths, strict=True):
        records.append({"image": image_path.name, **scene_features(image_path, label_path)})

    counts = np.asarray([float(row["polygon_count"]) for row in records])
    coverages = np.asarray([float(row["foreground_coverage"]) for row in records])
    texture = np.asarray([float(row["background_luma_std"]) for row in records])
    edges = np.asarray([float(row["background_edge_density"]) for row in records])
    raw_weights: list[float] = []
    for row in records:
        low_coverage = percentile_rank(coverages, float(row["foreground_coverage"]), inverse=True)
        dense_instances = percentile_rank(counts, float(row["polygon_count"]))
        texture_rank = percentile_rank(texture, float(row["background_luma_std"]))
        edge_rank = percentile_rank(edges, float(row["background_edge_density"]))
        mixed_background = float(row["background_ratio"]) * (0.5 * texture_rank + 0.5 * edge_rank)
        difficulty = 0.40 * low_coverage + 0.30 * dense_instances + 0.30 * mixed_background
        raw_weights.append(1.0 + strength * difficulty)
        row.update(
            low_coverage_score=low_coverage,
            dense_instance_score=dense_instances,
            mixed_background_score=mixed_background,
            difficulty_score=difficulty,
        )

    weights = np.asarray(raw_weights, dtype=np.float64)
    weights /= weights.mean()
    effective_sample_size = float(weights.sum() ** 2 / np.square(weights).sum())
    weights_by_image = {str(row["image"]): round(float(weight), 8) for row, weight in zip(records, weights, strict=True)}
    ranked = sorted(
        (
            {
                "image": row["image"],
                "weight": weights_by_image[str(row["image"])],
                "polygon_count": int(row["polygon_count"]),
                "foreground_coverage": round(float(row["foreground_coverage"]), 8),
                "background_luma_std": round(float(row["background_luma_std"]), 8),
                "background_edge_density": round(float(row["background_edge_density"]), 8),
                "low_coverage_score": round(float(row["low_coverage_score"]), 8),
                "dense_instance_score": round(float(row["dense_instance_score"]), 8),
                "mixed_background_score": round(float(row["mixed_background_score"]), 8),
                "difficulty_score": round(float(row["difficulty_score"]), 8),
            }
            for row in records
        ),
        key=lambda row: (-float(row["weight"]), str(row["image"])),
    )
    return {
        "version": "data_v8_train_difficulty_weights_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_split": "train",
        "selection_splits": ["train"],
        "held_out_split": "test",
        "held_out_split_used": False,
        "validation_split_used": False,
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "train_images": len(image_paths),
        "train_labels_sha256": labels_sha256(label_paths),
        "sampling": {
            "replacement": True,
            "samples_per_epoch": len(image_paths),
            "strength": strength,
            "score": "0.40*low_coverage + 0.30*dense_instances + 0.30*mixed_background",
            "normalization": "mean_weight_1",
            "min_weight": float(weights.min()),
            "mean_weight": float(weights.mean()),
            "max_weight": float(weights.max()),
            "max_to_min_ratio": float(weights.max() / weights.min()),
            "effective_sample_size": effective_sample_size,
        },
        "weights_by_image": weights_by_image,
        "ranked_train_scenes": ranked,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_TRAIN_ROOT / "images")
    parser.add_argument("--labels", type=Path, default=DEFAULT_TRAIN_ROOT / "labels")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strength", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_weights(args.images.resolve(), args.labels.resolve(), args.strength)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sampling = payload["sampling"]
    print(f"train_images={payload['train_images']}")
    print(
        f"weights=min:{sampling['min_weight']:.4f} mean:{sampling['mean_weight']:.4f} "
        f"max:{sampling['max_weight']:.4f} ratio:{sampling['max_to_min_ratio']:.4f}"
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
