"""Build a train-only sparse-context augmentation dataset for Data_v20.

The builder never reads validation or test labels.  It scans deterministic
windows inside the existing training images, selects locally sparse or empty
background crops, clips instance polygons to those windows, and hard-links the
unchanged source artifacts into a derived training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from multispectral import PROJECT_ROOT, read_image


SOURCE_DATASET = PROJECT_ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed"
SOURCE_FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v4_candidates"
VALID_IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "Data_v20_sparse_context_train"
OUTPUT_YAML = PROJECT_ROOT / "configs" / "generated" / "data_v20_sparse_context_5ch.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name.lower()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_polygons(path: Path, height: int, width: int) -> list[tuple[int, np.ndarray]]:
    polygons: list[tuple[int, np.ndarray]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"Invalid segmentation label at {path}:{line_number}")
        class_id = int(values[0])
        normalized = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(normalized).all() or (normalized < 0.0).any() or (normalized > 1.0).any():
            raise ValueError(f"Out-of-range segmentation label at {path}:{line_number}")
        pixels = normalized * np.asarray([width - 1, height - 1], dtype=np.float64)
        polygons.append((class_id, pixels))
    return polygons


def union_mask(polygons: list[tuple[int, np.ndarray]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, points in polygons:
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


def window_positions(length: int, size: int, stride: int) -> list[int]:
    if size > length:
        return []
    positions = list(range(0, length - size + 1, stride))
    end = length - size
    if not positions or positions[-1] != end:
        positions.append(end)
    return positions


def window_coverage(integral: np.ndarray, x: int, y: int, size: int) -> float:
    total = (
        int(integral[y + size, x + size])
        - int(integral[y, x + size])
        - int(integral[y + size, x])
        + int(integral[y, x])
    )
    return total / float(size * size)


def scan_candidates(
    images_dir: Path,
    labels_dir: Path,
    *,
    crop_size: int,
    stride: int,
    sparse_min: float,
    sparse_max: float,
    empty_max: float,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    image_paths = sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not image_paths:
        raise FileNotFoundError(f"No training images found in {images_dir}")
    numbered: list[tuple[int, str]] = []
    for image_path in image_paths:
        match = re.fullmatch(r"DJI_(\d+)", image_path.stem, flags=re.IGNORECASE)
        if match:
            numbered.append((int(match.group(1)), image_path.name))
    source_groups: dict[str, str] = {}
    current: list[tuple[int, str]] = []
    for number, name in sorted(numbered):
        if current and number - current[-1][0] > 20:
            label = f"train_sequence_{current[0][0]:04d}_{current[-1][0]:04d}"
            source_groups.update({item_name: label for _, item_name in current})
            current = []
        current.append((number, name))
    if current:
        label = f"train_sequence_{current[0][0]:04d}_{current[-1][0]:04d}"
        source_groups.update({item_name: label for _, item_name in current})
    candidates: list[dict[str, object]] = []
    full_coverages: list[float] = []
    for image_path in image_paths:
        feature_path = SOURCE_FEATURE_ROOT / "train" / "features" / f"{image_path.stem}.npy"
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not feature_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(feature_path if not feature_path.is_file() else label_path)
        feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if feature.ndim != 3 or feature.dtype != np.uint8:
            raise ValueError(f"Invalid feature array: {feature_path} {feature.shape} {feature.dtype}")
        height, width = feature.shape[:2]
        polygons = read_polygons(label_path, height, width)
        mask = union_mask(polygons, height, width)
        full_coverages.append(float(mask.mean()))
        integral = cv2.integral(mask, sdepth=cv2.CV_64F)
        per_image: list[dict[str, object]] = []
        for y in window_positions(height, crop_size, stride):
            for x in window_positions(width, crop_size, stride):
                coverage = window_coverage(integral, x, y, crop_size)
                kind = None
                if sparse_min <= coverage < sparse_max:
                    kind = "sparse_positive"
                elif coverage <= empty_max:
                    kind = "empty_background"
                if kind is not None:
                    per_image.append(
                        {
                            "source_image": image_path.name,
                            "source_stem": image_path.stem,
                            "source_group": source_groups.get(image_path.name, image_path.name),
                            "x": x,
                            "y": y,
                            "size": crop_size,
                            "coverage": coverage,
                            "kind": kind,
                        }
                    )
        # Keep only the strongest representative of each kind from one source.
        sparse = [row for row in per_image if row["kind"] == "sparse_positive"]
        empty = [row for row in per_image if row["kind"] == "empty_background"]
        if sparse:
            target = (sparse_min + sparse_max) / 2.0
            candidates.append(min(sparse, key=lambda row: (abs(float(row["coverage"]) - target), int(row["y"]), int(row["x"]))))
        if empty:
            candidates.append(min(empty, key=lambda row: (float(row["coverage"]), int(row["y"]), int(row["x"]))))
    summary: dict[str, float | int] = {
        "source_images": len(image_paths),
        "source_min_full_coverage": min(full_coverages),
        "source_mean_full_coverage": float(np.mean(full_coverages)),
        "sparse_candidate_sources": sum(row["kind"] == "sparse_positive" for row in candidates),
        "empty_candidate_sources": sum(row["kind"] == "empty_background" for row in candidates),
    }
    return candidates, summary


def choose_candidates(
    candidates: list[dict[str, object]], *, sparse_count: int, empty_count: int
) -> list[dict[str, object]]:
    sparse = sorted(
        (row for row in candidates if row["kind"] == "sparse_positive"),
        key=lambda row: (float(row["coverage"]), str(row["source_image"])),
    )
    empty = sorted(
        (row for row in candidates if row["kind"] == "empty_background"),
        key=lambda row: (float(row["coverage"]), str(row["source_image"])),
    )
    def diverse_order(pool: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in pool:
            grouped.setdefault(str(row.get("source_group", row["source_image"])), []).append(row)
        for values in grouped.values():
            values.sort(key=lambda row: (float(row["coverage"]), str(row["source_image"])))
        ordered: list[dict[str, object]] = []
        group_names = list(grouped)
        while any(grouped[name] for name in group_names):
            for name in group_names:
                if grouped[name]:
                    ordered.append(grouped[name].pop(0))
        return ordered

    selected: list[dict[str, object]] = []
    used_sources: set[str] = set()
    sparse_sources = {str(row["source_image"]) for row in sparse}
    empty = diverse_order(sorted(
        empty,
        key=lambda row: (
            str(row["source_image"]) in sparse_sources,
            float(row["coverage"]),
            str(row["source_image"]),
        ),
    ))
    sparse = diverse_order(sparse)
    # Empty-background sources are rarer, so reserve them before selecting the
    # larger sparse-positive pool.  Exclusive empty sources are preferred.
    for pool, target in ((empty, empty_count), (sparse, sparse_count)):
        added = 0
        for row in pool:
            source = str(row["source_image"])
            if source in used_sources:
                continue
            selected.append(dict(row))
            used_sources.add(source)
            added += 1
            if added >= target:
                break
        if added < target:
            raise RuntimeError(f"Only selected {added}/{target} {pool[0]['kind'] if pool else 'candidate'} crops")
    return sorted(selected, key=lambda row: (str(row["kind"]), str(row["source_image"])))


def clipped_labels(
    polygons: list[tuple[int, np.ndarray]], x: int, y: int, size: int
) -> tuple[list[str], float]:
    lines: list[str] = []
    combined = np.zeros((size, size), dtype=np.uint8)
    offset = np.asarray([x, y], dtype=np.float64)
    for class_id, points in polygons:
        instance = np.zeros((size, size), dtype=np.uint8)
        local = np.rint(points - offset).astype(np.int32)
        cv2.fillPoly(instance, [local], 1)
        contours, _ = cv2.findContours(instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 16.0:
                continue
            contour = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
            if len(contour) < 3:
                continue
            normalized = np.clip(contour.astype(np.float64) / float(size - 1), 0.0, 1.0)
            coordinates = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
            lines.append(f"{class_id} {coordinates}")
            cv2.fillPoly(combined, [contour.astype(np.int32)], 1)
    return lines, float(combined.mean())


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def save_rgb_jpeg(path: Path, image_rgb: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".JPG", cv2.cvtColor(image_rgb[..., :3], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def build_dataset(args: argparse.Namespace, selected: list[dict[str, object]], scan: dict[str, object]) -> dict[str, object]:
    output_root = args.output_root.resolve()
    output_yaml = args.output_yaml.resolve()
    datasets_root = (PROJECT_ROOT / "datasets").resolve()
    generated_configs = (PROJECT_ROOT / "configs" / "generated").resolve()
    if output_root.parent != datasets_root or output_yaml.parent != generated_configs:
        raise ValueError("Output dataset/YAML must stay in the project datasets and configs/generated directories")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    train_images = SOURCE_DATASET / "train" / "images"
    train_labels = SOURCE_DATASET / "train" / "labels"
    source_images = sorted(path for path in train_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    source_labels = [train_labels / f"{path.stem}.txt" for path in source_images]
    if any(not path.is_file() for path in source_labels):
        raise FileNotFoundError("Source train labels are incomplete")

    for image_path, label_path in zip(source_images, source_labels, strict=True):
        feature_path = SOURCE_FEATURE_ROOT / "train" / "features" / f"{image_path.stem}.npy"
        hardlink(image_path, output_root / "train" / "images" / image_path.name)
        hardlink(label_path, output_root / "train" / "labels" / label_path.name)
        hardlink(feature_path, output_root / "train" / "features" / feature_path.name)
    valid_features = sorted((SOURCE_FEATURE_ROOT / "valid" / "features").glob("*.npy"))
    for feature_path in valid_features:
        hardlink(feature_path, output_root / "valid" / "features" / feature_path.name)

    generated: list[dict[str, object]] = []
    for index, row in enumerate(selected, 1):
        stem = str(row["source_stem"])
        x, y, size = int(row["x"]), int(row["y"]), int(row["size"])
        output_stem = f"spctx_{index:03d}_{stem}_x{x}_y{y}_s{size}"
        image_path = train_images / str(row["source_image"])
        label_path = train_labels / f"{stem}.txt"
        feature_path = SOURCE_FEATURE_ROOT / "train" / "features" / f"{stem}.npy"
        image = read_image(image_path)
        feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if image.shape[:2] != feature.shape[:2]:
            raise ValueError(f"Image/feature shape mismatch for {stem}: {image.shape} vs {feature.shape}")
        crop_image = np.ascontiguousarray(image[y : y + size, x : x + size])
        crop_feature = np.ascontiguousarray(feature[y : y + size, x : x + size])
        polygons = read_polygons(label_path, feature.shape[0], feature.shape[1])
        label_lines, realized_coverage = clipped_labels(polygons, x, y, size)
        if abs(realized_coverage - float(row["coverage"])) > 0.01:
            raise RuntimeError(
                f"Clipped label coverage drift for {output_stem}: {realized_coverage} vs {row['coverage']}"
            )
        output_image = output_root / "train" / "images" / f"{output_stem}.JPG"
        output_label = output_root / "train" / "labels" / f"{output_stem}.txt"
        output_feature = output_root / "train" / "features" / f"{output_stem}.npy"
        save_rgb_jpeg(output_image, crop_image)
        output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        np.save(output_feature, crop_feature, allow_pickle=False)
        generated.append(
            {
                **row,
                "output_image": output_image.name,
                "output_labels": len(label_lines),
                "realized_coverage": realized_coverage,
                "feature_shape": list(crop_feature.shape),
            }
        )

    manifest = {
        "version": args.version,
        "created_at": now_iso(),
        "purpose": "train-only sparse-context and empty-background risk mitigation",
        "channels": [
            "R", "G", "B", "NIR", "RedEdge", "NDVI", "NDRE", "GNDVI", "SAVI",
            "OSAVI", "MSAVI2", "EVI2", "RECI", "BI", "SCI", "ExG", "RE_BSI_PROXY",
        ],
        "selection_basis_splits": ["train"],
        "selection_splits": ["train", "valid"],
        "training_evaluation_splits": ["train", "valid"],
        "validation_labels_used_for_generation": False,
        "held_out_split": "test",
        "held_out_split_used": False,
        "source_dataset": str(SOURCE_DATASET.resolve()),
        "source_feature_root": str(SOURCE_FEATURE_ROOT.resolve()),
        "source_train_images": len(source_images),
        "source_train_labels_sha256": sha256_paths(source_labels),
        "linked_valid_features": len(valid_features),
        "crop_config": {
            "crop_size": args.crop_size,
            "stride": args.stride,
            "sparse_min": args.sparse_min,
            "sparse_max": args.sparse_max,
            "empty_max": args.empty_max,
            "requested_sparse": args.sparse_count,
            "requested_empty": args.empty_count,
        },
        "scan_summary": scan,
        "generated_images": len(generated),
        "total_train_images": len(source_images) + len(generated),
        "generated": generated,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = (
        f"# {args.version} train-only sparse-context augmentation; held-out test is not exposed.\n"
        f"path: {output_root.as_posix()}\n"
        "train: train/images\n"
        f"val: {VALID_IMAGES.resolve().as_posix()}\n\n"
        "nc: 1\n"
        "names: ['land']\n"
        "channels: 5\n"
    )
    output_yaml.write_text(yaml_text, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--sparse-min", type=float, default=0.01)
    parser.add_argument("--sparse-max", type=float, default=0.15)
    parser.add_argument("--empty-max", type=float, default=0.002)
    parser.add_argument("--sparse-count", type=int, default=14)
    parser.add_argument("--empty-count", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-yaml", type=Path, default=OUTPUT_YAML)
    parser.add_argument("--version", default="data_v20_sparse_context_train_v1")
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.crop_size < 256 or args.stride < 1:
        raise ValueError("Invalid crop geometry")
    if not 0.0 <= args.empty_max < args.sparse_min < args.sparse_max <= 1.0:
        raise ValueError("Expected empty_max < sparse_min < sparse_max within [0, 1]")
    train_root = SOURCE_DATASET / "train"
    candidates, scan = scan_candidates(
        train_root / "images",
        train_root / "labels",
        crop_size=args.crop_size,
        stride=args.stride,
        sparse_min=args.sparse_min,
        sparse_max=args.sparse_max,
        empty_max=args.empty_max,
    )
    print(json.dumps(scan, ensure_ascii=False, indent=2))
    if args.analyze_only:
        return 0
    selected = choose_candidates(candidates, sparse_count=args.sparse_count, empty_count=args.empty_count)
    manifest = build_dataset(args, selected, scan)
    print(f"generated_images={manifest['generated_images']}")
    print(f"total_train_images={manifest['total_train_images']}")
    print(f"manifest={args.output_root.resolve() / 'manifest.json'}")
    print(f"data={args.output_yaml.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
