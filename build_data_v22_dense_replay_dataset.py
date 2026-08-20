"""Build V22 by adding train-only high-overlap replay to the V21 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from analyze_v13_precision_regression import overlap_features
from multispectral import PROJECT_ROOT


SOURCE_ROOT = PROJECT_ROOT / "datasets" / "Data_v21_sparse_context_half_train"
ORIGINAL_ROOT = PROJECT_ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "Data_v22_sparse_context_dense_replay_train"
OUTPUT_YAML = PROJECT_ROOT / "configs" / "generated" / "data_v22_sparse_context_dense_replay_5ch.yaml"
VALID_IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_number(name: str) -> int:
    match = re.fullmatch(r"DJI_(\d+)\.JPG", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected original training image name: {name}")
    return int(match.group(1))


def flight_groups(names: list[str], maximum_gap: int = 20) -> dict[str, str]:
    ordered = sorted(names, key=image_number)
    groups: list[list[str]] = []
    for name in ordered:
        if not groups or image_number(name) - image_number(groups[-1][-1]) > maximum_gap:
            groups.append([name])
        else:
            groups[-1].append(name)
    output: dict[str, str] = {}
    for group in groups:
        first, last = image_number(group[0]), image_number(group[-1])
        label = f"train_sequence_{first:04d}_{last:04d}"
        output.update({name: label for name in group})
    return output


def select_balanced_replays(rows: list[dict[str, object]], count: int) -> tuple[list[dict[str, object]], float]:
    if count < 1 or len(rows) < count:
        raise ValueError("Replay count must be positive and no larger than the candidate pool")
    threshold = float(np.quantile([float(row["overlap_foreground_fraction"]) for row in rows], 0.75))
    eligible = [
        dict(row) for row in rows
        if float(row["overlap_foreground_fraction"]) >= threshold and int(row["overlap_instances"]) > 0
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in eligible:
        grouped.setdefault(str(row["source_group"]), []).append(row)
    for values in grouped.values():
        values.sort(
            key=lambda row: (
                -float(row["overlap_foreground_fraction"]),
                -int(row["overlap_instances"]),
                str(row["source_image"]),
            )
        )
    group_order = sorted(
        grouped,
        key=lambda name: (
            -float(grouped[name][0]["overlap_foreground_fraction"]),
            -int(grouped[name][0]["overlap_instances"]),
            name,
        ),
    )
    selected: list[dict[str, object]] = []
    while len(selected) < count and any(grouped[name] for name in group_order):
        for name in group_order:
            if grouped[name]:
                selected.append(grouped[name].pop(0))
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f"Only selected {len(selected)}/{count} high-overlap replay images")
    return selected, threshold


def scan_train_candidates() -> list[dict[str, object]]:
    images = sorted((ORIGINAL_ROOT / "train" / "images").glob("*.JPG"), key=lambda path: image_number(path.name))
    if len(images) != 210:
        raise RuntimeError(f"Expected 210 original train images, got {len(images)}")
    groups = flight_groups([path.name for path in images])
    rows: list[dict[str, object]] = []
    for image_path in images:
        label_path = ORIGINAL_ROOT / "train" / "labels" / f"{image_path.stem}.txt"
        feature_path = SOURCE_ROOT / "train" / "features" / f"{image_path.stem}.npy"
        if not label_path.is_file() or not feature_path.is_file():
            raise FileNotFoundError(label_path if not label_path.is_file() else feature_path)
        feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if feature.ndim != 3 or feature.dtype != np.uint8:
            raise ValueError(f"Invalid feature array: {feature_path}")
        overlap = overlap_features(label_path, int(feature.shape[0]), int(feature.shape[1]), 384)
        rows.append(
            {
                "source_image": image_path.name,
                "source_stem": image_path.stem,
                "source_group": groups[image_path.name],
                **overlap,
            }
        )
    return rows


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def build_dataset(output_root: Path, output_yaml: Path, selected: list[dict[str, object]], threshold: float) -> dict[str, object]:
    output_root = output_root.resolve()
    output_yaml = output_yaml.resolve()
    if output_root.parent != (PROJECT_ROOT / "datasets").resolve():
        raise ValueError("Output dataset must stay in the project datasets directory")
    if output_yaml.parent != (PROJECT_ROOT / "configs" / "generated").resolve():
        raise ValueError("Output YAML must stay in configs/generated")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    source_manifest_path = SOURCE_ROOT / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("version") != "data_v21_sparse_context_half_train_v1":
        raise RuntimeError("V21 source manifest contract drifted")

    linked_counts: dict[str, int] = {}
    for relative in ("train/images", "train/labels", "train/features", "valid/features"):
        source_dir = SOURCE_ROOT / relative
        paths = sorted(path for path in source_dir.iterdir() if path.is_file())
        for path in paths:
            hardlink(path, output_root / relative / path.name)
        linked_counts[relative] = len(paths)

    replayed: list[dict[str, object]] = []
    for index, row in enumerate(selected, 1):
        source_stem = str(row["source_stem"])
        output_stem = f"dense_replay_{index:03d}_{source_stem}"
        source_image = SOURCE_ROOT / "train" / "images" / f"{source_stem}.JPG"
        source_label = SOURCE_ROOT / "train" / "labels" / f"{source_stem}.txt"
        source_feature = SOURCE_ROOT / "train" / "features" / f"{source_stem}.npy"
        hardlink(source_image, output_root / "train" / "images" / f"{output_stem}.JPG")
        hardlink(source_label, output_root / "train" / "labels" / f"{output_stem}.txt")
        hardlink(source_feature, output_root / "train" / "features" / f"{output_stem}.npy")
        replayed.append({**row, "output_stem": output_stem})

    manifest = {
        "version": "data_v22_sparse_context_dense_replay_train_v1",
        "created_at": now_iso(),
        "purpose": "retain V21 sparse-context FP suppression while restoring dense/high-overlap recall",
        "channels": source_manifest["channels"],
        "selection_basis_splits": ["train"],
        "selection_splits": ["train", "valid"],
        "training_evaluation_splits": ["train", "valid"],
        "validation_labels_used_for_replay_selection": False,
        "held_out_split": "test",
        "held_out_split_used": False,
        "source_dataset": str(SOURCE_ROOT.resolve()),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_train_images": 220,
        "linked_counts": linked_counts,
        "sparse_context_generated": source_manifest["generated"],
        "replay_selection": {
            "metric": "train-label overlap_foreground_fraction",
            "candidate_quantile": 0.75,
            "threshold": threshold,
            "requested": len(selected),
            "selected_groups": len({str(row["source_group"]) for row in selected}),
            "raster_long_side": 384,
        },
        "replayed": replayed,
        "total_train_images": 220 + len(replayed),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(
        f"# {manifest['version']}; train-only replay, held-out test is not exposed.\n"
        f"path: {output_root.as_posix()}\n"
        "train: train/images\n"
        f"val: {VALID_IMAGES.resolve().as_posix()}\n\n"
        "nc: 1\n"
        "names: ['land']\n"
        "channels: 5\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-count", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-yaml", type=Path, default=OUTPUT_YAML)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = scan_train_candidates()
    selected, threshold = select_balanced_replays(rows, args.replay_count)
    preview = {
        "threshold": threshold,
        "selected": selected,
        "selected_groups": len({str(row["source_group"]) for row in selected}),
        "held_out_test_used": False,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if args.analyze_only:
        return 0
    manifest = build_dataset(args.output_root, args.output_yaml, selected, threshold)
    print(json.dumps({"total_train_images": manifest["total_train_images"], "manifest": str((args.output_root / 'manifest.json').resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
