"""Build deterministic train+valid-only grouped CV lists without touching the sealed test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from build_sparse_mixed_difficult_review import scene_features
from multispectral import PROJECT_ROOT


TRAIN_ROOT = PROJECT_ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed" / "train"
HUMAN_DATA_ROOT = PROJECT_ROOT / "datasets" / "Data_v3"
VALID_ROOT = PROJECT_ROOT / "datasets" / "Data_v3" / "valid"
FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v4_candidates"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "generated" / "data_v9_grouped_cv"
STEM_PATTERN = re.compile(r"^DJI_(\d+)$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_labels(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item).lower()):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def collect_split(root: Path, source_split: str) -> list[dict[str, object]]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    if root.name.lower() != source_split:
        raise ValueError(f"Expected {source_split} root, got {root}")
    records: list[dict[str, object]] = []
    for image_path in sorted(images_dir.glob("*.JPG"), key=lambda path: path.name.lower()):
        match = STEM_PATTERN.match(image_path.stem)
        if not match:
            raise ValueError(f"Unexpected image stem: {image_path.stem}")
        label_path = labels_dir / f"{image_path.stem}.txt"
        human_image_path = HUMAN_DATA_ROOT / source_split / "images" / image_path.name
        human_label_path = HUMAN_DATA_ROOT / source_split / "labels" / f"{image_path.stem}.txt"
        feature_path = FEATURE_ROOT / source_split / "features" / f"{image_path.stem}.npy"
        if not label_path.is_file() or not human_image_path.is_file() or not human_label_path.is_file() or not feature_path.is_file():
            raise FileNotFoundError(f"Missing refined/human label, image, or feature for {image_path}")
        image_sha256 = sha256_file(image_path)
        human_image_sha256 = image_sha256 if image_path.resolve() == human_image_path.resolve() else sha256_file(human_image_path)
        if image_sha256 != human_image_sha256:
            raise RuntimeError(
                f"H0/H1 image bytes differ for {image_path.stem}; the paired comparison may change labels only"
            )
        # Fold balancing and every validation list use human labels. Teacher
        # refinements are allowed only on the H0 training side of a fold.
        scene = scene_features(human_image_path, human_label_path)
        records.append(
            {
                "stem": image_path.stem,
                "number": int(match.group(1)),
                "source_split": source_split,
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "human_image": str(human_image_path.resolve()),
                "human_label": str(human_label_path.resolve()),
                "feature": str(feature_path.resolve()),
                "image_sha256": image_sha256,
                "human_image_sha256": human_image_sha256,
                "polygon_count": int(scene["polygon_count"]),
                "foreground_coverage": float(scene["foreground_coverage"]),
            }
        )
    return records


def build_sequence_groups(records: list[dict[str, object]], max_gap: int) -> list[dict[str, object]]:
    ordered = sorted(records, key=lambda row: (int(row["number"]), str(row["stem"])))
    if len({str(row["stem"]) for row in ordered}) != len(ordered):
        raise ValueError("Train and valid contain duplicate stems")
    groups: list[list[dict[str, object]]] = []
    for row in ordered:
        if not groups or int(row["number"]) - int(groups[-1][-1]["number"]) > max_gap:
            groups.append([])
        groups[-1].append(row)

    result: list[dict[str, object]] = []
    for index, rows in enumerate(groups):
        group_id = f"sequence_{index:02d}_{int(rows[0]['number']):04d}_{int(rows[-1]['number']):04d}"
        for row in rows:
            row["sequence_group"] = group_id
        result.append(
            {
                "group_id": group_id,
                "first_number": int(rows[0]["number"]),
                "last_number": int(rows[-1]["number"]),
                "images": len(rows),
                "instances": sum(int(row["polygon_count"]) for row in rows),
                "coverage_sum": sum(float(row["foreground_coverage"]) for row in rows),
                "original_valid_images": sum(row["source_split"] == "valid" for row in rows),
                "stems": [str(row["stem"]) for row in rows],
            }
        )
    return result


def assign_groups(groups: list[dict[str, object]], folds: int) -> dict[str, int]:
    totals = {
        "images": sum(int(group["images"]) for group in groups),
        "instances": sum(int(group["instances"]) for group in groups),
        "coverage_sum": sum(float(group["coverage_sum"]) for group in groups),
        "original_valid_images": sum(int(group["original_valid_images"]) for group in groups),
    }
    targets = {key: max(value / folds, 1e-9) for key, value in totals.items()}
    states = [dict(images=0, instances=0, coverage_sum=0.0, original_valid_images=0) for _ in range(folds)]
    assignment: dict[str, int] = {}
    ordered = sorted(
        groups,
        key=lambda group: (-int(group["images"]), -int(group["instances"]), str(group["group_id"])),
    )
    # Original train labels include teacher-refined boundaries while original
    # valid labels remain human-only. Balance that source distinction strongly
    # enough that no fold gets an artificially easy label-quality mix.
    weights = {"images": 0.35, "instances": 0.25, "coverage_sum": 0.10, "original_valid_images": 0.30}
    for group in ordered:
        scores: list[tuple[float, int, int]] = []
        for fold, state in enumerate(states):
            score = 0.0
            for key, weight in weights.items():
                projected = float(state[key]) + float(group[key])
                score += weight * (projected / targets[key]) ** 2
            scores.append((score, int(state["images"]), fold))
        _, _, selected = min(scores)
        assignment[str(group["group_id"])] = selected
        for key in states[selected]:
            states[selected][key] += group[key]
    return assignment


def build(output: Path, folds: int, max_gap: int) -> dict[str, object]:
    if folds < 2:
        raise ValueError("At least two folds are required")
    if max_gap < 10:
        raise ValueError("max_gap must be at least the nominal 10-frame interval")
    records = collect_split(TRAIN_ROOT, "train") + collect_split(VALID_ROOT, "valid")
    if len(records) != 270:
        raise RuntimeError(f"Expected 270 train+valid images, got {len(records)}")
    groups = build_sequence_groups(records, max_gap)
    assignment = assign_groups(groups, folds)
    output.mkdir(parents=True, exist_ok=True)
    fold_contracts: list[dict[str, object]] = []

    for fold in range(folds):
        val_groups = {group_id for group_id, assigned in assignment.items() if assigned == fold}
        val_rows = sorted(
            (row for row in records if str(row["sequence_group"]) in val_groups),
            key=lambda row: int(row["number"]),
        )
        train_rows = sorted(
            (row for row in records if str(row["sequence_group"]) not in val_groups),
            key=lambda row: int(row["number"]),
        )
        fold_dir = output / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_list = fold_dir / "train.txt"
        human_train_list = fold_dir / "train_human.txt"
        val_list = fold_dir / "val.txt"
        train_list.write_text("\n".join(Path(str(row["image"])).as_posix() for row in train_rows) + "\n", encoding="utf-8")
        human_train_list.write_text(
            "\n".join(Path(str(row["human_image"])).as_posix() for row in train_rows) + "\n",
            encoding="utf-8",
        )
        val_list.write_text(
            "\n".join(Path(str(row["human_image"])).as_posix() for row in val_rows) + "\n",
            encoding="utf-8",
        )
        yaml_path = fold_dir / "data_5ch.yaml"
        yaml_path.write_text(
            "\n".join(
                (
                    f"train: {train_list.resolve().as_posix()}",
                    f"val: {val_list.resolve().as_posix()}",
                    "",
                    "nc: 1",
                    "names: ['land']",
                    "channels: 5",
                    "",
                )
            ),
            encoding="utf-8",
        )
        human_yaml_path = fold_dir / "data_human_5ch.yaml"
        human_yaml_path.write_text(
            "\n".join(
                (
                    f"train: {human_train_list.resolve().as_posix()}",
                    f"val: {val_list.resolve().as_posix()}",
                    "",
                    "nc: 1",
                    "names: ['land']",
                    "channels: 5",
                    "",
                )
            ),
            encoding="utf-8",
        )
        fold_contracts.append(
            {
                "fold": fold,
                "train_images": len(train_rows),
                "validation_images": len(val_rows),
                "train_instances": sum(int(row["polygon_count"]) for row in train_rows),
                "validation_instances": sum(int(row["polygon_count"]) for row in val_rows),
                "validation_original_train": sum(row["source_split"] == "train" for row in val_rows),
                "validation_original_valid": sum(row["source_split"] == "valid" for row in val_rows),
                "validation_groups": sorted(val_groups),
                "train_list": str(train_list.resolve()),
                "human_train_list": str(human_train_list.resolve()),
                "validation_list": str(val_list.resolve()),
                "data_yaml": str(yaml_path.resolve()),
                "human_data_yaml": str(human_yaml_path.resolve()),
                "validation_stems": [str(row["stem"]) for row in val_rows],
            }
        )

    payload = {
        "version": "data_v9_grouped_cv_contract_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_splits": ["train", "valid"],
        "source_images": len(records),
        "source_train_images": sum(row["source_split"] == "train" for row in records),
        "source_valid_images": sum(row["source_split"] == "valid" for row in records),
        "held_out_split": "test",
        "held_out_test_used": False,
        "test_paths_enumerated": False,
        "folds": folds,
        "sequence_group_max_gap": max_gap,
        "sequence_groups": len(groups),
        "group_assignment": assignment,
        "fold_contracts": fold_contracts,
        "refined_mixed_train_label_sha256": sha256_labels([Path(str(row["label"])) for row in records]),
        "human_label_sha256": sha256_labels([Path(str(row["human_label"])) for row in records]),
        "validation_label_policy": "human_only_for_every_fold",
        "training_label_variants": {
            "h0_refined": "teacher-refined original train labels plus human original valid labels",
            "h1_human": "human labels for original train and original valid",
        },
        "records": sorted(records, key=lambda row: int(row["number"])),
    }
    (output / "contract.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-gap", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.output.resolve(), args.folds, args.max_gap)
    print(f"source_images={payload['source_images']}")
    print(f"sequence_groups={payload['sequence_groups']}")
    for fold in payload["fold_contracts"]:
        print(
            f"fold={fold['fold']} train={fold['train_images']} val={fold['validation_images']} "
            f"val_instances={fold['validation_instances']} groups={len(fold['validation_groups'])}"
        )
    print(f"contract={args.output.resolve() / 'contract.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
