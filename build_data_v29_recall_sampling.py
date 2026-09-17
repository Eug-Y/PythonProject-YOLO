"""Build conservative train-only V29 recall-priority sampling weights.

The source is the frozen V19 YOLO11 prediction export over the 210 training
images.  Only images with many unmatched human instances receive increased
sampling probability.  No negative crops are added and validation/test/blind
data are never used for sample selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from build_data_v28_yolo11_hard_examples import (
    PREDICTION_ROOT,
    SOURCE_DATASET,
    SOURCE_FEATURE_ROOT,
    V19_CHECKPOINT,
    analyze_train_predictions,
    reject_forbidden_path,
    validate_prediction_contract,
)
from build_data_v22_dense_replay_dataset import flight_groups
from multispectral import PROJECT_ROOT


OUTPUT = PROJECT_ROOT / "configs" / "generated" / "data_v29_train_recall_weights.json"
ANALYSIS_OUTPUT = PROJECT_ROOT / "logs" / "v29_recall_sampling_analysis.summary.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_recall_scenes(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Select high-FN images, using each flight group once before repeats."""
    if count < 1 or count > len(rows):
        raise ValueError("Recall scene count must be within the available train rows")
    ordered = sorted(
        (dict(row) for row in rows if int(row["fn"]) > 0),
        key=lambda row: (-int(row["fn"]), -float(row["fn_rate"]), str(row["source_image"])),
    )
    selected: list[dict[str, object]] = []
    used_sources: set[str] = set()
    used_groups: set[str] = set()
    for require_new_group in (True, False):
        for row in ordered:
            source = str(row["source_image"])
            group = str(row["source_group"])
            if source in used_sources or (require_new_group and group in used_groups):
                continue
            selected.append(row)
            used_sources.add(source)
            used_groups.add(group)
            if len(selected) == count:
                return selected
    raise RuntimeError(f"Only selected {len(selected)}/{count} recall-priority scenes")


def build_weights(
    image_names: list[str],
    selected: list[dict[str, object]],
    *,
    minimum_boost: float,
    maximum_boost: float,
) -> dict[str, float]:
    if len(image_names) != len(set(image_names)):
        raise ValueError("Training image names must be unique")
    if not 1.0 < minimum_boost <= maximum_boost <= 2.0:
        raise ValueError("Expected 1 < minimum_boost <= maximum_boost <= 2")
    weights = {name: 1.0 for name in image_names}
    count = len(selected)
    if not count:
        raise ValueError("At least one recall-priority scene is required")
    for rank, row in enumerate(selected):
        name = str(row["source_image"])
        if name not in weights:
            raise ValueError(f"Selected scene is absent from train: {name}")
        fraction = 1.0 - rank / max(count - 1, 1)
        weights[name] = minimum_boost + (maximum_boost - minimum_boost) * fraction
    return {name: round(value, 8) for name, value in weights.items()}


def effective_sample_size(weights: list[float]) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(values.sum() ** 2 / np.square(values).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--analysis-output", type=Path, default=ANALYSIS_OUTPUT)
    parser.add_argument("--selected-count", type=int, default=24)
    parser.add_argument("--minimum-boost", type=float, default=1.15)
    parser.add_argument("--maximum-boost", type=float, default=1.40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selected_count != 24 or args.minimum_boost != 1.15 or args.maximum_boost != 1.40:
        raise ValueError("V29 protocol is locked to 24 scenes with 1.15-1.40 weights")
    for path in (args.prediction_root, args.output, args.analysis_output, SOURCE_DATASET, SOURCE_FEATURE_ROOT):
        reject_forbidden_path(path)
    prediction_manifest = validate_prediction_contract(args.prediction_root)
    analysis = analyze_train_predictions(
        args.prediction_root,
        crop_size=640,
        match_iou=0.5,
        maximum_target_overlap=0.1,
    )
    train_images = sorted((SOURCE_DATASET / "train" / "images").glob("*.JPG"))
    if len(train_images) != 210:
        raise RuntimeError(f"Expected 210 V19 train images, got {len(train_images)}")
    groups = flight_groups([path.name for path in train_images])
    rows = analysis["image_rows"]
    if len(rows) != 210 or {str(row["source_image"]) for row in rows} != {path.name for path in train_images}:
        raise RuntimeError("Train mining rows do not exactly match the V19 training images")
    if any(str(row["source_group"]) != groups[str(row["source_image"])] for row in rows):
        raise RuntimeError("Train flight grouping drifted")
    selected = select_recall_scenes(rows, args.selected_count)
    weights = build_weights(
        [path.name for path in train_images],
        selected,
        minimum_boost=args.minimum_boost,
        maximum_boost=args.maximum_boost,
    )
    values = list(weights.values())
    labels = sorted((SOURCE_DATASET / "train" / "labels").glob("*.txt"))
    payload = {
        "version": "data_v29_train_recall_weights_v1",
        "created_at": now_iso(),
        "purpose": "recall-priority fine-tuning from V19 using train-only false-negative evidence",
        "primary_model": "YOLO11",
        "source_split": "train",
        "selection_splits": ["train"],
        "validation_split_used": False,
        "held_out_split": "test",
        "held_out_split_used": False,
        "sealed_blind_used": False,
        "source_dataset": str(SOURCE_DATASET.resolve()),
        "source_feature_root": str(SOURCE_FEATURE_ROOT.resolve()),
        "prediction_export": str(args.prediction_root.resolve()),
        "prediction_manifest_sha256": sha256(args.prediction_root / "manifest.json"),
        "prediction_checkpoint": prediction_manifest["checkpoint"],
        "prediction_checkpoint_sha256": sha256(V19_CHECKPOINT),
        "train_images": len(train_images),
        "train_labels_sha256": hashlib.sha256(b"\0".join(path.read_bytes() for path in labels)).hexdigest(),
        "selection": {
            "metric": "unmatched train ground-truth instances at IoU 0.5",
            "selected_count": len(selected),
            "selected_groups": len({str(row["source_group"]) for row in selected}),
            "minimum_boost": args.minimum_boost,
            "maximum_boost": args.maximum_boost,
            "negative_examples_added": 0,
        },
        "sampling": {
            "replacement": True,
            "samples_per_epoch": 210,
            "normalization": "relative raw weights; unselected images remain 1.0",
            "min_weight": min(values),
            "mean_weight": float(np.mean(values)),
            "max_weight": max(values),
            "max_to_min_ratio": max(values) / min(values),
            "effective_sample_size": effective_sample_size(values),
        },
        "train_mining_totals": analysis["totals"],
        "selected_recall_scenes": [
            {
                **row,
                "sampling_weight": weights[str(row["source_image"])],
            }
            for row in selected
        ],
        "weights_by_image": weights,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        key: value for key, value in payload.items()
        if key not in {"weights_by_image"}
    }
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "selected_count": len(selected),
        "selected_groups": payload["selection"]["selected_groups"],
        "weight_range": [min(values), max(values)],
        "effective_sample_size": payload["sampling"]["effective_sample_size"],
        "held_out_test_used": False,
        "sealed_blind_used": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
