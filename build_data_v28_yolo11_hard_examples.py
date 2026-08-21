"""Build the V28 train-only YOLO11 hard-example dataset.

The selection source is a frozen V19 YOLO11 prediction export over the
training split.  High-confidence unmatched predictions provide a very small
hard-negative crop set, while train images with the most unmatched ground
truth instances are replayed at a 2:1 ratio to protect recall.  Validation is
used only by the later model gate; test and sealed blind data are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from build_data_v20_sparse_context_dataset import clipped_labels, read_polygons, save_rgb_jpeg
from build_data_v22_dense_replay_dataset import flight_groups
from calibrate_yolo_ensemble import load_yolo_ground_truth
from multispectral import PROJECT_ROOT, read_image
from prediction_bundle import load_prediction_bundle
from yolo_mask_ensemble import bbox_from_mask, cropped_mask_iou


SOURCE_DATASET = PROJECT_ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed"
SOURCE_FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v4_candidates"
PREDICTION_ROOT = PROJECT_ROOT / "reports" / "Data_v28_yolo11_train_hard_mining"
V19_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "land_seg"
    / "v19_yolo11_c0_nooverlap_b1_seed0_v1"
    / "weights"
    / "best.pt"
)
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "Data_v28_yolo11_hard_examples_train"
OUTPUT_YAML = PROJECT_ROOT / "configs" / "generated" / "data_v28_yolo11_hard_examples_5ch.yaml"
ANALYSIS_OUTPUT = PROJECT_ROOT / "logs" / "v28_yolo11_hard_example_analysis.summary.json"
REVIEW_OUTPUT = PROJECT_ROOT / "logs" / "v28_yolo11_experiment_review.summary.json"
VALID_IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images"
CHANNELS = ["R", "G", "B", "NDVI", "NDRE"]
ALL_CHANNELS = [
    "R", "G", "B", "NIR", "RedEdge", "NDVI", "NDRE", "GNDVI", "SAVI",
    "OSAVI", "MSAVI2", "EVI2", "RECI", "BI", "SCI", "ExG", "RE_BSI_PROXY",
]
FORBIDDEN_TOKENS = ("data_v3/test", "data_release_blind_v1")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_path(path: Path) -> None:
    normalized = path.resolve().as_posix().lower()
    if any(token in normalized for token in FORBIDDEN_TOKENS):
        raise ValueError(f"V28 refuses test/blind data path: {path}")


def centered_crop(box: list[float] | tuple[float, ...], image_hw: tuple[int, int], size: int) -> tuple[int, int, int]:
    height, width = image_hw
    if size < 256 or size > min(height, width):
        raise ValueError(f"Invalid crop size {size} for image {image_hw}")
    x1, y1, x2, y2 = (float(value) for value in box)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    x = int(np.clip(round(center_x - size / 2.0), 0, width - size))
    y = int(np.clip(round(center_y - size / 2.0), 0, height - size))
    return x, y, size


def greedy_match_iou_matrix(iou_matrix: np.ndarray, threshold: float = 0.5) -> tuple[dict[int, int], list[int], list[int]]:
    """Match predictions in their confidence order, mirroring deployed evaluation."""
    matrix = np.asarray(iou_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("IoU matrix must be finite and two-dimensional")
    matched_targets: set[int] = set()
    matches: dict[int, int] = {}
    for prediction_index, values in enumerate(matrix):
        available = [index for index in range(matrix.shape[1]) if index not in matched_targets]
        if not available:
            break
        target_index = max(available, key=lambda index: (float(values[index]), -index))
        if float(values[target_index]) >= threshold:
            matches[prediction_index] = target_index
            matched_targets.add(target_index)
    unmatched_predictions = [index for index in range(matrix.shape[0]) if index not in matches]
    unmatched_targets = [index for index in range(matrix.shape[1]) if index not in matched_targets]
    return matches, unmatched_predictions, unmatched_targets


def diverse_select(
    rows: list[dict[str, object]],
    count: int,
    *,
    rank_fields: tuple[str, ...],
    excluded_sources: set[str] | None = None,
) -> list[dict[str, object]]:
    """Select strong rows while using each source and flight group once first."""
    if count < 1:
        return []
    excluded = excluded_sources or set()
    ordered = sorted(
        (dict(row) for row in rows if str(row["source_image"]) not in excluded),
        key=lambda row: tuple(-float(row[field]) for field in rank_fields) + (str(row["source_image"]),),
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
    raise RuntimeError(f"Only selected {len(selected)}/{count} diverse hard examples")


def pairwise_mask_ious(predictions: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    output = np.zeros((len(predictions), len(targets)), dtype=np.float32)
    if not len(predictions) or not targets:
        return output
    prediction_boxes = [bbox_from_mask(mask) for mask in predictions]
    target_boxes = [bbox_from_mask(mask) for mask in targets]
    for prediction_index, (prediction, prediction_box) in enumerate(zip(predictions, prediction_boxes, strict=True)):
        for target_index, (target, target_box) in enumerate(zip(targets, target_boxes, strict=True)):
            output[prediction_index, target_index] = cropped_mask_iou(
                prediction, target, prediction_box, target_box
            )
    return output


def validate_prediction_contract(prediction_root: Path) -> dict[str, object]:
    reject_forbidden_path(prediction_root)
    manifest_path = prediction_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "version": "multichannel_yolo_prediction_export_v1",
        "split": "train",
        "held_out_test_used": False,
        "channels": CHANNELS,
        "image_count": 210,
        "imgsz": 1024,
        "conf": 0.371,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"V28 prediction contract drifted for {key}: {manifest.get(key)!r}")
    if manifest.get("failures") != []:
        raise RuntimeError("V28 prediction export contains failures")
    resolved_dataset = Path(str(manifest["dataset"])).resolve()
    resolved_features = Path(str(manifest["feature_root"])).resolve()
    resolved_checkpoint = Path(str(manifest["checkpoint"])).resolve()
    for path in (resolved_dataset, resolved_features, resolved_checkpoint):
        reject_forbidden_path(path)
    if resolved_dataset != SOURCE_DATASET.resolve():
        raise RuntimeError("V28 mining predictions are not from the frozen train source")
    if resolved_features != SOURCE_FEATURE_ROOT.resolve():
        raise RuntimeError("V28 mining feature source drifted")
    if resolved_checkpoint != V19_CHECKPOINT.resolve():
        raise RuntimeError("V28 mining must use the frozen V19 YOLO11 checkpoint")
    bundles = list((prediction_root / "predictions").glob("*.npz"))
    if len(bundles) != 210:
        raise RuntimeError(f"Expected 210 train prediction bundles, got {len(bundles)}")
    return manifest


def analyze_train_predictions(
    prediction_root: Path,
    *,
    crop_size: int,
    match_iou: float,
    maximum_target_overlap: float,
) -> dict[str, object]:
    validate_prediction_contract(prediction_root)
    train_images = sorted(
        (path for path in (SOURCE_DATASET / "train" / "images").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if len(train_images) != 210:
        raise RuntimeError(f"Expected 210 source train images, got {len(train_images)}")
    groups = flight_groups([path.name for path in train_images])
    false_positive_candidates: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    totals = {"predictions": 0, "ground_truth": 0, "tp": 0, "fp": 0, "fn": 0}
    for image_path in train_images:
        label_path = SOURCE_DATASET / "train" / "labels" / f"{image_path.stem}.txt"
        feature_path = SOURCE_FEATURE_ROOT / "train" / "features" / f"{image_path.stem}.npy"
        bundle_path = prediction_root / "predictions" / f"{image_path.stem}.npz"
        for path in (label_path, feature_path, bundle_path):
            reject_forbidden_path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
        prediction = load_prediction_bundle(bundle_path)
        image_hw = tuple(int(value) for value in prediction["image_hw"])
        targets = load_yolo_ground_truth(label_path, image_hw)
        masks = prediction["masks"]
        ious = pairwise_mask_ious(masks, targets)
        matches, unmatched_predictions, unmatched_targets = greedy_match_iou_matrix(ious, match_iou)
        union_target = np.logical_or.reduce(targets) if targets else np.zeros(image_hw, dtype=bool)
        feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if feature.shape[:2] != image_hw or feature.ndim != 3 or feature.dtype != np.uint8:
            raise ValueError(f"Feature contract drifted for {feature_path}: {feature.shape} {feature.dtype}")
        tp, fp, fn = len(matches), len(unmatched_predictions), len(unmatched_targets)
        totals["predictions"] += len(masks)
        totals["ground_truth"] += len(targets)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        image_rows.append(
            {
                "source_image": image_path.name,
                "source_stem": image_path.stem,
                "source_group": groups[image_path.name],
                "predictions": len(masks),
                "ground_truth": len(targets),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "fn_rate": fn / len(targets) if targets else 0.0,
            }
        )
        for prediction_index in unmatched_predictions:
            mask = masks[prediction_index]
            area = int(mask.sum())
            if area < 64:
                continue
            overlap_fraction = float(np.logical_and(mask, union_target).sum()) / area
            max_iou = float(ious[prediction_index].max()) if ious.shape[1] else 0.0
            if overlap_fraction > maximum_target_overlap:
                continue
            box = [float(value) for value in prediction["boxes"][prediction_index]]
            x, y, size = centered_crop(box, image_hw, crop_size)
            crop_coverage = float(union_target[y : y + size, x : x + size].mean())
            score = float(prediction["scores"][prediction_index])
            false_positive_candidates.append(
                {
                    "source_image": image_path.name,
                    "source_stem": image_path.stem,
                    "source_group": groups[image_path.name],
                    "prediction_index": int(prediction_index),
                    "score": score,
                    "max_gt_iou": max_iou,
                    "target_overlap_fraction": overlap_fraction,
                    "prediction_area": area,
                    "prediction_area_fraction": area / float(image_hw[0] * image_hw[1]),
                    "box": box,
                    "x": x,
                    "y": y,
                    "size": size,
                    "crop_gt_coverage": crop_coverage,
                    "hard_negative_priority": score * (1.0 - max_iou) * (1.0 - overlap_fraction),
                }
            )
    totals["precision"] = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 1.0
    totals["recall"] = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 1.0
    return {
        "totals": totals,
        "false_positive_candidates": false_positive_candidates,
        "image_rows": image_rows,
    }


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def build_dataset(
    output_root: Path,
    output_yaml: Path,
    selected_negatives: list[dict[str, object]],
    selected_positives: list[dict[str, object]],
    analysis: dict[str, object],
    prediction_manifest: dict[str, object],
) -> dict[str, object]:
    output_root = output_root.resolve()
    output_yaml = output_yaml.resolve()
    if output_root.parent != (PROJECT_ROOT / "datasets").resolve():
        raise ValueError("Output dataset must stay directly inside datasets")
    if output_yaml.parent != (PROJECT_ROOT / "configs" / "generated").resolve():
        raise ValueError("Output YAML must stay inside configs/generated")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    train_images = SOURCE_DATASET / "train" / "images"
    train_labels = SOURCE_DATASET / "train" / "labels"
    originals = sorted(path for path in train_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(originals) != 210:
        raise RuntimeError(f"Expected 210 source images, got {len(originals)}")
    for image_path in originals:
        stem = image_path.stem
        label_path = train_labels / f"{stem}.txt"
        feature_path = SOURCE_FEATURE_ROOT / "train" / "features" / f"{stem}.npy"
        for path in (image_path, label_path, feature_path):
            reject_forbidden_path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
        hardlink(image_path, output_root / "train" / "images" / image_path.name)
        hardlink(label_path, output_root / "train" / "labels" / label_path.name)
        hardlink(feature_path, output_root / "train" / "features" / feature_path.name)
    valid_features = sorted((SOURCE_FEATURE_ROOT / "valid" / "features").glob("*.npy"))
    if len(valid_features) != 60:
        raise RuntimeError(f"Expected 60 valid features, got {len(valid_features)}")
    for path in valid_features:
        reject_forbidden_path(path)
        hardlink(path, output_root / "valid" / "features" / path.name)

    generated_negatives: list[dict[str, object]] = []
    for index, row in enumerate(selected_negatives, 1):
        stem = str(row["source_stem"])
        x, y, size = int(row["x"]), int(row["y"]), int(row["size"])
        output_stem = f"v28_hardneg_{index:02d}_{stem}_x{x}_y{y}_s{size}"
        image_path = train_images / str(row["source_image"])
        label_path = train_labels / f"{stem}.txt"
        feature_path = SOURCE_FEATURE_ROOT / "train" / "features" / f"{stem}.npy"
        image = read_image(image_path)
        feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if image.shape[:2] != feature.shape[:2]:
            raise ValueError(f"Image/feature mismatch for {stem}")
        polygons = read_polygons(label_path, feature.shape[0], feature.shape[1])
        label_lines, coverage = clipped_labels(polygons, x, y, size)
        crop_image = np.ascontiguousarray(image[y : y + size, x : x + size])
        crop_feature = np.ascontiguousarray(feature[y : y + size, x : x + size])
        output_image = output_root / "train" / "images" / f"{output_stem}.JPG"
        output_label = output_root / "train" / "labels" / f"{output_stem}.txt"
        output_feature = output_root / "train" / "features" / f"{output_stem}.npy"
        save_rgb_jpeg(output_image, crop_image)
        output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        np.save(output_feature, crop_feature, allow_pickle=False)
        generated_negatives.append(
            {
                **row,
                "output_stem": output_stem,
                "output_labels": len(label_lines),
                "realized_gt_coverage": coverage,
            }
        )

    replayed_positives: list[dict[str, object]] = []
    for index, row in enumerate(selected_positives, 1):
        stem = str(row["source_stem"])
        output_stem = f"v28_hardpos_{index:02d}_{stem}"
        hardlink(train_images / str(row["source_image"]), output_root / "train" / "images" / f"{output_stem}.JPG")
        hardlink(train_labels / f"{stem}.txt", output_root / "train" / "labels" / f"{output_stem}.txt")
        hardlink(
            SOURCE_FEATURE_ROOT / "train" / "features" / f"{stem}.npy",
            output_root / "train" / "features" / f"{output_stem}.npy",
        )
        replayed_positives.append({**row, "output_stem": output_stem})

    manifest = {
        "version": "data_v28_yolo11_hard_examples_train_v1",
        "created_at": now_iso(),
        "purpose": "conservative train-only YOLO11 false-positive mining with 2:1 hard-positive recall protection",
        "primary_model": "YOLO11",
        "auxiliary_model_training": False,
        "channels": ALL_CHANNELS,
        "selected_channels": CHANNELS,
        "selection_basis_splits": ["train"],
        "training_split_sources": ["train"],
        "evaluation_splits": ["valid"],
        "validation_labels_used_for_selection": False,
        "forbidden_splits_used": False,
        "held_out_test_used": False,
        "sealed_blind_used": False,
        "source_dataset": str(SOURCE_DATASET.resolve()),
        "source_feature_root": str(SOURCE_FEATURE_ROOT.resolve()),
        "prediction_export": str(PREDICTION_ROOT.resolve()),
        "prediction_checkpoint": prediction_manifest["checkpoint"],
        "prediction_checkpoint_sha256": sha256(V19_CHECKPOINT),
        "locked_confidence": prediction_manifest["conf"],
        "match_iou": 0.5,
        "source_train_images": 210,
        "linked_valid_features": len(valid_features),
        "hard_negative_count": len(generated_negatives),
        "hard_positive_replay_count": len(replayed_positives),
        "positive_to_negative_ratio": len(replayed_positives) / len(generated_negatives),
        "total_train_images": 210 + len(generated_negatives) + len(replayed_positives),
        "train_mining_totals": analysis["totals"],
        "hard_negatives": generated_negatives,
        "hard_positive_replays": replayed_positives,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(
        f"# {manifest['version']}; test and sealed blind data are not exposed.\n"
        f"path: {output_root.as_posix()}\n"
        "train: train/images\n"
        f"val: {VALID_IMAGES.resolve().as_posix()}\n\n"
        "nc: 1\n"
        "names: ['land']\n"
        "channels: 5\n",
        encoding="utf-8",
    )
    return manifest


def previous_experiment_review() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for version, filename in (
        ("V20", "v20_sparse_context_multiseed.summary.json"),
        ("V21", "v21_sparse_context_half_multiseed.summary.json"),
        ("V22", "v22_dense_replay_multiseed.summary.json"),
    ):
        payload = json.loads((PROJECT_ROOT / "logs" / filename).read_text(encoding="utf-8"))
        row = payload["rows"][0]
        rows.append(
            {
                "version": version,
                "candidate_map50_95_mask": row["candidate_map"],
                "map_delta": row["map_delta"],
                "fixed_precision": row["candidate_fixed"]["precision"],
                "fixed_recall": row["candidate_fixed"]["recall"],
                "fixed_f1": row["candidate_fixed"]["f1"],
                "fixed_f1_delta": row["fixed_f1_delta"],
                "decision": payload["decision"],
            }
        )
    reference = json.loads((PROJECT_ROOT / "logs" / "v20_sparse_context_multiseed.summary.json").read_text(encoding="utf-8"))["rows"][0]
    return {
        "version": "v28_yolo11_experiment_review_v1",
        "created_at": now_iso(),
        "reference": {
            "version": "V19",
            "map50_95_mask": reference["reference_map"],
            **reference["reference_fixed"],
        },
        "candidates": rows,
        "decision": "Use model-error-driven train-only mining; cap hard negatives at 4 and pair with 8 hard-positive replays",
        "held_out_test_used": False,
        "sealed_blind_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-yaml", type=Path, default=OUTPUT_YAML)
    parser.add_argument("--analysis-output", type=Path, default=ANALYSIS_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=REVIEW_OUTPUT)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--maximum-target-overlap", type=float, default=0.1)
    parser.add_argument("--hard-negative-count", type=int, default=4)
    parser.add_argument("--hard-positive-count", type=int, default=8)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--reuse-analysis",
        action="store_true",
        help="Reuse the audited analysis summary instead of recomputing train mask IoUs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hard_negative_count != 4 or args.hard_positive_count != 8:
        raise ValueError("V28 formal protocol is locked to 4 hard negatives and 8 hard-positive replays")
    if not 0.0 < args.match_iou <= 1.0 or not 0.0 <= args.maximum_target_overlap <= 0.25:
        raise ValueError("Invalid V28 matching thresholds")
    prediction_manifest = validate_prediction_contract(args.prediction_root)
    if args.reuse_analysis:
        summary = json.loads(args.analysis_output.read_text(encoding="utf-8"))
        if (
            summary.get("version") != "v28_yolo11_hard_example_analysis_v1"
            or summary.get("selection_basis_splits") != ["train"]
            or summary.get("validation_labels_used") is not False
            or summary.get("held_out_test_used") is not False
            or summary.get("sealed_blind_used") is not False
            or summary.get("prediction_checkpoint_sha256") != sha256(V19_CHECKPOINT)
            or len(summary.get("selected_hard_negatives", [])) != args.hard_negative_count
            or len(summary.get("selected_hard_positive_replays", [])) != args.hard_positive_count
        ):
            raise RuntimeError("Refusing unaudited or drifted V28 analysis summary")
        negatives = summary["selected_hard_negatives"]
        positives = summary["selected_hard_positive_replays"]
        analysis = {"totals": summary["totals"]}
    else:
        analysis = analyze_train_predictions(
            args.prediction_root,
            crop_size=args.crop_size,
            match_iou=args.match_iou,
            maximum_target_overlap=args.maximum_target_overlap,
        )
        negatives = diverse_select(
            analysis["false_positive_candidates"],
            args.hard_negative_count,
            rank_fields=("hard_negative_priority", "score", "prediction_area_fraction"),
        )
        positives = diverse_select(
            [row for row in analysis["image_rows"] if int(row["fn"]) > 0],
            args.hard_positive_count,
            rank_fields=("fn", "fn_rate", "fp"),
            excluded_sources={str(row["source_image"]) for row in negatives},
        )
        summary = {
            "version": "v28_yolo11_hard_example_analysis_v1",
            "created_at": now_iso(),
            "selection_basis_splits": ["train"],
            "validation_labels_used": False,
            "held_out_test_used": False,
            "sealed_blind_used": False,
            "prediction_manifest": str((args.prediction_root / "manifest.json").resolve()),
            "prediction_checkpoint_sha256": sha256(V19_CHECKPOINT),
            "match_iou": args.match_iou,
            "maximum_target_overlap": args.maximum_target_overlap,
            "totals": analysis["totals"],
            "eligible_false_positive_candidates": len(analysis["false_positive_candidates"]),
            "selected_hard_negatives": negatives,
            "selected_hard_positive_replays": positives,
            "positive_to_negative_ratio": len(positives) / len(negatives),
        }
        args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
        args.analysis_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    review = previous_experiment_review()
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.analyze_only:
        return 0
    manifest = build_dataset(
        args.output_root,
        args.output_yaml,
        negatives,
        positives,
        analysis,
        prediction_manifest,
    )
    print(json.dumps({"dataset": str(args.output_root.resolve()), "total_train_images": manifest["total_train_images"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
