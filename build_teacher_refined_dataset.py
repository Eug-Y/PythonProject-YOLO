"""Build train labels refined by high-confidence teacher masks while preserving unmatched human instances."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from ultralytics.utils.ops import segments2boxes

from calibrate_yolo_ensemble import load_yolo_ground_truth
from multispectral import PROJECT_ROOT, read_image
from prediction_bundle import load_prediction_bundle


def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None]:
    """Return mask area and exclusive xy bounds without changing IoU semantics."""
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return 0, None
    return int(len(rows)), (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _bounded_mask_iou(
    first: np.ndarray,
    first_geometry: tuple[int, tuple[int, int, int, int] | None],
    second: np.ndarray,
    second_geometry: tuple[int, tuple[int, int, int, int] | None],
) -> float:
    """Compute exact mask IoU while touching only the overlapping bounding-box region."""
    first_area, first_bounds = first_geometry
    second_area, second_bounds = second_geometry
    if not first_area or not second_area or first_bounds is None or second_bounds is None:
        return 0.0
    x1 = max(first_bounds[0], second_bounds[0])
    y1 = max(first_bounds[1], second_bounds[1])
    x2 = min(first_bounds[2], second_bounds[2])
    y2 = min(first_bounds[3], second_bounds[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    intersection = int(np.logical_and(first[y1:y2, x1:x2], second[y1:y2, x1:x2]).sum())
    if not intersection:
        return 0.0
    return intersection / float(first_area + second_area - intersection)


def mask_to_yolo_line(mask: np.ndarray, class_id: int = 0, simplify_ratio: float = 0.002) -> str | None:
    """Convert the largest external component of one mask to a valid YOLO polygon."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or not mask.any():
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 3.0:
        return None
    epsilon = max(0.5, cv2.arcLength(contour, True) * simplify_ratio)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    height, width = mask.shape
    normalized = polygon.astype(np.float64) / np.asarray([width, height], dtype=np.float64)
    normalized = np.clip(normalized, 0.0, 1.0)
    coordinates = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
    return f"{int(class_id)} {coordinates}"


def _yolo_box_signature(line: str) -> bytes:
    """Match Ultralytics' float32 class+xywh duplicate-label check exactly."""
    values = line.split()
    class_id = np.asarray([values[0]], dtype=np.float32)
    points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
    return np.concatenate((class_id, segments2boxes([points])[0])).tobytes()


def refine_label_lines(
    original_lines: list[str],
    ground_truth_masks: list[np.ndarray],
    teacher_masks: np.ndarray,
    teacher_scores: np.ndarray,
    *,
    score_threshold: float,
    match_iou_threshold: float,
) -> tuple[list[str], dict[str, int]]:
    """Replace a GT line only with a unique, confident, sufficiently overlapping teacher mask."""
    if len(original_lines) != len(ground_truth_masks):
        raise ValueError("Human label line/mask count mismatch")
    ground_truth_geometry = [_mask_geometry(mask) for mask in ground_truth_masks]
    candidates: list[tuple[float, float, int, int]] = []
    for teacher_index, (mask, score) in enumerate(zip(teacher_masks, teacher_scores)):
        if float(score) < score_threshold:
            continue
        teacher_geometry = _mask_geometry(mask)
        if not teacher_geometry[0]:
            continue
        for gt_index, (target, target_geometry) in enumerate(zip(ground_truth_masks, ground_truth_geometry)):
            overlap = _bounded_mask_iou(mask, teacher_geometry, target, target_geometry)
            if overlap >= match_iou_threshold:
                candidates.append((overlap, float(score), teacher_index, gt_index))
    candidates.sort(reverse=True)
    used_teacher: set[int] = set()
    used_gt: set[int] = set()
    replacements: dict[int, str] = {}
    for _, _, teacher_index, gt_index in candidates:
        if teacher_index in used_teacher or gt_index in used_gt:
            continue
        class_id = int(float(original_lines[gt_index].split()[0]))
        line = mask_to_yolo_line(teacher_masks[teacher_index], class_id)
        if line is None:
            continue
        replacements[gt_index] = line
        used_teacher.add(teacher_index)
        used_gt.add(gt_index)
    duplicate_rejections = 0
    while replacements:
        proposed = [replacements.get(index, line) for index, line in enumerate(original_lines)]
        line_indices: dict[bytes, list[int]] = {}
        for index, line in enumerate(proposed):
            line_indices.setdefault(_yolo_box_signature(line), []).append(index)
        conflicting = {
            index
            for indices in line_indices.values()
            if len(indices) > 1
            for index in indices
            if index in replacements
        }
        if not conflicting:
            break
        for index in conflicting:
            replacements.pop(index)
            duplicate_rejections += 1
    output = [replacements.get(index, line) for index, line in enumerate(original_lines)]
    return output, {
        "human_instances": len(original_lines),
        "teacher_instances": len(teacher_masks),
        "refined_instances": len(replacements),
        "preserved_instances": len(original_lines) - len(replacements),
        "unmatched_teacher_instances": len(teacher_masks) - len(used_teacher) + duplicate_rejections,
        "duplicate_replacement_rejections": duplicate_rejections,
    }


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v6_teacher_refined_yolo11")
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--match-iou", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0 or not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("Thresholds must be in [0, 1]")
    prediction_manifest = json.loads((args.teacher_predictions / "manifest.json").read_text(encoding="utf-8"))
    if prediction_manifest.get("split") != "train" or prediction_manifest.get("held_out_test_used") is not False:
        raise RuntimeError("Teacher refinement requires a train-only prediction export")
    images = sorted((args.dataset / "train" / "images").glob("*.JPG"))
    output = args.output.resolve()
    counts: Counter[str] = Counter()
    per_image: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for image_path in images:
        try:
            image_hw = tuple(read_image(image_path).shape[:2])
            label_path = args.dataset / "train" / "labels" / f"{image_path.stem}.txt"
            original_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            gt_masks = load_yolo_ground_truth(label_path, image_hw)
            teacher = load_prediction_bundle(args.teacher_predictions / "predictions" / f"{image_path.stem}.npz")
            if tuple(teacher["image_hw"]) != image_hw:
                raise ValueError(f"Teacher/image shape mismatch: {teacher['image_hw']} versus {image_hw}")
            refined, stats = refine_label_lines(
                original_lines,
                gt_masks,
                teacher["masks"],
                teacher["scores"],
                score_threshold=args.score_threshold,
                match_iou_threshold=args.match_iou,
            )
            link_status = link_or_copy(image_path, output / "train" / "images" / image_path.name)
            label_target = output / "train" / "labels" / f"{image_path.stem}.txt"
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("\n".join(refined) + ("\n" if refined else ""), encoding="utf-8")
            counts.update(stats)
            counts[link_status] += 1
            per_image.append({"stem": image_path.stem, **stats})
        except Exception as exc:
            failures.append({"image": image_path.name, "error": f"{type(exc).__name__}: {exc}"})
            counts["failed_images"] += 1

    yaml_path = output / "data_5ch_teacher_refined.yaml"
    yaml_path.write_text(
        "\n".join(
            (
                f"path: {output.as_posix()}",
                "train: train/images",
                f"val: {(args.dataset.resolve() / 'valid' / 'images').as_posix()}",
                "",
                "nc: 1",
                "names: ['land']",
                "channels: 5",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": "data_v6_teacher_refined_yolo11_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_dataset": str(args.dataset.resolve()),
        "teacher_prediction_export": str(args.teacher_predictions.resolve()),
        "teacher_checkpoint": prediction_manifest.get("checkpoint"),
        "selection_splits": ["train", "valid"],
        "teacher_prediction_split": "train",
        "validation_labels": "original human valid labels",
        "held_out_test_used": False,
        "score_threshold": args.score_threshold,
        "match_iou_threshold": args.match_iou,
        "image_count": len(images),
        "counts": dict(counts),
        "failures": failures,
        "per_image": per_image,
        "policy": "Replace only uniquely matched human instance boundaries; preserve all unmatched human instances; never add unmatched teacher predictions.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "per_image"}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
