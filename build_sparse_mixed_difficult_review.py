"""Build a validation-only review queue focused on sparse and mixed-background scenes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from multispectral import PROJECT_ROOT, read_image


DEFAULT_ERRORS = PROJECT_ROOT / "reports" / "Data_v6_final_reviewed_best_valid_errors"
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "labels"
DEFAULT_ISSUES = PROJECT_ROOT / "reports" / "Data_v3_final_review_trainval_audit" / "issues.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "Data_v6_sparse_mixed_difficult_review"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def label_mask(label_path: Path, height: int, width: int) -> tuple[np.ndarray, int]:
    mask = np.zeros((height, width), dtype=np.uint8)
    count = 0
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        points = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
        pixels = np.rint(points * np.asarray([width - 1, height - 1])).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 1)
        count += 1
    return mask.astype(bool), count


def scene_features(image_path: Path, label_path: Path) -> dict[str, float | int]:
    image = read_image(image_path)
    height, width = image.shape[:2]
    foreground, polygon_count = label_mask(label_path, height, width)
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150) > 0
    background = ~foreground
    coverage = float(foreground.mean())
    if background.any():
        background_std = float(gray[background].std()) / 255.0
        background_edge = float(edges[background].mean())
    else:
        background_std = 0.0
        background_edge = 0.0
    return {
        "polygon_count": polygon_count,
        "foreground_coverage": coverage,
        "background_ratio": 1.0 - coverage,
        "background_luma_std": background_std,
        "background_edge_density": background_edge,
    }


def percentile_rank(values: np.ndarray, value: float, *, inverse: bool = False) -> float:
    if not len(values):
        return 0.0
    rank = float(np.mean(values <= value))
    return 1.0 - rank if inverse else rank


def build_queue(
    metrics_path: Path,
    analysis_path: Path,
    image_dir: Path,
    label_dir: Path,
    issues_path: Path,
    output: Path,
    limit: int,
) -> list[dict[str, object]]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("split") != "val":
        raise ValueError(f"Refusing non-validation metrics: split={analysis.get('split')!r}")
    metrics = read_csv(metrics_path)
    image_names = sorted(path.name for path in image_dir.glob("*.JPG"))
    if not metrics or set(row["image"] for row in metrics) != set(image_names):
        raise ValueError("Validation images and per-image metrics must match exactly")
    batch_index = {name: index for index, name in enumerate(image_names)}
    warnings: dict[str, set[str]] = defaultdict(set)
    if issues_path.is_file():
        for row in read_csv(issues_path):
            if row.get("split") == "valid":
                warnings[row.get("stem", "")].add(row.get("code", ""))

    records: list[dict[str, object]] = []
    metric_by_image = {row["image"]: row for row in metrics}
    for name in image_names:
        stem = Path(name).stem
        scene = scene_features(image_dir / name, label_dir / f"{stem}.txt")
        records.append({"image": name, "stem": stem, **scene})

    counts = np.asarray([float(row["polygon_count"]) for row in records])
    coverages = np.asarray([float(row["foreground_coverage"]) for row in records])
    texture = np.asarray([float(row["background_luma_std"]) for row in records])
    edges = np.asarray([float(row["background_edge_density"]) for row in records])
    max_fn = max(as_float(row, "mask_fn") for row in metrics) or 1.0
    max_fp = max(as_float(row, "mask_fp") for row in metrics) or 1.0

    for row in records:
        metric = metric_by_image[str(row["image"])]
        low_count = percentile_rank(counts, float(row["polygon_count"]), inverse=True)
        low_coverage = percentile_rank(coverages, float(row["foreground_coverage"]), inverse=True)
        texture_rank = percentile_rank(texture, float(row["background_luma_std"]))
        edge_rank = percentile_rank(edges, float(row["background_edge_density"]))
        sparse_score = 0.45 * low_count + 0.55 * low_coverage
        mixed_score = float(row["background_ratio"]) * (0.55 * texture_rank + 0.45 * edge_rank)
        error_score = (
            0.50 * (1.0 - as_float(metric, "mask_f1"))
            + 0.35 * as_float(metric, "mask_fn") / max_fn
            + 0.15 * as_float(metric, "mask_fp") / max_fp
        )
        row.update(
            mask_f1=as_float(metric, "mask_f1"),
            mask_precision=as_float(metric, "mask_precision"),
            mask_recall=as_float(metric, "mask_recall"),
            mask_fp=int(as_float(metric, "mask_fp")),
            mask_fn=int(as_float(metric, "mask_fn")),
            sparse_score=sparse_score,
            mixed_background_score=mixed_score,
            error_score=error_score,
            priority_score=100.0 * (0.50 * error_score + 0.25 * sparse_score + 0.25 * mixed_score),
        )

    sparse = sorted(records, key=lambda row: (-float(row["sparse_score"]), -float(row["error_score"]), str(row["image"])))
    mixed = sorted(records, key=lambda row: (-float(row["mixed_background_score"]), -float(row["error_score"]), str(row["image"])))
    overall = sorted(records, key=lambda row: (-float(row["priority_score"]), str(row["image"])))
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in sparse[: max(8, limit // 2)]:
        reasons[str(row["image"])].add("稀疏/低覆盖")
    for row in mixed[: max(8, limit // 2)]:
        reasons[str(row["image"])].add("混合背景")
    for row in sorted(records, key=lambda item: (-int(item["mask_fn"]), str(item["image"])))[:8]:
        reasons[str(row["image"])].add("高漏检 FN")
    selected = [row for row in overall if str(row["image"]) in reasons]
    if len(selected) < limit:
        selected_names = {str(row["image"]) for row in selected}
        for row in overall:
            if str(row["image"]) in selected_names:
                continue
            reasons[str(row["image"])].add("综合困难度")
            selected.append(row)
            selected_names.add(str(row["image"]))
            if len(selected) >= limit:
                break
    selected.sort(key=lambda row: (-float(row["priority_score"]), str(row["image"])))
    selected = selected[:limit]

    queue: list[dict[str, object]] = []
    for rank, row in enumerate(selected, 1):
        stem = str(row["stem"])
        queue.append(
            {
                "rank": rank,
                "image": row["image"],
                "stem": stem,
                "split": "valid",
                "batch_index": batch_index[str(row["image"])],
                "mask_f1": round(float(row["mask_f1"]), 6),
                "mask_precision": round(float(row["mask_precision"]), 6),
                "mask_recall": round(float(row["mask_recall"]), 6),
                "mask_fp": row["mask_fp"],
                "mask_fn": row["mask_fn"],
                "polygon_count": row["polygon_count"],
                "foreground_coverage": round(float(row["foreground_coverage"]), 6),
                "background_luma_std": round(float(row["background_luma_std"]), 6),
                "background_edge_density": round(float(row["background_edge_density"]), 6),
                "sparse_score": round(float(row["sparse_score"]), 6),
                "mixed_background_score": round(float(row["mixed_background_score"]), 6),
                "difficulty_score": round(float(row["priority_score"]), 2),
                "reasons": ", ".join(sorted(reasons[str(row["image"])])),
                "label_warning_codes": ", ".join(sorted(filter(None, warnings.get(stem, set())))),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    summary = {
        "version": "data_v6_sparse_mixed_difficult_review_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_split": "valid",
        "held_out_test_used": False,
        "checkpoint": analysis.get("checkpoint"),
        "imgsz": analysis.get("imgsz"),
        "confidence": analysis.get("confidence"),
        "validation_images": len(metrics),
        "selected_images": len(queue),
        "focus": ["sparse_or_low_coverage", "mixed_background", "high_false_negative"],
        "selection_is_read_only": True,
        "label_changes_applied": False,
        "review_state": str((output / "review_state.json").resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    queue = build_queue(
        (args.errors / "per_image_metrics.csv").resolve(),
        (args.errors / "analysis_config.json").resolve(),
        args.images.resolve(),
        args.labels.resolve(),
        args.issues.resolve(),
        args.output.resolve(),
        args.limit,
    )
    print(f"selected={len(queue)}")
    print(f"queue={args.output.resolve() / 'review_queue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
