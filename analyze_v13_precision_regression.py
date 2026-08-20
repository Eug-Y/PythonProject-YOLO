"""Pair V13 control/candidate validation errors without touching the sealed test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from build_sparse_mixed_difficult_review import scene_features
from multispectral import PROJECT_ROOT


DEFAULT_ROOT = PROJECT_ROOT / "reports" / "Data_v13_overlap_precision_analysis"
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DEFAULT_ROOT / "control_independent_conf025")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_ROOT / "candidate_independent_conf025")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--segment-gap", type=int, default=40)
    parser.add_argument("--overlap-long-side", type=int, default=384)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else 0.0


def validate_analysis(directory: Path) -> dict:
    config = json.loads((directory / "analysis_config.json").read_text(encoding="utf-8"))
    if config.get("split") != "val" or config.get("held_out_test_used") is not False:
        raise RuntimeError(f"Refusing analysis outside the valid-only contract: {directory}")
    if config.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError(f"Unexpected channel contract: {directory}")
    if config.get("validation_overlap_mask") is not False:
        raise RuntimeError(f"Expected independent overlapping-instance validation masks: {directory}")
    return config


def image_number(name: str) -> int:
    match = re.fullmatch(r"DJI_(\d+)\.JPG", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected image name: {name}")
    return int(match.group(1))


def flight_segments(names: list[str], maximum_gap: int) -> dict[str, str]:
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
        label = f"DJI_{first:04d}-{last:04d}"
        output.update({name: label for name in group})
    return output


def label_polygons(path: Path) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) <= 6:
            continue
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        polygons.append(points)
    return polygons


def overlap_features(label_path: Path, height: int, width: int, long_side: int) -> dict[str, float | int]:
    scale = long_side / max(height, width)
    out_h = max(1, round(height * scale))
    out_w = max(1, round(width * scale))
    union = np.zeros((out_h, out_w), dtype=np.uint8)
    overlap = np.zeros_like(union)
    instance = np.zeros_like(union)
    polygons = label_polygons(label_path)
    for polygon in polygons:
        points = np.rint(polygon * np.asarray([out_w - 1, out_h - 1])).astype(np.int32)
        instance.fill(0)
        cv2.fillPoly(instance, [points], 1)
        cv2.bitwise_or(overlap, cv2.bitwise_and(union, instance), dst=overlap)
        cv2.bitwise_or(union, instance, dst=union)
    union_pixels = int(union.sum())
    overlap_pixels = int(overlap.sum())
    return {
        "overlap_instances": len(polygons),
        "overlap_union_pixels": union_pixels,
        "overlap_pixels": overlap_pixels,
        "overlap_foreground_fraction": overlap_pixels / union_pixels if union_pixels else 0.0,
        "has_instance_overlap": int(overlap_pixels > 0),
    }


def aggregate(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int]:
    tp = sum(int(row[f"{prefix}_tp"]) for row in rows)
    fp = sum(int(row[f"{prefix}_fp"]) for row in rows)
    fn = sum(int(row[f"{prefix}_fn"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def delta_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    control = aggregate(rows, "control")
    candidate = aggregate(rows, "candidate")
    return {
        "images": len(rows),
        "control": control,
        "candidate": candidate,
        "delta": {
            "tp": int(candidate["tp"]) - int(control["tp"]),
            "fp": int(candidate["fp"]) - int(control["fp"]),
            "fn": int(candidate["fn"]) - int(control["fn"]),
            "precision": float(candidate["precision"]) - float(control["precision"]),
            "recall": float(candidate["recall"]) - float(control["recall"]),
            "f1": float(candidate["f1"]) - float(control["f1"]),
        },
        "improved_f1_images": sum(float(row["f1_delta"]) > 0 for row in rows),
        "degraded_f1_images": sum(float(row["f1_delta"]) < 0 for row in rows),
        "unchanged_f1_images": sum(float(row["f1_delta"]) == 0 for row in rows),
    }


def correlation(rows: list[dict[str, object]], x_key: str, y_key: str) -> float | None:
    x = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    if len(x) < 2 or math.isclose(float(x.std()), 0.0) or math.isclose(float(y.std()), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize_slices(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    slice_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        for name in str(row["scene_slices"]).split("|"):
            if name:
                slice_rows[name].append(row)
    return {name: delta_summary(selected) for name, selected in sorted(slice_rows.items())}


def summarize_segments(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["flight_segment"])].append(row)
    output = []
    for segment, selected in grouped.items():
        summary = delta_summary(selected)
        output.append({
            "flight_segment": segment,
            "images": summary["images"],
            "fp_delta": summary["delta"]["fp"],
            "fn_delta": summary["delta"]["fn"],
            "f1_delta": summary["delta"]["f1"],
            "mean_per_image_f1_delta": statistics.fmean(float(row["f1_delta"]) for row in selected),
        })
    return sorted(output, key=lambda row: (-int(row["fp_delta"]), float(row["f1_delta"]), str(row["flight_segment"])))


def main() -> int:
    args = parse_args()
    if args.segment_gap < 1 or args.overlap_long_side < 64:
        raise ValueError("Invalid segment or overlap rasterization settings")
    control_dir = args.control.resolve()
    candidate_dir = args.candidate.resolve()
    images = args.images.resolve()
    labels = args.labels.resolve()
    output = args.output.resolve()
    control_config = validate_analysis(control_dir)
    candidate_config = validate_analysis(candidate_dir)
    if control_config["data"] != candidate_config["data"] or control_config["confidence"] != candidate_config["confidence"]:
        raise RuntimeError("Control and candidate analysis configurations are not paired")
    control_rows = {row["image"]: row for row in read_csv(control_dir / "per_image_metrics.csv")}
    candidate_rows = {row["image"]: row for row in read_csv(candidate_dir / "per_image_metrics.csv")}
    image_names = sorted((path.name for path in images.glob("*.JPG")), key=image_number)
    if set(control_rows) != set(candidate_rows) or set(control_rows) != set(image_names):
        raise RuntimeError("Control, candidate, and valid image sets must match exactly")
    segments = flight_segments(image_names, args.segment_gap)
    records: list[dict[str, object]] = []
    for name in image_names:
        image_path = images / name
        label_path = labels / f"{Path(name).stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        scene = scene_features(image_path, label_path)
        overlap = overlap_features(label_path, image.shape[0], image.shape[1], args.overlap_long_side)
        base = control_rows[name]
        trial = candidate_rows[name]
        record: dict[str, object] = {
            "image": name,
            "image_number": image_number(name),
            "flight_segment": segments[name],
            **scene,
            **overlap,
        }
        for prefix, row in (("control", base), ("candidate", trial)):
            for metric in ("f1", "precision", "recall", "tp", "fp", "fn"):
                value = as_float(row, f"mask_{metric}")
                record[f"{prefix}_{metric}"] = int(round(value)) if metric in {"tp", "fp", "fn"} else value
        for metric in ("f1", "precision", "recall"):
            record[f"{metric}_delta"] = float(record[f"candidate_{metric}"]) - float(record[f"control_{metric}"])
        for metric in ("tp", "fp", "fn"):
            record[f"{metric}_delta"] = int(record[f"candidate_{metric}"]) - int(record[f"control_{metric}"])
        records.append(record)

    overlap_threshold = float(np.quantile([float(row["overlap_foreground_fraction"]) for row in records], 0.75))
    for row in records:
        slices = []
        if float(row["overlap_foreground_fraction"]) >= overlap_threshold:
            slices.append("high_instance_overlap")
        if float(row["foreground_coverage"]) < 0.15:
            slices.append("sparse_low_coverage")
        if float(row["background_edge_density"]) >= 0.15:
            slices.append("road_facility_edge_proxy")
        if float(row["background_luma_std"]) >= 0.18:
            slices.append("mixed_luminance_background")
        if int(row["polygon_count"]) == 0:
            slices.append("empty_label_background")
        row["scene_slices"] = "|".join(slices or ["general"])

    fields = list(records[0])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    fixed_threshold = delta_summary(records)
    worst_fp = sorted(records, key=lambda row: (-int(row["fp_delta"]), float(row["f1_delta"]), str(row["image"])))[:12]
    worst_f1 = sorted(records, key=lambda row: (float(row["f1_delta"]), -int(row["fp_delta"]), str(row["image"])))[:12]
    summary = {
        "version": "v13_precision_regression_localization_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_split": "valid",
        "held_out_test_used": False,
        "gate_overridden": False,
        "deployment_eligible": False,
        "confidence": control_config["confidence"],
        "mask_match_iou": 0.5,
        "channels": control_config["channels"],
        "fixed_threshold_summary": fixed_threshold,
        "overlap_high_slice_threshold": overlap_threshold,
        "slice_summary": summarize_slices(records),
        "flight_segment_summary": summarize_segments(records),
        "correlations": {
            "overlap_fraction_vs_fp_delta": correlation(records, "overlap_foreground_fraction", "fp_delta"),
            "foreground_coverage_vs_fp_delta": correlation(records, "foreground_coverage", "fp_delta"),
            "background_edge_density_vs_fp_delta": correlation(records, "background_edge_density", "fp_delta"),
            "background_luma_std_vs_fp_delta": correlation(records, "background_luma_std", "fp_delta"),
        },
        "worst_fp_increases": [
            {key: row[key] for key in ("image", "flight_segment", "scene_slices", "fp_delta", "fn_delta", "f1_delta")}
            for row in worst_fp
        ],
        "worst_f1_regressions": [
            {key: row[key] for key in ("image", "flight_segment", "scene_slices", "fp_delta", "fn_delta", "f1_delta")}
            for row in worst_f1
        ],
        "diagnosis": (
            "At confidence 0.25 the candidate does not reproduce the training-summary precision regression; "
            "run a preregistered confidence sweep before choosing a training-side correction."
        ),
        "v13_gate_status": "failed_and_unchanged",
    }
    (output / "paired_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(fixed_threshold, ensure_ascii=False, indent=2))
    print(f"paired={output / 'paired_per_image_metrics.csv'}")
    print(f"summary={output / 'paired_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
