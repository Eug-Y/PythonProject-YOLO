"""Build a train/valid-only difficult-sample review queue from per-image errors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS = ROOT / "reports" / "Data_v3_deployment_valid_errors_v1" / "per_image_metrics.csv"
DEFAULT_ANALYSIS = ROOT / "reports" / "Data_v3_deployment_valid_errors_v1" / "analysis_config.json"
DEFAULT_ISSUES = ROOT / "reports" / "Data_v3_label_quality_trainval_p2_reviewed" / "issues.csv"
DEFAULT_VALID_IMAGES = ROOT / "datasets" / "Data_v3" / "valid" / "images"
DEFAULT_OUTPUT = ROOT / "reports" / "Data_v3_difficult_sample_review_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    parser.add_argument("--valid-images", type=Path, default=DEFAULT_VALID_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def build_queue(
    metrics_path: Path,
    analysis_path: Path,
    issues_path: Path,
    valid_images: Path,
    output: Path,
    limit: int,
) -> list[dict[str, object]]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("split") != "val":
        raise ValueError(f"Refusing non-validation metrics: split={analysis.get('split')!r}")

    metrics = read_csv(metrics_path)
    if not metrics:
        raise ValueError(f"No per-image metrics: {metrics_path}")
    image_names = sorted(path.name for path in valid_images.glob("*.JPG"))
    batch_index = {name: idx for idx, name in enumerate(image_names)}
    if set(row["image"] for row in metrics) != set(image_names):
        raise ValueError("Validation image names and per-image metrics do not match exactly")

    warning_codes: dict[str, list[str]] = defaultdict(list)
    warning_count: dict[str, int] = defaultdict(int)
    if issues_path.is_file():
        for row in read_csv(issues_path):
            if row.get("split") != "valid":
                continue
            stem = row.get("stem", "")
            code = row.get("code", "")
            if not stem or not code:
                continue
            warning_count[stem] += 1
            warning_codes[stem].append(code)

    by_f1 = sorted(metrics, key=lambda row: (as_float(row, "mask_f1"), row["image"]))
    by_fp = sorted(metrics, key=lambda row: (-as_float(row, "mask_fp"), row["image"]))
    by_fn = sorted(metrics, key=lambda row: (-as_float(row, "mask_fn"), row["image"]))
    reason_by_image: dict[str, set[str]] = defaultdict(set)
    for row in by_f1[:8]:
        reason_by_image[row["image"]].add("低 F1")
    for row in by_fp[:6]:
        reason_by_image[row["image"]].add("高误检 FP")
    for row in by_fn[:6]:
        reason_by_image[row["image"]].add("高漏检 FN")

    max_fp = max(as_float(row, "mask_fp") for row in metrics) or 1.0
    max_fn = max(as_float(row, "mask_fn") for row in metrics) or 1.0

    def score(row: dict[str, str]) -> float:
        return 100.0 * (
            0.50 * (1.0 - as_float(row, "mask_f1"))
            + 0.25 * as_float(row, "mask_fp") / max_fp
            + 0.25 * as_float(row, "mask_fn") / max_fn
        )

    selected = [row for row in metrics if row["image"] in reason_by_image]
    selected.sort(key=lambda row: (-score(row), row["image"]))
    if len(selected) < limit:
        selected_names = {row["image"] for row in selected}
        for row in sorted(metrics, key=lambda item: (-score(item), item["image"])):
            if row["image"] in selected_names:
                continue
            reason_by_image[row["image"]].add("综合困难度")
            selected.append(row)
            selected_names.add(row["image"])
            if len(selected) >= limit:
                break
    selected.sort(key=lambda row: (-score(row), row["image"]))
    selected = selected[:limit]

    queue: list[dict[str, object]] = []
    for rank, row in enumerate(selected, 1):
        stem = Path(row["image"]).stem
        codes = warning_codes.get(stem, [])
        queue.append(
            {
                "rank": rank,
                "image": row["image"],
                "stem": stem,
                "split": "valid",
                "batch_index": batch_index[row["image"]],
                "mask_f1": round(as_float(row, "mask_f1"), 6),
                "mask_precision": round(as_float(row, "mask_precision"), 6),
                "mask_recall": round(as_float(row, "mask_recall"), 6),
                "mask_fp": int(as_float(row, "mask_fp")),
                "mask_fn": int(as_float(row, "mask_fn")),
                "difficulty_score": round(score(row), 2),
                "reasons": ", ".join(sorted(reason_by_image[row["image"]])),
                "label_warning_count": warning_count.get(stem, 0),
                "label_warning_codes": ", ".join(sorted(set(codes))),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(queue[0])
    with (output / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queue)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "version": "data_v3_difficult_sample_review_v1",
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "source_split": "valid",
                "held_out_test_used": False,
                "checkpoint": analysis.get("checkpoint"),
                "imgsz": analysis.get("imgsz"),
                "confidence": analysis.get("confidence"),
                "validation_images": len(metrics),
                "selected_images": len(queue),
                "selection": {"low_f1": 8, "high_fp": 6, "high_fn": 6, "deduplicated_limit": limit},
                "review_state": str(output / "review_state.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return queue


def main() -> int:
    args = parse_args()
    queue = build_queue(
        args.metrics.resolve(),
        args.analysis.resolve(),
        args.issues.resolve(),
        args.valid_images.resolve(),
        args.output.resolve(),
        args.limit,
    )
    print(f"selected={len(queue)}")
    print(f"queue={args.output.resolve() / 'review_queue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
