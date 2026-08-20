"""Build a focused contour-review queue from model errors and the current label audit."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from multispectral import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v3_recommended_model_errors" / "per_image_metrics.csv",
    )
    parser.add_argument(
        "--issues",
        type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v3_label_quality_post_training" / "issues.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v3_model_error_review",
    )
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    metrics_path = args.metrics.resolve()
    issues_path = args.issues.resolve()
    output = args.output.resolve()
    metrics = read_rows(metrics_path)[: args.top]
    metric_by_stem = {Path(row["image"]).stem: row for row in metrics}
    stems = set(metric_by_stem)

    grouped: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in read_rows(issues_path):
        if row["stem"] not in stems or not row.get("line"):
            continue
        if row["code"] not in {"self_intersection", "duplicate_vertex", "tiny_polygon"}:
            continue
        grouped[(row["split"], row["stem"], int(row["line"]))].add(row["code"])

    geometry_rows: list[dict[str, object]] = []
    issue_lines_by_stem: dict[str, list[str]] = defaultdict(list)
    for (split, stem, line), codes in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2])):
        joined_codes = ",".join(sorted(codes))
        label_path = PROJECT_ROOT / "datasets" / "Data_v3" / split / "labels" / f"{stem}.txt"
        geometry_rows.append(
            {
                "stem": stem,
                "split": split,
                "line": line,
                "codes": joined_codes,
                "label_path": str(label_path),
                "crop_path": "",
            }
        )
        issue_lines_by_stem[stem].append(f"{joined_codes}:{line}")

    queue_rows: list[dict[str, object]] = []
    for rank, metric in enumerate(metrics, 1):
        stem = Path(metric["image"]).stem
        issue_lines = issue_lines_by_stem.get(stem, [])
        if not issue_lines:
            continue
        split = next(row["split"] for row in geometry_rows if row["stem"] == stem)
        queue_rows.append(
            {
                "priority_rank": rank,
                "priority": "P1-model-error",
                "source": "base",
                "split": split,
                "stem": stem,
                "mask_f1": metric["mask_f1"],
                "mask_fp": metric["mask_fp"],
                "mask_fn": metric["mask_fn"],
                "issue_count": len(issue_lines),
                "issue_lines": ";".join(issue_lines),
            }
        )

    write_rows(
        output / "review_queue.csv",
        [
            "priority_rank",
            "priority",
            "source",
            "split",
            "stem",
            "mask_f1",
            "mask_fp",
            "mask_fn",
            "issue_count",
            "issue_lines",
        ],
        queue_rows,
    )
    write_rows(
        output / "geometry_review.csv",
        ["stem", "split", "line", "codes", "label_path", "crop_path"],
        geometry_rows,
    )
    print(f"images={len(queue_rows)}")
    print(f"contours={len(geometry_rows)}")
    print(f"queue={output / 'review_queue.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
