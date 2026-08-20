"""Build an image-level label-correction queue from difficult-sample decisions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW = ROOT / "reports" / "Data_v3_difficult_sample_review_v1"
DEFAULT_OUTPUT = ROOT / "reports" / "Data_v3_difficult_sample_correction_v1"


def build_queue(review_dir: Path, output_dir: Path) -> dict:
    queue_path = review_dir / "review_queue.csv"
    state_path = review_dir / "review_state.json"
    with queue_path.open(encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    decisions = state.get("decisions", {})

    selected = []
    for row in review_rows:
        review = decisions.get(row["image"], {})
        if review.get("decision") != "needs_label_fix":
            continue
        selected.append(
            {
                **row,
                "review_decision": review["decision"],
                "review_note": review.get("note", ""),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0]) if selected else ["image", "stem", "split", "review_decision", "review_note"]
    with (output_dir / "correction_queue.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "version": "difficult_sample_correction_queue_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_review_dir": str(review_dir.resolve()),
        "source_total": len(review_rows),
        "source_completed": sum(1 for row in review_rows if row["image"] in decisions),
        "selected_for_correction": len(selected),
        "split_counts": {
            split: sum(1 for row in selected if row.get("split") == split)
            for split in sorted({row.get("split", "") for row in selected})
        },
        "held_out_test_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_queue(args.review_dir.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
