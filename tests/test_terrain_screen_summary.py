from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from summarize_terrain_model_screen import experiment_name, render_html, summarize_pairs


def write_results(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "metrics/recall(M)"],
        )
        writer.writeheader()
        for epoch, value in enumerate(values):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50-95(M)": value,
                    "metrics/mAP50(M)": value + 0.2,
                    "metrics/precision(M)": 0.7,
                    "metrics/recall(M)": 0.6,
                }
            )


class TerrainScreenSummaryTest(unittest.TestCase):
    def test_paired_delta_and_promotion(self) -> None:
        model = "yolov8n-seg.pt"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            write_results(project / experiment_name("current_5ch", model, 0) / "results.csv", [0.30, 0.31])
            write_results(project / experiment_name("slope_6ch", model, 0) / "results.csv", [0.31, 0.318])
            summary = summarize_pairs(
                project,
                models=(model,),
                candidate_groups=("slope_6ch",),
                seed=0,
                promotion_delta=0.003,
            )
        row = summary["rows"][0]
        self.assertAlmostEqual(row["delta_pp"], 0.8)
        self.assertTrue(row["promoted"])
        self.assertEqual(summary["recommended_next"]["model"], model)
        page = render_html(summary)
        self.assertIn("YOLO", page.upper())
        self.assertIn("test 未读取", page)

    def test_missing_candidate_is_not_promoted(self) -> None:
        model = "yolo11n-seg.pt"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            write_results(project / experiment_name("current_5ch", model, 0) / "results.csv", [0.3])
            summary = summarize_pairs(
                project,
                models=(model,),
                candidate_groups=("slope_6ch",),
                seed=0,
                promotion_delta=0.003,
            )
        self.assertEqual(summary["rows"][0]["status"], "missing_candidate")
        self.assertIsNone(summary["recommended_next"])


if __name__ == "__main__":
    unittest.main()

