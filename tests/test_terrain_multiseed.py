from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_terrain_multiseed import resolve_candidate
from summarize_terrain_model_screen import experiment_name
from summarize_terrain_multiseed import render_html, summarize_multiseed


def write_result(project: Path, group: str, model: str, seed: int, score: float, attention: str = "none") -> None:
    path = project / experiment_name(group, model, seed, attention) / "results.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "metrics/recall(M)"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "metrics/mAP50-95(M)": score,
                "metrics/mAP50(M)": score + 0.2,
                "metrics/precision(M)": 0.7,
                "metrics/recall(M)": 0.6,
            }
        )


class TerrainMultiseedSummaryTest(unittest.TestCase):
    def test_stable_two_of_three_candidate_is_promoted(self) -> None:
        model = "yolo12n-seg.pt"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for seed, delta in enumerate((0.006, 0.005, -0.001)):
                write_result(project, "current_5ch", model, seed, 0.30)
                write_result(project, "slope_6ch", model, seed, 0.30 + delta)
            summary = summarize_multiseed(
                project,
                model=model,
                candidate_group="slope_6ch",
                seeds=(0, 1, 2),
                attention="none",
            )
        self.assertTrue(summary["stable_for_fusion"])
        self.assertEqual(summary["positive_seeds"], 2)
        self.assertEqual(summary["decision"], "promote_to_fusion")
        self.assertIn("可进入模型融合", render_html(summary))

    def test_large_single_seed_regression_rejects_candidate(self) -> None:
        model = "yolo26n-seg.pt"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for seed, delta in enumerate((0.02, 0.01, -0.01)):
                write_result(project, "current_5ch", model, seed, 0.30)
                write_result(project, "slope_6ch", model, seed, 0.30 + delta)
            summary = summarize_multiseed(
                project,
                model=model,
                candidate_group="slope_6ch",
                seeds=(0, 1, 2),
                attention="none",
            )
        self.assertFalse(summary["stable_for_fusion"])
        self.assertEqual(summary["decision"], "reject_unstable")

    def test_candidate_can_be_loaded_from_stage_one_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "attention": "ca_eca",
                        "recommended_next": {"model": "yolov8n-seg.pt", "candidate_group": "slope_6ch"},
                    }
                ),
                encoding="utf-8",
            )
            result = resolve_candidate(None, None, None, summary_path)
        self.assertEqual(result, ("yolov8n-seg.pt", "slope_6ch", "ca_eca"))


if __name__ == "__main__":
    unittest.main()

