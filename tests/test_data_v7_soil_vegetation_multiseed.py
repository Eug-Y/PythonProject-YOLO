from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_data_v7_soil_vegetation_multiseed import resolve_candidate
from run_data_v7_soil_vegetation_screen import experiment_name
from summarize_data_v7_soil_vegetation_multiseed import (
    BASELINE_RUNS,
    render_html,
    summarize_multiseed,
)


def write_result(path: Path, score: float, precision: float = 0.70, recall: float = 0.58) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "metrics/mAP50-95(M)",
                "metrics/mAP50(M)",
                "metrics/precision(M)",
                "metrics/recall(M)",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "metrics/mAP50-95(M)": score,
                "metrics/mAP50(M)": score + 0.2,
                "metrics/precision(M)": precision,
                "metrics/recall(M)": recall,
            }
        )


class DataV7CandidateResolutionTest(unittest.TestCase):
    def test_promoted_screen_candidate_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "complete_groups": 3,
                        "expected_groups": 3,
                        "recommended_next": {"group": "soil_vegetation_7ch", "promoted": True},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_candidate(path), "soil_vegetation_7ch")

    def test_no_promoted_candidate_stops_confirmation_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps({"complete_groups": 3, "expected_groups": 3, "recommended_next": None}),
                encoding="utf-8",
            )
            self.assertIsNone(resolve_candidate(path))


class DataV7MultiseedSummaryTest(unittest.TestCase):
    def test_two_positive_seeds_and_stable_mean_pass(self) -> None:
        group = "add_mcari_6ch"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for seed, delta in enumerate((0.006, 0.005, -0.001)):
                baseline = 0.37
                write_result(project / BASELINE_RUNS[seed] / "results.csv", baseline, recall=0.57)
                run = project / experiment_name(group, seed)
                write_result(run / "results.csv", baseline + delta, recall=0.568)
                (run / "weights").mkdir(parents=True)
                (run / "weights" / "best.pt").touch()
            summary = summarize_multiseed(project, group)
        self.assertTrue(summary["stable_candidate"])
        self.assertEqual(summary["positive_seeds"], 2)
        self.assertEqual(summary["decision"], "promote_to_frozen_candidate_review")
        self.assertIn("冻结 test 未读取", render_html(summary))

    def test_recall_regression_blocks_promotion(self) -> None:
        group = "add_re_bsi_proxy_6ch"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for seed in (0, 1, 2):
                write_result(project / BASELINE_RUNS[seed] / "results.csv", 0.37, recall=0.58)
                run = project / experiment_name(group, seed)
                write_result(run / "results.csv", 0.38, recall=0.55)
                (run / "weights").mkdir(parents=True)
                (run / "weights" / "best.pt").touch()
            summary = summarize_multiseed(project, group)
        self.assertFalse(summary["stable_candidate"])
        self.assertEqual(summary["decision"], "reject_unstable")


if __name__ == "__main__":
    unittest.main()
