from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from run_data_v11_mask_supervision_cv import data_yaml, experiment_name, summarize, training_command
from run_data_v9_grouped_cv import experiment_name as baseline_experiment_name


def write_result(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    (path.parent / "weights").mkdir(exist_ok=True)
    (path.parent / "weights" / "best.pt").touch()


def add_fold(project: Path, fold: int, delta: float) -> None:
    write_result(project / baseline_experiment_name(fold, "h0_refined") / "results.csv", 0.30)
    write_result(project / experiment_name(fold) / "results.csv", 0.30 + delta)


class MaskSupervisionRunnerTests(unittest.TestCase):
    def test_command_changes_mask_ratio_only_on_shared_fold_data(self) -> None:
        cli = argparse.Namespace(epochs=15, patience=6, imgsz=640, batch=4, device="0", seed=0)
        command = training_command(cli, 0)
        self.assertEqual(command[command.index("--mask-ratio") + 1], "2")
        self.assertEqual(command[command.index("--data") + 1], str(data_yaml(0)))
        self.assertNotIn("test", " ".join(command).lower())
        self.assertEqual(command[command.index("--attention") + 1], "none")

    def test_stable_three_fold_gain_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for fold, delta in enumerate((0.004, 0.003, -0.001)):
                add_fold(project, fold, delta)
            summary = summarize(project, deployment=None)
        self.assertTrue(summary["fold0_eligible"])
        self.assertAlmostEqual(summary["mean_delta"], 0.002)
        self.assertTrue(summary["promote_to_yolo11_seed0"])

    def test_fold0_large_regression_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            add_fold(project, 0, -0.006)
            summary = summarize(project, deployment=None)
        self.assertTrue(summary["stopped_after_fold0"])
        self.assertEqual(summary["decision"], "retain_mask_ratio_4")

    def test_worst_fold_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for fold, delta in enumerate((0.010, 0.010, -0.010)):
                add_fold(project, fold, delta)
            summary = summarize(project, deployment=None)
        self.assertGreater(summary["mean_delta"], summary["mean_gate"])
        self.assertFalse(summary["promote_to_yolo11_seed0"])


if __name__ == "__main__":
    unittest.main()
