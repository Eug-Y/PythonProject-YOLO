from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from run_data_v9_grouped_cv import (
    FEATURE_ROOT,
    VARIANTS,
    data_yaml,
    experiment_name,
    summarize,
    training_command,
)


def write_result(project: Path, fold: int, variant: str, score: float) -> None:
    run_dir = project / experiment_name(fold, variant)
    path = run_dir / "results.csv"
    path.parent.mkdir(parents=True)
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
                "epoch": 0,
                "metrics/mAP50-95(M)": score,
                "metrics/mAP50(M)": score + 0.2,
                "metrics/precision(M)": 0.7,
                "metrics/recall(M)": 0.6,
            }
        )
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").touch()


class GroupedCVRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cli = argparse.Namespace(
            epochs=15,
            patience=6,
            imgsz=640,
            batch=4,
            device="0",
            seed=0,
        )

    def test_h0_h1_commands_differ_only_in_data_and_run_name(self) -> None:
        h0 = training_command(self.cli, 0, "h0_refined")
        h1 = training_command(self.cli, 0, "h1_human")
        self.assertEqual(h0[h0.index("--feature-root") + 1], str(FEATURE_ROOT))
        self.assertEqual(h0[h0.index("--data") + 1], str(data_yaml(0, "h0_refined")))
        self.assertEqual(h1[h1.index("--data") + 1], str(data_yaml(0, "h1_human")))
        self.assertNotIn("test", " ".join(h0).lower())
        self.assertNotIn("test", " ".join(h1).lower())

        def without_pair_identity(command: list[str]) -> list[str]:
            copy = list(command)
            for option in ("--data", "--name"):
                index = copy.index(option)
                del copy[index : index + 2]
            return copy

        self.assertEqual(without_pair_identity(h0), without_pair_identity(h1))

    def test_mean_and_worst_fold_gate_promotes_stable_h1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for fold, delta in enumerate((0.004, 0.003, -0.001)):
                write_result(project, fold, VARIANTS[0], 0.30)
                write_result(project, fold, VARIANTS[1], 0.30 + delta)
            summary = summarize(project, deployment=None)
        self.assertTrue(summary["all_complete"])
        self.assertAlmostEqual(summary["mean_delta"], 0.002)
        self.assertAlmostEqual(summary["worst_fold_delta"], -0.001)
        self.assertTrue(summary["promote_h1_to_yolo11_seed0"])

    def test_worst_fold_regression_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for fold, delta in enumerate((0.010, 0.010, -0.010)):
                write_result(project, fold, VARIANTS[0], 0.30)
                write_result(project, fold, VARIANTS[1], 0.30 + delta)
            summary = summarize(project, deployment=None)
        self.assertGreater(summary["mean_delta"], summary["mean_gate"])
        self.assertLess(summary["worst_fold_delta"], summary["worst_gate"])
        self.assertFalse(summary["promote_h1_to_yolo11_seed0"])

    def test_incomplete_fold_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            write_result(project, 0, VARIANTS[0], 0.30)
            summary = summarize(project, deployment=None)
        self.assertFalse(summary["all_complete"])
        self.assertEqual(summary["decision"], "incomplete")


if __name__ == "__main__":
    unittest.main()
