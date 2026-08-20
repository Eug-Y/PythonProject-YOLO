from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from run_data_v13_yolo11_overlap_multiseed import (
    candidate_name,
    control_name,
    summarize,
    training_command,
)


def write_result(path: Path, score: float, precision: float = 0.7, recall: float = 0.6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "metrics/recall(M)"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "metrics/mAP50-95(M)": score, "metrics/mAP50(M)": score + 0.2, "metrics/precision(M)": precision, "metrics/recall(M)": recall})
    (path.parent / "weights").mkdir(exist_ok=True)
    (path.parent / "weights" / "best.pt").touch()


def add_pair(project: Path, seed: int, delta: float, precision_delta: float = 0.0, recall_delta: float = 0.0) -> None:
    write_result(project / control_name(seed) / "results.csv", 0.35)
    write_result(project / candidate_name(seed) / "results.csv", 0.35 + delta, 0.7 + precision_delta, 0.6 + recall_delta)


class Yolo11OverlapConfirmationTests(unittest.TestCase):
    def test_command_is_locked_yolo11_attention_pair(self) -> None:
        cli = argparse.Namespace(epochs=60, patience=20, imgsz=1024, batch=2, device="0", workers=2)
        command = training_command(cli, 0)
        self.assertEqual(command[command.index("--base-model") + 1], "yolo11n-seg.pt")
        self.assertEqual(command[command.index("--attention") + 1], "ca_eca")
        self.assertEqual(command[command.index("--imgsz") + 1], "1024")
        self.assertIn("--no-overlap-mask", command)
        self.assertIn("--deterministic", command)
        self.assertNotIn("test", " ".join(command).lower())

    def test_seed0_gate_stops_small_gain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            add_pair(project, 0, 0.001)
            summary = summarize(project, deployment=None)
        self.assertFalse(summary["seed0_passed"])
        self.assertEqual(summary["decision"], "reject_seed0_retain_current_deployment")

    def test_seed0_gate_includes_precision_and_recall_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            add_pair(project, 0, 0.01, precision_delta=-0.02)
            summary = summarize(project, deployment=None)
        self.assertFalse(summary["seed0_passed"])

    def test_stable_three_seed_gain_freezes_candidate_without_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for seed, delta in enumerate((0.006, 0.005, -0.001)):
                add_pair(project, seed, delta)
            summary = summarize(project, deployment=None)
        self.assertTrue(summary["seed0_passed"])
        self.assertTrue(summary["gate_passed"])
        self.assertEqual(summary["decision"], "freeze_nooverlap_candidate_for_review_no_auto_deployment")
        self.assertFalse(summary["deployment_write_allowed"])

    def test_worst_seed_blocks_final_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for seed, delta in enumerate((0.02, 0.02, -0.01)):
                add_pair(project, seed, delta)
            summary = summarize(project, deployment=None)
        self.assertTrue(summary["seed0_passed"])
        self.assertFalse(summary["gate_passed"])
        self.assertEqual(summary["decision"], "reject_multiseed_retain_current_deployment")


if __name__ == "__main__":
    unittest.main()
