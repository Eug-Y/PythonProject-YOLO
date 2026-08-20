from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_data_v10_grouped_feature_cv import (
    CANDIDATES,
    FEATURE_ROOT,
    data_yaml,
    experiment_name,
    select_candidate,
    summarize,
    training_command,
)
from run_data_v9_grouped_cv import experiment_name as baseline_experiment_name


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "generated" / "data_v10_grouped_feature_cv" / "contract.json"


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


def add_fold(project: Path, fold: int, group: str, baseline: float, delta: float) -> None:
    write_result(project / baseline_experiment_name(fold, "h0_refined") / "results.csv", baseline)
    write_result(project / experiment_name(fold, group) / "results.csv", baseline + delta)


class DataV10ContractTests(unittest.TestCase):
    def test_contract_keeps_test_sealed_and_shared_channels_exact(self) -> None:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertFalse(payload["held_out_test_used"])
        self.assertFalse(payload["test_paths_enumerated"])
        self.assertEqual(payload["training_label_policy"], "h0_teacher_refined_mixed")
        self.assertTrue(payload["feature_pairing"]["exact_array_match"])
        self.assertEqual(payload["feature_pairing"]["checked_arrays"], 270)
        for fold in payload["fold_contracts"]:
            for key, channels in (("data_6ch", 6), ("data_7ch", 7)):
                text = Path(fold[key]).read_text(encoding="utf-8").lower()
                self.assertNotIn("test:", text)
                self.assertIn(f"channels: {channels}", text)

    def test_commands_use_paired_grouped_cv_contract(self) -> None:
        cli = argparse.Namespace(epochs=15, patience=6, imgsz=640, batch=4, device="0", seed=0)
        for group in CANDIDATES:
            command = training_command(cli, 0, group)
            self.assertEqual(command[command.index("--feature-root") + 1], str(FEATURE_ROOT))
            channels = 7 if group == "soil_vegetation_7ch" else 6
            self.assertEqual(command[command.index("--data") + 1], str(data_yaml(0, channels)))
            self.assertNotIn("test", " ".join(command).lower())
            self.assertEqual(command[command.index("--attention") + 1], "none")


class DataV10SelectionTests(unittest.TestCase):
    def test_fold0_selects_best_candidate_above_worst_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for group, delta in zip(CANDIDATES, (-0.006, 0.003, 0.001)):
                add_fold(project, 0, group, 0.30, delta)
            selected, rows = select_candidate(project)
        self.assertEqual(selected, "add_mcari_6ch")
        self.assertEqual(rows[0]["group"], "add_mcari_6ch")

    def test_three_fold_gate_requires_mean_and_worst_fold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            selected = "add_mcari_6ch"
            for group, delta in zip(CANDIDATES, (-0.006, 0.004, 0.001)):
                add_fold(project, 0, group, 0.30, delta)
            for fold, delta in ((1, 0.003), (2, -0.001)):
                add_fold(project, fold, selected, 0.30, delta)
            summary = summarize(project, deployment=None)
        self.assertEqual(summary["selected_candidate"], selected)
        self.assertAlmostEqual(summary["mean_delta"], 0.002)
        self.assertTrue(summary["promote_to_yolo11_seed0"])

    def test_no_fold0_candidate_above_guard_stops_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for group, delta in zip(CANDIDATES, (-0.006, -0.008, -0.010)):
                add_fold(project, 0, group, 0.30, delta)
            summary = summarize(project, deployment=None)
        self.assertIsNone(summary["selected_candidate"])
        self.assertEqual(summary["decision"], "retain_5ch_h0")


if __name__ == "__main__":
    unittest.main()
