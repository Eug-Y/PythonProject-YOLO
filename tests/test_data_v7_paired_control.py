from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from run_data_v7_soil_vegetation_screen import experiment_name
from summarize_data_v7_paired_control import CONTROL_NAME, render_html, summarize_paired


def write_result(path: Path, score: float, precision: float = 0.7, recall: float = 0.57) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "metrics/recall(M)"],
        )
        writer.writeheader()
        writer.writerow({
            "epoch": 1,
            "metrics/mAP50-95(M)": score,
            "metrics/mAP50(M)": score + 0.2,
            "metrics/precision(M)": precision,
            "metrics/recall(M)": recall,
        })


class DataV7PairedControlTest(unittest.TestCase):
    def test_exact_budget_candidate_can_promote(self) -> None:
        group = "add_mcari_6ch"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            control = project / CONTROL_NAME
            write_result(control / "results.csv", 0.31, recall=0.57)
            (control / "weights").mkdir(parents=True)
            (control / "weights" / "best.pt").touch()
            candidate = project / experiment_name(group, 0)
            write_result(candidate / "results.csv", 0.315, recall=0.565)
            (candidate / "weights").mkdir(parents=True)
            (candidate / "weights" / "best.pt").touch()
            summary = summarize_paired(project, (group,))
        self.assertTrue(summary["rows"][0]["promoted"])
        self.assertEqual(summary["decision"], "promote_best_to_three_seed")
        self.assertIn("冻结 test 未读取", render_html(summary))

    def test_recall_gate_rejects_map_gain(self) -> None:
        group = "add_re_bsi_proxy_6ch"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            control = project / CONTROL_NAME
            write_result(control / "results.csv", 0.31, recall=0.58)
            (control / "weights").mkdir(parents=True)
            (control / "weights" / "best.pt").touch()
            candidate = project / experiment_name(group, 0)
            write_result(candidate / "results.csv", 0.32, recall=0.55)
            (candidate / "weights").mkdir(parents=True)
            (candidate / "weights" / "best.pt").touch()
            summary = summarize_paired(project, (group,))
        self.assertFalse(summary["rows"][0]["promoted"])


if __name__ == "__main__":
    unittest.main()
