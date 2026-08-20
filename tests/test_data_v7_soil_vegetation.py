from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_data_v7_soil_vegetation_screen import (
    DATA_BY_CHANNELS,
    GROUPS,
    INITIALIZATION,
    training_command,
)
from spectral_indices import INDEX_REGISTRY
from summarize_data_v7_soil_vegetation_screen import experiment_name, render_html, summarize


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data_v7_soil_vegetation_channels.json"
GROUP_CONFIG = ROOT / "configs" / "feature_ablation_groups_v7_soil_vegetation.json"


def write_results(path: Path, values: list[float]) -> None:
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
        for epoch, value in enumerate(values):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50-95(M)": value,
                    "metrics/mAP50(M)": value + 0.2,
                    "metrics/precision(M)": 0.70,
                    "metrics/recall(M)": 0.58,
                }
            )


class DataV7ConfigurationTest(unittest.TestCase):
    def test_candidate_contract_has_only_the_requested_seven_channels(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            config["channels"],
            ["R", "G", "B", "NDVI", "NDRE", "RE_BSI_PROXY", "MCARI"],
        )
        self.assertTrue(set(config["channels"][3:]).issubset(INDEX_REGISTRY))
        self.assertFalse(config["feature_policy"]["terrain_channels_included"])
        self.assertFalse(config["feature_policy"]["slope_included"])

    def test_soil_proxy_is_explicitly_nonstandard_and_mcari_is_band_adapted(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(config["feature_policy"]["soil_candidate"]["standard_bsi"])
        self.assertFalse(INDEX_REGISTRY["RE_BSI_PROXY"].standard)
        self.assertIn("not the standard", INDEX_REGISTRY["RE_BSI_PROXY"].description)
        self.assertEqual(INDEX_REGISTRY["MCARI"].category, "rededge_chlorophyll")
        self.assertIn("P4M", INDEX_REGISTRY["MCARI"].caveat or "")

    def test_groups_isolate_soil_vegetation_and_combined_effects(self) -> None:
        groups = json.loads(GROUP_CONFIG.read_text(encoding="utf-8"))["groups"]
        self.assertEqual(tuple(group for group in groups if group != "current_5ch"), GROUPS)
        self.assertEqual(groups["add_re_bsi_proxy_6ch"][-1], "RE_BSI_PROXY")
        self.assertEqual(groups["add_mcari_6ch"][-1], "MCARI")
        self.assertEqual(groups["soil_vegetation_7ch"][-2:], ["RE_BSI_PROXY", "MCARI"])
        for name, channels in groups.items():
            self.assertEqual(channels[:5], ["R", "G", "B", "NDVI", "NDRE"], name)
            self.assertFalse(any(token in channel for token in ("SLOPE", "DEM", "DSM", "DTM") for channel in channels))

    def test_selection_yamls_are_train_valid_only(self) -> None:
        for channels, path in DATA_BY_CHANNELS.items():
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("test:", text)
            self.assertIn(f"channels: {channels}", text)
            self.assertIn("teacher_refined", text)

    def test_runner_uses_shared_five_channel_initialization(self) -> None:
        cli = argparse.Namespace(
            seed=0,
            epochs=30,
            patience=12,
            imgsz=1024,
            batch=2,
            device="0",
        )
        command = training_command(cli, "soil_vegetation_7ch", 7)
        self.assertIn(str(INITIALIZATION), command)
        self.assertIn("--init-checkpoint", command)
        self.assertIn("--attention", command)
        self.assertEqual(command[command.index("--attention") + 1], "none")
        self.assertEqual(command[command.index("--data") + 1], str(DATA_BY_CHANNELS[7]))
        self.assertNotIn("test", " ".join(command).lower())


class DataV7SummaryTest(unittest.TestCase):
    def test_promotion_uses_locked_five_channel_seed0_baseline(self) -> None:
        group = "add_mcari_6ch"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run = project / experiment_name(group, 0)
            write_results(run / "results.csv", [0.36, 0.374])
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "best.pt").touch()
            summary = summarize(project, (group,), seed=0, promotion_delta=0.003)
        row = summary["rows"][0]
        self.assertAlmostEqual(float(row["delta_vs_5ch_pp"]), 0.432)
        self.assertTrue(row["promoted"])
        self.assertEqual(summary["recommended_next"]["group"], group)
        page = render_html(summary)
        self.assertIn("冻结 test 未读取", page)
        self.assertIn("本轮不含坡度", page)

    def test_missing_result_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = summarize(Path(directory), ("add_re_bsi_proxy_6ch",))
        self.assertEqual(summary["decision"], "retain_current_5ch")
        self.assertIsNone(summary["recommended_next"])


if __name__ == "__main__":
    unittest.main()
