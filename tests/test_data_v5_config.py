from __future__ import annotations

import json
import unittest
from pathlib import Path

from spectral_indices import INDEX_REGISTRY
from run_data_v5_ablations import GROUP_ORDER
from run_data_v5_multiseed import GROUPS as MULTISEED_GROUPS, SEEDS as MULTISEED_SEEDS
from run_data_v5_combo import GROUP as COMBO_GROUP, SEEDS as COMBO_SEEDS
from run_data_v5_highres_screen import GROUPS as HIGHRES_GROUPS
from run_data_v5_final_training import GROUP as FINAL_GROUP, RUN_NAME as FINAL_RUN_NAME
from compare_data_v5_final import is_upgrade_eligible


ROOT = Path(__file__).resolve().parents[1]


class DataV5ConfigTest(unittest.TestCase):
    def test_compact_candidates_exclude_data_v4_low_contributors(self) -> None:
        config = json.loads((ROOT / "configs" / "data_v5_channels.json").read_text(encoding="utf-8"))
        channels = config["channels"]
        self.assertEqual(channels[:3], ["R", "G", "B"])
        self.assertEqual(len(channels), len(set(channels)))
        excluded = set(config["feature_policy"]["excluded_from_new_training_candidates"])
        self.assertTrue(excluded.isdisjoint(channels))
        self.assertTrue(set(channels[3:]).issubset(INDEX_REGISTRY))

    def test_ablation_groups_follow_candidate_contract(self) -> None:
        config = json.loads((ROOT / "configs" / "data_v5_channels.json").read_text(encoding="utf-8"))
        groups = json.loads(
            (ROOT / "configs" / "feature_ablation_groups_v5.json").read_text(encoding="utf-8")
        )["groups"]
        available = set(config["channels"])
        for name, channels in groups.items():
            self.assertEqual(channels[:3], ["R", "G", "B"], name)
            self.assertTrue(set(channels).issubset(available), name)

    def test_screened_out_features_are_not_scheduled_for_yolo(self) -> None:
        policy = json.loads(
            (ROOT / "configs" / "feature_ablation_groups_v5.json").read_text(encoding="utf-8")
        )
        scheduled = {channel for channels in policy["groups"].values() for channel in channels}
        screened_out = set(policy["statistical_screen"]["screened_out"])
        self.assertTrue(screened_out.isdisjoint(scheduled))

    def test_stage1_runner_contains_rgb_and_each_promoted_single_feature(self) -> None:
        policy = json.loads(
            (ROOT / "configs" / "feature_ablation_groups_v5.json").read_text(encoding="utf-8")
        )
        expected = {"rgb", *(f"add_{name.lower()}" for name in policy["statistical_screen"]["promoted"])}
        self.assertEqual(set(GROUP_ORDER), expected)

    def test_multiseed_runner_only_uses_stage1_winners(self) -> None:
        self.assertEqual(set(MULTISEED_GROUPS), {"rgb", "add_ndvi_std5", "add_mcari"})
        self.assertEqual(MULTISEED_SEEDS, (1, 2))

    def test_combo_runner_uses_the_two_stable_features(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "feature_ablation_groups_v5.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["groups"][COMBO_GROUP], ["R", "G", "B", "MCARI", "NDVI_STD5"])
        self.assertEqual(COMBO_SEEDS, (0, 1, 2))

    def test_highres_screen_compares_rgb_with_both_stable_singles(self) -> None:
        self.assertEqual(set(HIGHRES_GROUPS), {"rgb", "add_ndvi_std5", "add_mcari"})

    def test_formal_training_uses_highres_winner(self) -> None:
        self.assertEqual(FINAL_GROUP, "add_mcari")
        self.assertIn("rgb_mcari_1024", FINAL_RUN_NAME)

    def test_final_yaml_exposes_frozen_test_with_four_channels(self) -> None:
        text = (ROOT / "configs" / "data_v5_final_4ch.yaml").read_text(encoding="utf-8")
        self.assertIn("test: test/images", text)
        self.assertIn("channels: 4", text)

    def test_upgrade_rule_requires_map_gain_without_recall_regression(self) -> None:
        good = {
            "metrics/mAP50-95(M)": 0.003,
            "metrics/mAP50(M)": 0.0,
            "metrics/recall(M)": -0.005,
        }
        bad = {**good, "metrics/recall(M)": -0.02}
        self.assertTrue(is_upgrade_eligible(good))
        self.assertFalse(is_upgrade_eligible(bad))


if __name__ == "__main__":
    unittest.main()
