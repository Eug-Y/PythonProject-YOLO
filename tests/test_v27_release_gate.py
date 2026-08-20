from __future__ import annotations

import unittest
from unittest.mock import patch

from promote_v27_production import build_production_config
from run_v27_release_blind_final import protocol


class V27ReleaseGateTests(unittest.TestCase):
    def test_blind_protocol_is_single_use_and_keeps_original_guards(self) -> None:
        value = protocol()
        self.assertEqual(value["evaluation_limit"], 1)
        self.assertFalse(value["configuration_search_allowed"])
        self.assertTrue(value["result_may_not_be_used_for_further_tuning"])
        self.assertTrue(value["every_output_requires_yolo11"])
        self.assertEqual(value["gate"]["fusion_f1_delta_vs_incumbent_min"], 0.0)
        self.assertEqual(value["gate"]["recall_delta_vs_incumbent_min"], -0.01)

    def test_promoted_config_remains_yolo11_primary_and_embeds_rollback(self) -> None:
        current = {"model_name": "incumbent", "checkpoint": "old.pt", "sha256": "oldhash"}
        shadow = {
            "model_name": "v27_yolo11_primary_fusion_v1",
            "release_candidate": "rc1",
            "primary": {"architecture": "YOLO11n-seg", "checkpoint": "p.pt", "sha256": "p"},
            "auxiliary": {"architecture": "YOLOv8n-seg", "checkpoint": "a.pt", "sha256": "a"},
            "fusion_config": "fusion.json",
            "fusion_config_sha256": "f",
            "fusion": {"score_threshold": 0.2, "unconfirmed_score_threshold": 0.225},
            "imgsz": 1024,
            "channels": ["R", "G", "B", "NDVI", "NDRE"],
        }
        with patch("promote_v27_production.sha256", return_value="blindhash"):
            promoted = build_production_config(current, shadow, {"gate_passed": True})
        self.assertEqual(promoted["deployment_type"], "strict_yolo11_primary_fusion")
        self.assertEqual(promoted["primary"]["architecture"], "YOLO11n-seg")
        self.assertEqual(promoted["auxiliary"]["architecture"], "YOLOv8n-seg")
        self.assertTrue(promoted["every_output_requires_yolo11"])
        self.assertEqual(promoted["rollback_model"], current)


if __name__ == "__main__":
    unittest.main()
