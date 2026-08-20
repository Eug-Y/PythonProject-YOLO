import unittest

from run_data_v19_final_test import extract_frozen_candidate, upgrade_decision


class DataV19FinalTestTests(unittest.TestCase):
    def test_extracts_top_level_v19_frozen_candidate(self) -> None:
        summary = {
            "candidate_checkpoint": "best.pt",
            "candidate_checkpoint_sha256": "abc",
            "rows": [{"seed": 0, "candidate_checkpoint": "best.pt"}],
        }
        self.assertEqual(
            extract_frozen_candidate(summary),
            {"seed": 0, "candidate_checkpoint": "best.pt", "candidate_checkpoint_sha256": "abc"},
        )

    def test_upgrade_rule_remains_strict(self) -> None:
        metrics = {
            "metrics/mAP50-95(M)": 0.00201,
            "metrics/mAP50(M)": -0.002,
            "metrics/recall(M)": -0.01,
        }
        self.assertTrue(upgrade_decision(metrics, {"f1": 0.0, "precision": -0.01}))


if __name__ == "__main__":
    unittest.main()
