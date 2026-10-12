import unittest

from run_data_v20_final_test import upgrade_decision


class DataV20FinalTestTests(unittest.TestCase):
    def test_upgrade_requires_aggregate_and_fixed_threshold_gates(self) -> None:
        metrics = {
            "metrics/mAP50-95(M)": 0.01,
            "metrics/mAP50(M)": 0.0,
            "metrics/recall(M)": 0.0,
        }
        fixed = {"f1": 0.01, "precision": 0.0}
        self.assertTrue(upgrade_decision(metrics, fixed))
        fixed["f1"] = -0.0001
        self.assertFalse(upgrade_decision(metrics, fixed))

    def test_upgrade_rejects_insufficient_map_gain(self) -> None:
        metrics = {
            "metrics/mAP50-95(M)": 0.002,
            "metrics/mAP50(M)": 0.0,
            "metrics/recall(M)": 0.0,
        }
        self.assertFalse(upgrade_decision(metrics, {"f1": 0.01, "precision": 0.0}))


if __name__ == "__main__":
    unittest.main()
