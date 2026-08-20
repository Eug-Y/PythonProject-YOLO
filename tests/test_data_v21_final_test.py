import unittest

from run_data_v21_final_test import upgrade_decision


class DataV21FinalTestTests(unittest.TestCase):
    def test_upgrade_requires_all_preregistered_gates(self) -> None:
        metrics = {
            "metrics/mAP50-95(M)": 0.00201,
            "metrics/mAP50(M)": -0.002,
            "metrics/recall(M)": -0.01,
        }
        fixed = {"f1": 0.0, "precision": -0.01}
        self.assertTrue(upgrade_decision(metrics, fixed))

        metrics["metrics/mAP50-95(M)"] = 0.002
        self.assertFalse(upgrade_decision(metrics, fixed))


if __name__ == "__main__":
    unittest.main()
