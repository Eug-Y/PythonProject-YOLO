import unittest

from run_data_v28_yolo11_hard_examples import promotion_gate


class RunDataV28Yolo11HardExamplesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = {
            "f1": 0.64,
            "precision": 0.65,
            "recall": 0.64,
            "mean_matched_iou": 0.84,
        }

    def test_promotion_gate_requires_all_locked_deltas(self) -> None:
        candidate = {
            "f1": 0.655,
            "precision": 0.65,
            "recall": 0.63,
            "mean_matched_iou": 0.835,
        }
        result = promotion_gate(candidate, self.baseline)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_promotion_gate_rejects_precision_regression(self) -> None:
        candidate = {
            "f1": 0.67,
            "precision": 0.649,
            "recall": 0.64,
            "mean_matched_iou": 0.84,
        }
        result = promotion_gate(candidate, self.baseline)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["precision_not_lower"])


if __name__ == "__main__":
    unittest.main()
