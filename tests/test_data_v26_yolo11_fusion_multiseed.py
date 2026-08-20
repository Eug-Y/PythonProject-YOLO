import unittest

from run_data_v26_yolo11_fusion_multiseed import aggregate, protocol


class DataV26FusionMultiseedTests(unittest.TestCase):
    def test_protocol_keeps_yolo11_required_and_test_sealed(self) -> None:
        value = protocol()
        self.assertIn("every output requires YOLO11", value["architecture_requirement"])
        self.assertFalse(value["held_out_test_used"])

    def test_aggregate_accepts_stable_three_seed_gain(self) -> None:
        rows = [
            {"f1_delta_vs_yolo11": 0.01, "f1_delta_vs_yolov8": 0.006,
             "precision_delta_vs_better": 0.01, "recall_delta_vs_better": -0.005,
             "iou_delta_vs_better": -0.003}
            for _ in range(3)
        ]
        self.assertTrue(aggregate(rows)["passed"])

    def test_aggregate_rejects_unstable_worst_seed(self) -> None:
        rows = [
            {"f1_delta_vs_yolo11": value, "f1_delta_vs_yolov8": value,
             "precision_delta_vs_better": 0.0, "recall_delta_vs_better": 0.0,
             "iou_delta_vs_better": 0.0}
            for value in (0.02, 0.02, -0.01)
        ]
        self.assertFalse(aggregate(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
