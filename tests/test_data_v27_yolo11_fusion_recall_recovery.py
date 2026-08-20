import unittest

from run_data_v27_yolo11_fusion_recall_recovery import configs, protocol, row_for_gate


class DataV27FusionRecallRecoveryTests(unittest.TestCase):
    def test_protocol_keeps_test_and_blind_holdout_sealed(self) -> None:
        value = protocol()
        self.assertFalse(value["held_out_test_used"])
        self.assertFalse(value["release_blind_holdout_used"])

    def test_every_candidate_requires_yolo11_and_equal_weights(self) -> None:
        candidates = configs()
        self.assertEqual(len(candidates), 15)
        self.assertTrue(all(row["required_model_indices"] == [0] for row in candidates))
        self.assertTrue(all(row["model_weights"] == [1.0, 1.0] for row in candidates))
        self.assertTrue(all(row["mask_vote"] == 0.5 for row in candidates))

    def test_row_for_gate_uses_better_standalone_guards(self) -> None:
        metrics = {
            "fusion": {"f1": 0.7, "precision": 0.8, "recall": 0.6, "mean_matched_iou": 0.85},
            "primary_yolo11": {"f1": 0.65, "precision": 0.7, "recall": 0.62, "mean_matched_iou": 0.8},
            "auxiliary_yolov8": {"f1": 0.66, "precision": 0.72, "recall": 0.61, "mean_matched_iou": 0.84},
        }
        row = row_for_gate(2, metrics)
        self.assertAlmostEqual(row["recall_delta_vs_better"], -0.02)
        self.assertAlmostEqual(row["precision_delta_vs_better"], 0.08)
        self.assertAlmostEqual(row["iou_delta_vs_better"], 0.01)


if __name__ == "__main__":
    unittest.main()
