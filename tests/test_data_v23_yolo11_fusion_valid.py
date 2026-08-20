import unittest

from run_data_v23_yolo11_fusion_valid import gate, protocol


def payload(fusion_f1=0.7, primary_f1=0.69, auxiliary_f1=0.68):
    return {
        "locked_full_resolution_metrics": {
            "fusion": {"f1": fusion_f1, "precision": 0.72, "recall": 0.68, "mean_matched_iou": 0.75},
            "primary_yolo11": {"f1": primary_f1, "precision": 0.71, "recall": 0.67, "mean_matched_iou": 0.74},
            "auxiliary_yolov8": {"f1": auxiliary_f1, "precision": 0.70, "recall": 0.66, "mean_matched_iou": 0.73},
        }
    }


class DataV23Yolo11FusionTests(unittest.TestCase):
    def test_protocol_makes_yolo11_primary_and_never_reads_test(self) -> None:
        locked = protocol()
        self.assertIn("YOLO11 must be primary", locked["architecture_requirement"])
        self.assertFalse(locked["held_out_test_used"])
        self.assertFalse(locked["release_blind_holdout_used"])

    def test_gate_accepts_fusion_that_beats_both_models(self) -> None:
        self.assertTrue(gate(payload())["passed"])

    def test_gate_rejects_fusion_below_auxiliary(self) -> None:
        self.assertFalse(gate(payload(fusion_f1=0.67, primary_f1=0.66, auxiliary_f1=0.68))["passed"])


if __name__ == "__main__":
    unittest.main()
