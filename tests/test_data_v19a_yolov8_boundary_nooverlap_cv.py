import argparse
import unittest

from run_data_v19a_yolov8_boundary_nooverlap_cv import gate_decision, training_command


class DataV19AYolov8BoundaryNoOverlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cli = argparse.Namespace(
            epochs=30,
            patience=12,
            imgsz=640,
            batch=4,
            workers=2,
            device="0",
            seed=0,
        )

    def test_training_pair_differs_only_in_augmentation_name(self) -> None:
        control = training_command(self.cli, 0, "control")
        candidate = training_command(self.cli, 0, "candidate")
        for required in ("--no-overlap-mask", "--deterministic", "--mask-ratio"):
            self.assertIn(required, control)
            self.assertIn(required, candidate)
        self.assertEqual(control[control.index("--mosaic") + 1], "1.0")
        self.assertEqual(control[control.index("--mixup") + 1], "0.1")
        self.assertEqual(control[control.index("--close-mosaic") + 1], "10")
        self.assertEqual(candidate[candidate.index("--mosaic") + 1], "0.5")
        self.assertEqual(candidate[candidate.index("--mixup") + 1], "0.0")
        self.assertEqual(candidate[candidate.index("--close-mosaic") + 1], "15")

    def test_gate_passes_stable_gain(self) -> None:
        rows = [
            {"map_delta": 0.004, "precision_delta": 0.001, "recall_delta": 0.001},
            {"map_delta": 0.003, "precision_delta": -0.002, "recall_delta": 0.000},
            {"map_delta": 0.001, "precision_delta": 0.000, "recall_delta": -0.002},
        ]
        self.assertTrue(gate_decision(rows)["passed"])

    def test_gate_rejects_bad_worst_fold(self) -> None:
        rows = [
            {"map_delta": 0.010, "precision_delta": 0.001, "recall_delta": 0.001},
            {"map_delta": 0.008, "precision_delta": 0.001, "recall_delta": 0.001},
            {"map_delta": -0.006, "precision_delta": 0.001, "recall_delta": 0.001},
        ]
        self.assertFalse(gate_decision(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
