import unittest

from run_data_v24_yolo11_hybrid_fusion_valid import THRESHOLDS, calibration_command, protocol


class DataV24HybridFusionTests(unittest.TestCase):
    def test_protocol_is_valid_only_and_keeps_v23_gate(self) -> None:
        value = protocol()
        self.assertFalse(value["held_out_test_used"])
        self.assertFalse(value["release_blind_holdout_used"])
        self.assertTrue(value["gate_unchanged_from_v23"])
        self.assertEqual(tuple(value["unconfirmed_score_thresholds"]), THRESHOLDS)

    def test_strict_command_requires_yolo11_and_hybrid_threshold(self) -> None:
        command = calibration_command(None if False else __import__('pathlib').Path('out'), 0.4, True)
        self.assertIn("--require-primary", command)
        self.assertEqual(command[command.index("--unconfirmed-score-thresholds") + 1], "0.4")
        self.assertEqual(command[command.index("--min-model-options") + 1], "1")


if __name__ == "__main__":
    unittest.main()
