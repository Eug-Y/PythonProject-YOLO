import argparse
import unittest

from run_data_v19b_yolo11_boundary_nooverlap_multiseed import summarize, training_command


def row(seed: int, map_delta: float, f1_delta: float = 0.001) -> dict[str, object]:
    return {
        "seed": seed,
        "map_delta": map_delta,
        "precision_delta": 0.001,
        "recall_delta": 0.001,
        "fixed_f1_delta": f1_delta,
        "candidate_map": 0.38 + map_delta,
        "candidate_checkpoint": f"seed{seed}.pt",
        "candidate_checkpoint_sha256": str(seed) * 64,
    }


class DataV19BYolo11BoundaryNoOverlapTests(unittest.TestCase):
    def test_command_locks_combination(self) -> None:
        cli = argparse.Namespace(epochs=60, patience=20, imgsz=1024, batch=2, workers=2, device="0")
        command = training_command(cli, 0)
        self.assertIn("--no-overlap-mask", command)
        self.assertEqual(command[command.index("--mosaic") + 1], "0.5")
        self.assertEqual(command[command.index("--mixup") + 1], "0.0")
        self.assertEqual(command[command.index("--close-mosaic") + 1], "15")

    def test_seed0_gate_stops_on_fixed_f1_regression(self) -> None:
        summary = summarize([row(0, 0.004, -0.001)], "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b")
        self.assertFalse(summary["seed0_passed"])
        self.assertEqual(summary["decision"], "reject_v19b_retain_v15_seed2")

    def test_three_seed_stable_gain_passes(self) -> None:
        summary = summarize(
            [row(0, 0.004), row(1, 0.0035), row(2, 0.003)],
            "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b",
        )
        self.assertTrue(summary["gate_passed"])
        self.assertEqual(summary["decision"], "freeze_v19b_candidate_for_review_no_auto_deployment")


if __name__ == "__main__":
    unittest.main()
