import argparse
import unittest

from run_data_v20_sparse_context_multiseed import summarize, training_command


def row(seed: int, *, map_delta: float = 0.0, fixed: float = 0.0, sparse_f1: float = 0.03, sparse_fp: int = -10) -> dict:
    return {
        "seed": seed,
        "map_delta": map_delta,
        "precision_delta": 0.0,
        "recall_delta": 0.0,
        "fixed_f1_delta": fixed,
        "sparse_f1_delta": sparse_f1,
        "sparse_fp_delta": sparse_fp,
        "candidate_map": 0.39 + map_delta,
        "candidate_checkpoint": f"seed{seed}.pt",
    }


class DataV20SparseContextMultiseedTests(unittest.TestCase):
    def test_command_changes_dataset_but_keeps_v19_training_contract(self) -> None:
        cli = argparse.Namespace(epochs=60, patience=20, imgsz=1024, batch=2, workers=2, device="0")
        command = training_command(cli, 0)
        self.assertIn("data_v20_sparse_context_5ch.yaml", " ".join(command))
        self.assertIn("--no-overlap-mask", command)
        self.assertEqual(command[command.index("--mosaic") + 1], "0.5")
        self.assertEqual(command[command.index("--mixup") + 1], "0.0")
        self.assertEqual(command[command.index("--close-mosaic") + 1], "15")

    def test_seed0_gate_requires_sparse_recovery(self) -> None:
        result = summarize([row(0, sparse_f1=0.01)], "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b")
        self.assertFalse(result["seed0_passed"])

    def test_three_seed_stable_risk_reduction_passes(self) -> None:
        result = summarize(
            [row(0, map_delta=0.001), row(1, map_delta=0.0), row(2, map_delta=-0.001)],
            "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b",
        )
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["decision"], "freeze_v20_candidate_for_one_time_test")

    def test_three_seed_gate_rejects_aggregate_regression(self) -> None:
        result = summarize(
            [row(0), row(1, map_delta=-0.006), row(2)],
            "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b",
        )
        self.assertFalse(result["gate_passed"])


if __name__ == "__main__":
    unittest.main()
