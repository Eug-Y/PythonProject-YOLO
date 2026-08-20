import argparse
import unittest

import run_data_v21_sparse_context_half_multiseed as v21


def row(seed: int, *, map_delta: float = 0.0, fixed: float = 0.0, sparse_f1: float = 0.015, sparse_fp: int = -9) -> dict:
    return {
        "seed": seed,
        "map_delta": map_delta,
        "precision_delta": 0.0,
        "recall_delta": 0.0,
        "fixed_f1_delta": fixed,
        "sparse_f1_delta": sparse_f1,
        "sparse_fp_delta": sparse_fp,
        "candidate_map": 0.396 + map_delta,
        "candidate_checkpoint": f"seed{seed}.pt",
    }


class DataV21SparseContextHalfMultiseedTests(unittest.TestCase):
    def test_command_uses_half_exposure_dataset_and_locked_training(self) -> None:
        cli = argparse.Namespace(epochs=60, patience=20, imgsz=1024, batch=2, workers=2, device="0")
        command = v21.training_command(cli, 0)
        joined = " ".join(command)
        self.assertIn("data_v21_sparse_context_half_5ch.yaml", joined)
        self.assertIn("v21_yolo11_5ch_sparsectx_half_b1_seed0_v1", joined)
        self.assertIn("--no-overlap-mask", command)

    def test_seed0_requires_both_sparse_and_aggregate_gates(self) -> None:
        result = v21.summarize([row(0, sparse_f1=0.009)], "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b")
        self.assertFalse(result["seed0_passed"])

    def test_three_seed_balanced_result_passes(self) -> None:
        result = v21.summarize(
            [row(0), row(1, map_delta=0.001), row(2, map_delta=-0.001)],
            "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b",
        )
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["decision"], "freeze_v21_candidate_for_one_time_test")


if __name__ == "__main__":
    unittest.main()
