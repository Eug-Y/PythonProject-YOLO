import argparse
import unittest

from run_data_v22_dense_replay_multiseed import summarize, training_command


def row(seed: int, *, high_overlap: float = 0.0, map_delta: float = 0.0) -> dict[str, object]:
    return {
        "seed": seed,
        "candidate_map": 0.4 + seed / 1000,
        "map_delta": map_delta,
        "precision_delta": 0.0,
        "recall_delta": 0.0,
        "fixed_f1_delta": 0.0,
        "sparse_f1_delta": 0.02,
        "sparse_fp_delta": -10,
        "slice_summary": {"high_instance_overlap": {"delta": {"f1": high_overlap}}},
    }


class DataV22DenseReplayMultiseedTests(unittest.TestCase):
    def test_command_uses_v22_dataset_and_locked_contract(self) -> None:
        cli = argparse.Namespace(epochs=60, patience=20, imgsz=1024, batch=2, workers=2, device="0")
        command = training_command(cli, 0)
        joined = " ".join(command)
        self.assertIn("data_v22_sparse_context_dense_replay_5ch.yaml", joined)
        self.assertIn("--batch 2", joined)
        self.assertIn("--no-overlap-mask", joined)

    def test_seed0_requires_high_overlap_recovery(self) -> None:
        result = summarize([row(0, high_overlap=-0.011)], "hash")
        self.assertFalse(result["seed0_passed"])
        self.assertTrue(summarize([row(0, high_overlap=-0.01)], "hash")["seed0_passed"])

    def test_three_seed_balanced_result_passes(self) -> None:
        result = summarize([row(0), row(1, map_delta=-0.001), row(2, map_delta=-0.001)], "hash")
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["decision"], "freeze_v22_candidate_for_one_time_test")


if __name__ == "__main__":
    unittest.main()
