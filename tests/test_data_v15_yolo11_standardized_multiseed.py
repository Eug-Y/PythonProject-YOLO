import unittest

from run_data_v15_yolo11_standardized_multiseed import summarize


class V15StandardizedMultiseedTests(unittest.TestCase):
    @staticmethod
    def row(seed: int, map_delta: float, precision_delta: float = 0.0, recall_delta: float = 0.0) -> dict:
        return {
            "seed": seed,
            "map_delta": map_delta,
            "precision_delta": precision_delta,
            "recall_delta": recall_delta,
            "candidate_map": 0.4 + map_delta,
            "candidate_checkpoint": f"seed{seed}.pt",
        }

    def test_seed0_gate_is_staged(self) -> None:
        summary = summarize([self.row(0, 0.01)], "hash")
        self.assertTrue(summary["seed0_passed"])
        self.assertFalse(summary["gate_passed"])
        self.assertEqual(summary["decision"], "continue_multiseed")

    def test_final_gate_passes_stable_three_seed_gain(self) -> None:
        summary = summarize([self.row(seed, delta) for seed, delta in enumerate((0.01, 0.006, 0.004))], "hash")
        self.assertTrue(summary["gate_passed"])
        self.assertEqual(summary["decision"], "freeze_candidate_for_review_no_auto_deployment")

    def test_seed0_precision_guard_stops_progression(self) -> None:
        summary = summarize([self.row(0, 0.02, precision_delta=-0.02)], "hash")
        self.assertFalse(summary["seed0_passed"])
        self.assertEqual(summary["decision"], "reject_retain_current_deployment")


if __name__ == "__main__":
    unittest.main()
