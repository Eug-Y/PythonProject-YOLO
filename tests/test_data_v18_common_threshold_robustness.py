import unittest

from run_data_v18_common_threshold_robustness import compare_rows, gate


class V18CommonThresholdRobustnessTests(unittest.TestCase):
    @staticmethod
    def row(seed: int, p_delta: float, r_delta: float, f1_delta: float) -> dict:
        return {
            "seed": seed,
            "precision_delta": p_delta,
            "recall_delta": r_delta,
            "f1_delta": f1_delta,
        }

    def test_compare_rows_calculates_fixed_threshold_deltas(self) -> None:
        control = {"confidence": 0.371, "tp": 10, "fp": 4, "fn": 5, "precision": 0.7, "recall": 0.6, "f1": 0.64}
        candidate = {"confidence": 0.371, "tp": 11, "fp": 3, "fn": 4, "precision": 0.72, "recall": 0.61, "f1": 0.65}
        row = compare_rows(0, control, candidate)
        self.assertAlmostEqual(row["precision_delta"], 0.02)
        self.assertEqual(row["fp_delta"], -1)

    def test_gate_requires_nonnegative_f1_in_every_seed(self) -> None:
        passing = [self.row(seed, 0.01, -0.005, 0.001) for seed in range(3)]
        self.assertTrue(gate(passing)["passed"])
        failing = list(passing)
        failing[2] = self.row(2, 0.01, -0.005, -0.0001)
        self.assertFalse(gate(failing)["passed"])

    def test_gate_requires_exact_three_seeds(self) -> None:
        with self.assertRaises(ValueError):
            gate([self.row(0, 0.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
