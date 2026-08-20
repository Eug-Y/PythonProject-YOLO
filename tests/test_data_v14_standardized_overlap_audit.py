import unittest

from run_data_v14_standardized_overlap_audit import gate_decision


class StandardizedOverlapAuditTests(unittest.TestCase):
    def test_gate_passes_stable_improvement(self) -> None:
        rows = [
            {"map_delta": value, "precision_delta": 0.0, "recall_delta": 0.0}
            for value in (0.003, 0.004, 0.001)
        ]
        self.assertTrue(gate_decision(rows)["passed"])

    def test_gate_rejects_precision_regression(self) -> None:
        rows = [
            {"map_delta": 0.01, "precision_delta": -0.02, "recall_delta": 0.0}
            for _ in range(3)
        ]
        self.assertFalse(gate_decision(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
