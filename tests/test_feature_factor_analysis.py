from __future__ import annotations

import unittest

import numpy as np

from analyze_feature_factors import average_precision, binary_auc, discretized_mutual_information


class FeatureFactorMetricsTest(unittest.TestCase):
    def test_perfect_auc_and_average_precision(self) -> None:
        y = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(binary_auc(y, scores), 1.0)
        self.assertAlmostEqual(average_precision(y, scores), 1.0)

    def test_auc_handles_ties(self) -> None:
        y = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        self.assertAlmostEqual(binary_auc(y, np.ones(4)), 0.5)

    def test_mutual_information_orders_signal_over_constant(self) -> None:
        y = np.asarray([0] * 50 + [1] * 50, dtype=np.uint8)
        signal = np.concatenate((np.zeros(50), np.ones(50)))
        constant = np.ones(100)
        self.assertGreater(discretized_mutual_information(signal, y), 0.5)
        self.assertEqual(discretized_mutual_information(constant, y), 0.0)


if __name__ == "__main__":
    unittest.main()
