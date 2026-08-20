import unittest

import numpy as np

from sweep_multichannel_confidence import threshold_rows


class ConfidenceSweepTests(unittest.TestCase):
    def test_threshold_rows_counts_tp_fp_fn(self) -> None:
        rows = threshold_rows(
            confidence=np.asarray([0.9, 0.7, 0.2]),
            true_positive_mask=np.asarray([True, False, True]),
            target_count=3,
            thresholds=np.asarray([0.5]),
        )
        self.assertEqual(rows[0]["tp"], 1)
        self.assertEqual(rows[0]["fp"], 1)
        self.assertEqual(rows[0]["fn"], 2)
        self.assertEqual(rows[0]["precision"], 0.5)
        self.assertEqual(rows[0]["recall"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
