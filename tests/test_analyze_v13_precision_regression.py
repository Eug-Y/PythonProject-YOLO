import unittest

from analyze_v13_precision_regression import aggregate, flight_segments


class PrecisionRegressionAnalysisTests(unittest.TestCase):
    def test_flight_segments_split_large_numeric_gaps(self) -> None:
        segments = flight_segments(["DJI_0010.JPG", "DJI_0030.JPG", "DJI_0100.JPG"], maximum_gap=40)
        self.assertEqual(segments["DJI_0010.JPG"], "DJI_0010-0030")
        self.assertEqual(segments["DJI_0030.JPG"], "DJI_0010-0030")
        self.assertEqual(segments["DJI_0100.JPG"], "DJI_0100-0100")

    def test_aggregate_uses_micro_counts(self) -> None:
        rows = [
            {"model_tp": 2, "model_fp": 1, "model_fn": 2},
            {"model_tp": 3, "model_fp": 0, "model_fn": 1},
        ]
        result = aggregate(rows, "model")
        self.assertEqual(result["tp"], 5)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 3)
        self.assertEqual(result["precision"], 5 / 6)
        self.assertEqual(result["recall"], 5 / 8)


if __name__ == "__main__":
    unittest.main()
