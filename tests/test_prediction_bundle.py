from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prediction_bundle import load_prediction_bundle, save_prediction_bundle


class PredictionBundleTest(unittest.TestCase):
    def test_mask_round_trip_is_lossless(self) -> None:
        masks = np.zeros((2, 11, 13), dtype=bool)
        masks[0, 1:5, 2:8] = True
        masks[1, 5:10, 7:12] = True
        scores = np.asarray([0.9, 0.6], dtype=np.float32)
        classes = np.asarray([0, 0], dtype=np.int16)
        boxes = np.asarray([[2, 1, 8, 5], [7, 5, 12, 10]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            save_prediction_bundle(path, masks, scores, classes, boxes)
            loaded = load_prediction_bundle(path)
        np.testing.assert_array_equal(loaded["masks"], masks)
        np.testing.assert_allclose(loaded["scores"], scores)
        np.testing.assert_array_equal(loaded["class_ids"], classes)
        np.testing.assert_allclose(loaded["boxes"], boxes)
        np.testing.assert_array_equal(loaded["image_hw"], [11, 13])

    def test_empty_prediction_set_preserves_image_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.npz"
            save_prediction_bundle(
                path,
                np.zeros((0, 7, 9), dtype=bool),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int16),
            )
            loaded = load_prediction_bundle(path)
        self.assertEqual(loaded["masks"].shape, (0, 7, 9))

    def test_inconsistent_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            with self.assertRaisesRegex(ValueError, "one value per mask"):
                save_prediction_bundle(
                    path,
                    np.ones((2, 5, 5), dtype=bool),
                    np.ones(1, dtype=np.float32),
                    np.zeros(2, dtype=np.int16),
                )


if __name__ == "__main__":
    unittest.main()

