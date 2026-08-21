import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_data_v28_yolo11_hard_examples import (
    centered_crop,
    diverse_select,
    greedy_match_iou_matrix,
    reject_forbidden_path,
)


class DataV28Yolo11HardExampleTests(unittest.TestCase):
    def test_centered_crop_clamps_to_image_boundary(self) -> None:
        self.assertEqual(centered_crop([0, 0, 20, 20], (1300, 1600), 640), (0, 0, 640))
        self.assertEqual(centered_crop([1580, 1280, 1600, 1300], (1300, 1600), 640), (960, 660, 640))

    def test_greedy_match_respects_prediction_order_and_uniqueness(self) -> None:
        matrix = np.asarray([[0.7, 0.6], [0.8, 0.1], [0.2, 0.9]])
        matches, unmatched_predictions, unmatched_targets = greedy_match_iou_matrix(matrix, 0.5)
        self.assertEqual(matches, {0: 0, 2: 1})
        self.assertEqual(unmatched_predictions, [1])
        self.assertEqual(unmatched_targets, [])

    def test_diverse_selection_prefers_unique_groups_and_sources(self) -> None:
        rows = [
            {"source_image": "a.JPG", "source_group": "g1", "score": 0.9},
            {"source_image": "b.JPG", "source_group": "g1", "score": 0.8},
            {"source_image": "c.JPG", "source_group": "g2", "score": 0.7},
        ]
        selected = diverse_select(rows, 2, rank_fields=("score",))
        self.assertEqual([row["source_image"] for row in selected], ["a.JPG", "c.JPG"])

    def test_forbidden_test_and_blind_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reject_forbidden_path(Path(directory) / "datasets" / "Data_v3" / "train")
            with self.assertRaises(ValueError):
                reject_forbidden_path(Path(directory) / "datasets" / "Data_v3" / "test")
            with self.assertRaises(ValueError):
                reject_forbidden_path(Path(directory) / "datasets" / "Data_release_blind_v1")


if __name__ == "__main__":
    unittest.main()
