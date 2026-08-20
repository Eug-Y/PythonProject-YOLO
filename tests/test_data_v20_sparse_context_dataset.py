import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_data_v20_sparse_context_dataset import (
    choose_candidates,
    clipped_labels,
    read_polygons,
    union_mask,
    window_coverage,
    window_positions,
)


class DataV20SparseContextDatasetTests(unittest.TestCase):
    def test_window_positions_include_final_boundary(self) -> None:
        self.assertEqual(window_positions(1600, 1024, 128), [0, 128, 256, 384, 512, 576])
        self.assertEqual(window_positions(100, 101, 10), [])

    def test_integral_window_coverage(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        import cv2

        integral = cv2.integral(mask, sdepth=cv2.CV_64F)
        self.assertAlmostEqual(window_coverage(integral, 2, 2, 4), 1.0)
        self.assertAlmostEqual(window_coverage(integral, 0, 0, 4), 0.25)

    def test_clipped_labels_preserve_crop_coverage(self) -> None:
        polygons = [(0, np.asarray([[2.0, 2.0], [15.0, 2.0], [15.0, 15.0], [2.0, 15.0]]))]
        lines, coverage = clipped_labels(polygons, 4, 4, 16)
        self.assertEqual(len(lines), 1)
        self.assertGreater(coverage, 0.5)
        values = lines[0].split()
        self.assertEqual(values[0], "0")
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in values[1:]))

    def test_read_and_rasterize_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
            polygons = read_polygons(path, 20, 20)
            mask = union_mask(polygons, 20, 20)
        self.assertEqual(len(polygons), 1)
        self.assertGreater(float(mask.mean()), 0.2)

    def test_candidate_selection_uses_distinct_sources(self) -> None:
        candidates = [
            {"kind": "sparse_positive", "source_image": "a.JPG", "coverage": 0.05},
            {"kind": "sparse_positive", "source_image": "b.JPG", "coverage": 0.07},
            {"kind": "empty_background", "source_image": "a.JPG", "coverage": 0.0},
            {"kind": "empty_background", "source_image": "c.JPG", "coverage": 0.0},
        ]
        selected = choose_candidates(candidates, sparse_count=2, empty_count=1)
        self.assertEqual({row["source_image"] for row in selected}, {"a.JPG", "b.JPG", "c.JPG"})


if __name__ == "__main__":
    unittest.main()
