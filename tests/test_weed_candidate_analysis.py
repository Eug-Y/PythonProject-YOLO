from __future__ import annotations

import unittest

import cv2
import numpy as np

from weed_candidate_analysis import analyze_weed_candidates, estimate_row_orientation


class WeedCandidateAnalysisTests(unittest.TestCase):
    def synthetic(self):
        height, width = 240, 320
        image = np.full((height, width, 3), 70, dtype=np.uint8)
        ndvi = np.full((height, width), 0.35, dtype=np.float32)
        ndre = np.full((height, width), 0.42, dtype=np.float32)
        exg = np.full((height, width), 0.42, dtype=np.float32)
        for x in (45, 105, 165, 225, 285):
            cv2.rectangle(image, (x - 3, 15), (x + 3, 225), (35, 175, 50), -1)
            ndvi[15:226, x - 3:x + 4] = 0.82
            ndre[15:226, x - 3:x + 4] = 0.72
            exg[15:226, x - 3:x + 4] = 0.78
        # A distinct green component between crop rows.
        cv2.circle(image, (75, 120), 9, (25, 205, 40), -1)
        yy, xx = np.ogrid[:height, :width]
        weed = (xx - 75) ** 2 + (yy - 120) ** 2 <= 9 ** 2
        ndvi[weed] = 0.86
        ndre[weed] = 0.55
        exg[weed] = 0.83
        return image, ndvi, ndre, exg, weed

    def test_vertical_crop_rows_have_vertical_orientation(self) -> None:
        mask = np.zeros((200, 260), dtype=bool)
        mask[:, 40:45] = True
        mask[:, 110:115] = True
        mask[:, 180:185] = True
        angle, confidence, lines = estimate_row_orientation(mask)
        self.assertLess(abs(abs(angle) - 90.0), 8.0)
        self.assertGreater(confidence, 0.5)
        self.assertGreater(lines, 0)

    def test_off_row_vegetation_is_emitted_as_review_candidate(self) -> None:
        image, ndvi, ndre, exg, weed = self.synthetic()
        result = analyze_weed_candidates(image, ndvi, ndre, exg=exg, gsd=0.03)
        self.assertEqual(result["overlay"].shape, image.shape)
        self.assertTrue(result["metrics"]["review_required"])
        self.assertGreater(result["metrics"]["weed_candidate_count"], 0)
        self.assertGreater(float(np.mean(result["candidate_mask"][weed])), 0.5)
        self.assertIn("off_row_vegetation", {row["reason"] for row in result["candidates"]})

    def test_empty_vegetation_is_safe(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        low = np.zeros((80, 100), dtype=np.float32)
        result = analyze_weed_candidates(image, low, low, exg=low)
        self.assertEqual(result["metrics"]["weed_candidate_count"], 0)
        self.assertFalse(result["candidate_mask"].any())


if __name__ == "__main__":
    unittest.main()
