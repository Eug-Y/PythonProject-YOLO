from __future__ import annotations

import unittest

import cv2
import numpy as np

from build_teacher_refined_dataset import (
    _bounded_mask_iou,
    _mask_geometry,
    _yolo_box_signature,
    mask_to_yolo_line,
    refine_label_lines,
)
from yolo_mask_ensemble import mask_iou


def rectangle(y1: int, x1: int, y2: int, x2: int, shape: tuple[int, int] = (40, 50)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class MaskPolygonConversionTest(unittest.TestCase):
    def test_duplicate_signature_matches_ultralytics_box_semantics(self) -> None:
        rectangle_line = "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4"
        diamond_line = "0 0.1 0.2 0.2 0.1 0.4 0.2 0.2 0.4"
        self.assertNotEqual(rectangle_line, diamond_line)
        self.assertEqual(_yolo_box_signature(rectangle_line), _yolo_box_signature(diamond_line))

    def test_bounded_iou_is_exactly_equivalent_to_full_mask_iou(self) -> None:
        first = rectangle(4, 5, 20, 24)
        second = rectangle(9, 12, 30, 40)
        self.assertAlmostEqual(
            _bounded_mask_iou(first, _mask_geometry(first), second, _mask_geometry(second)),
            mask_iou(first, second),
        )

    def test_mask_converts_to_valid_yolo_polygon(self) -> None:
        mask = rectangle(8, 10, 30, 38)
        line = mask_to_yolo_line(mask, class_id=2)
        self.assertIsNotNone(line)
        values = line.split()
        self.assertEqual(values[0], "2")
        coordinates = np.asarray([float(value) for value in values[1:]], dtype=np.float64)
        self.assertGreaterEqual(len(coordinates), 6)
        self.assertEqual(len(coordinates) % 2, 0)
        self.assertTrue(np.all((coordinates >= 0.0) & (coordinates <= 1.0)))

        polygon = coordinates.reshape(-1, 2) * np.asarray([mask.shape[1], mask.shape[0]])
        reconstructed = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(reconstructed, [np.rint(polygon).astype(np.int32)], 1)
        intersection = np.logical_and(mask, reconstructed).sum()
        union = np.logical_or(mask, reconstructed).sum()
        self.assertGreater(intersection / union, 0.85)

    def test_empty_mask_is_rejected(self) -> None:
        self.assertIsNone(mask_to_yolo_line(np.zeros((20, 20), dtype=bool)))


class TeacherRefinementPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lines = [
            "0 0.10 0.10 0.40 0.10 0.40 0.50 0.10 0.50",
            "0 0.60 0.20 0.90 0.20 0.90 0.60 0.60 0.60",
        ]
        self.ground_truth = [rectangle(4, 5, 20, 20), rectangle(8, 30, 24, 45)]

    def test_high_iou_teacher_replaces_only_its_matching_human_boundary(self) -> None:
        teacher = np.stack([rectangle(3, 4, 21, 21)])
        refined, stats = refine_label_lines(
            self.lines,
            self.ground_truth,
            teacher,
            np.asarray([0.92], dtype=np.float32),
            score_threshold=0.60,
            match_iou_threshold=0.50,
        )
        self.assertNotEqual(refined[0], self.lines[0])
        self.assertEqual(refined[1], self.lines[1])
        self.assertEqual(stats["refined_instances"], 1)
        self.assertEqual(stats["preserved_instances"], 1)

    def test_low_score_and_unmatched_teachers_are_never_added(self) -> None:
        teachers = np.stack([rectangle(3, 4, 21, 21), rectangle(28, 2, 38, 12)])
        refined, stats = refine_label_lines(
            self.lines,
            self.ground_truth,
            teachers,
            np.asarray([0.40, 0.99], dtype=np.float32),
            score_threshold=0.60,
            match_iou_threshold=0.50,
        )
        self.assertEqual(refined, self.lines)
        self.assertEqual(len(refined), len(self.lines))
        self.assertEqual(stats["refined_instances"], 0)
        self.assertEqual(stats["unmatched_teacher_instances"], 2)

    def test_one_teacher_cannot_replace_two_human_instances(self) -> None:
        overlapping_gt = [rectangle(5, 5, 20, 20), rectangle(6, 6, 21, 21)]
        teacher = np.stack([rectangle(5, 5, 20, 20)])
        refined, stats = refine_label_lines(
            self.lines,
            overlapping_gt,
            teacher,
            np.asarray([0.95], dtype=np.float32),
            score_threshold=0.60,
            match_iou_threshold=0.50,
        )
        self.assertEqual(stats["refined_instances"], 1)
        self.assertEqual(sum(a != b for a, b in zip(refined, self.lines)), 1)

    def test_duplicate_teacher_polygons_fall_back_to_human_boundaries(self) -> None:
        overlapping_gt = [rectangle(5, 5, 20, 20), rectangle(6, 6, 21, 21)]
        identical_teachers = np.stack([rectangle(5, 5, 20, 20), rectangle(5, 5, 20, 20)])
        refined, stats = refine_label_lines(
            self.lines,
            overlapping_gt,
            identical_teachers,
            np.asarray([0.99, 0.98], dtype=np.float32),
            score_threshold=0.60,
            match_iou_threshold=0.50,
        )
        self.assertEqual(refined, self.lines)
        self.assertEqual(len(set(refined)), len(refined))
        self.assertEqual(stats["duplicate_replacement_rejections"], 2)


if __name__ == "__main__":
    unittest.main()
