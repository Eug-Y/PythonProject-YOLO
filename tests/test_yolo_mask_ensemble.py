from __future__ import annotations

import unittest

import numpy as np

from yolo_mask_ensemble import MaskPrediction, bbox_from_mask, cropped_mask_iou, fuse_mask_predictions, mask_iou


def rectangle(y1: int, x1: int, y2: int, x2: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class MaskEnsembleTest(unittest.TestCase):
    def test_overlapping_yolo11_and_yolov8_instances_are_fused(self) -> None:
        first = MaskPrediction(rectangle(5, 5, 18, 18), 0.8, source="yolo11_5ch")
        second = MaskPrediction(rectangle(6, 6, 19, 19), 0.6, source="yolov8_terrain")
        fused = fuse_mask_predictions([[first], [second]], match_iou=0.5)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].sources, ("yolo11_5ch", "yolov8_terrain"))
        self.assertAlmostEqual(fused[0].score, 0.7)
        self.assertGreater(mask_iou(fused[0].mask, first.mask), 0.8)

    def test_distant_instances_remain_separate(self) -> None:
        first = MaskPrediction(rectangle(2, 2, 8, 8), 0.9, source="yolo11")
        second = MaskPrediction(rectangle(20, 20, 28, 28), 0.8, source="yolov8")
        fused = fuse_mask_predictions([[first], [second]])
        self.assertEqual(len(fused), 2)

    def test_min_models_suppresses_unconfirmed_instances(self) -> None:
        first = MaskPrediction(rectangle(2, 2, 8, 8), 0.9, source="yolo11")
        second = MaskPrediction(rectangle(20, 20, 28, 28), 0.8, source="yolov8")
        fused = fuse_mask_predictions([[first], [second]], min_models=2)
        self.assertEqual(fused, [])

    def test_same_model_instances_are_not_merged(self) -> None:
        first = MaskPrediction(rectangle(5, 5, 18, 18), 0.9, source="yolo11")
        second = MaskPrediction(rectangle(6, 6, 19, 19), 0.7, source="yolo11")
        fused = fuse_mask_predictions([[first, second]])
        self.assertEqual(len(fused), 2)

    def test_required_primary_blocks_auxiliary_only_instances(self) -> None:
        primary = MaskPrediction(rectangle(2, 2, 8, 8), 0.8, source="yolo11")
        auxiliary_match = MaskPrediction(rectangle(2, 2, 8, 8), 0.9, source="yolov8")
        auxiliary_only = MaskPrediction(rectangle(20, 20, 28, 28), 0.95, source="yolov8")
        fused = fuse_mask_predictions(
            [[primary], [auxiliary_match, auxiliary_only]], required_model_indices=(0,)
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(set(fused[0].sources), {"yolo11", "yolov8"})

    def test_required_model_index_must_exist(self) -> None:
        primary = MaskPrediction(rectangle(2, 2, 8, 8), 0.8)
        with self.assertRaises(ValueError):
            fuse_mask_predictions([[primary]], required_model_indices=(1,))

    def test_high_confidence_primary_can_survive_without_confirmation(self) -> None:
        high = MaskPrediction(rectangle(2, 2, 8, 8), 0.8, source="yolo11")
        low = MaskPrediction(rectangle(20, 20, 28, 28), 0.3, source="yolo11")
        fused = fuse_mask_predictions(
            [[high, low], []], required_model_indices=(0,),
            unconfirmed_score_threshold=0.5,
        )
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].score, 0.8)

    def test_shape_mismatch_is_rejected(self) -> None:
        first = MaskPrediction(rectangle(2, 2, 8, 8), 0.9)
        other = np.ones((16, 16), dtype=bool)
        second = MaskPrediction(other, 0.8)
        with self.assertRaisesRegex(ValueError, "original-size"):
            fuse_mask_predictions([[first], [second]])

    def test_cropped_iou_matches_full_frame_iou(self) -> None:
        first = rectangle(5, 5, 18, 18)
        second = rectangle(6, 6, 19, 19)
        self.assertAlmostEqual(
            cropped_mask_iou(first, second, bbox_from_mask(first), bbox_from_mask(second)),
            mask_iou(first, second),
        )


if __name__ == "__main__":
    unittest.main()
