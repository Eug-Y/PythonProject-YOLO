from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibrate_yolo_ensemble import (
    calibrate,
    calibration_shape,
    evaluate_fusion,
    load_yolo_ground_truth,
    match_instances,
    resize_masks,
)
from yolo_mask_ensemble import MaskPrediction


def rectangle(y1: int, x1: int, y2: int, x2: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class EnsembleCalibrationTest(unittest.TestCase):
    def test_instance_metrics_count_tp_fp_fn(self) -> None:
        target = rectangle(4, 4, 14, 14)
        false_positive = rectangle(20, 20, 28, 28)
        metrics = match_instances([target, false_positive], [target], threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))

    def test_fusion_grid_returns_ranked_valid_configuration(self) -> None:
        target = rectangle(5, 5, 18, 18)
        ground_truth = {"DJI_0010": [target]}
        first = {"DJI_0010": [MaskPrediction(target, 0.9, source="yolo11")]}
        second = {"DJI_0010": [MaskPrediction(rectangle(6, 6, 19, 19), 0.8, source="terrain")]}
        ranking = calibrate(
            ground_truth,
            (first, second),
            second_model_weights=(0.75, 1.0),
            match_ious=(0.5,),
            mask_votes=(0.5,),
            score_thresholds=(0.2,),
            min_model_options=(1, 2),
        )
        self.assertEqual(len(ranking), 4)
        self.assertEqual(ranking[0]["f1"], 1.0)
        self.assertGreater(ranking[0]["mean_matched_iou"], 0.7)

    def test_evaluation_handles_negative_image(self) -> None:
        target = rectangle(3, 3, 10, 10)
        ground_truth = {"positive": [target], "negative": []}
        first = {
            "positive": [MaskPrediction(target, 0.9, source="first")],
            "negative": [MaskPrediction(rectangle(20, 20, 25, 25), 0.7, source="first")],
        }
        second = {"positive": [], "negative": []}
        metrics = evaluate_fusion(
            ground_truth,
            (first, second),
            model_weights=(1.0, 1.0),
            match_iou=0.5,
            mask_vote=0.5,
            score_threshold=0.2,
            min_models=1,
        )
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))

    def test_primary_requirement_excludes_auxiliary_only_detection(self) -> None:
        target = rectangle(3, 3, 10, 10)
        metrics = evaluate_fusion(
            {"sample": [target]},
            (
                {"sample": [MaskPrediction(target, 0.9, source="yolo11")]},
                {"sample": [MaskPrediction(rectangle(20, 20, 25, 25), 0.95, source="yolov8")]},
            ),
            model_weights=(1.0, 1.0),
            match_iou=0.5,
            mask_vote=0.5,
            score_threshold=0.2,
            min_models=1,
            required_model_indices=(0,),
        )
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))

    def test_yolo_polygon_is_rasterized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
            masks = load_yolo_ground_truth(path, (20, 30))
        self.assertEqual(len(masks), 1)
        self.assertGreater(int(masks[0].sum()), 100)

    def test_calibration_masks_are_downsampled_with_aspect_ratio(self) -> None:
        self.assertEqual(calibration_shape((1300, 1600), 320), (260, 320))
        masks = np.stack([rectangle(4, 4, 14, 14)])
        resized = resize_masks(masks, (16, 16))
        self.assertEqual(resized.shape, (1, 16, 16))
        self.assertTrue(resized.any())

    def test_grid_records_unconfirmed_threshold(self) -> None:
        target = rectangle(5, 5, 18, 18)
        ranking = calibrate(
            {"sample": [target]},
            ({"sample": [MaskPrediction(target, 0.9)]}, {"sample": []}),
            second_model_weights=(1.0,), match_ious=(0.5,), mask_votes=(0.5,),
            score_thresholds=(0.2,), min_model_options=(1,),
            required_model_indices=(0,), unconfirmed_score_thresholds=(0.4, 0.6),
        )
        self.assertEqual({row["unconfirmed_score_threshold"] for row in ranking}, {0.4, 0.6})


if __name__ == "__main__":
    unittest.main()
