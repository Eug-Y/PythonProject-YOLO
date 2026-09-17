from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_progress_report as report
from calibrate_yolo_ensemble import match_instances


def square(top: int, left: int, size: int, shape=(200, 260)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + size, left:left + size] = True
    return mask


class BuildProgressReportTests(unittest.TestCase):
    def test_production_config_is_active_yolo11_primary(self) -> None:
        production, fusion = report.load_production()
        self.assertEqual(production["state"], "active")
        self.assertTrue(production["every_output_requires_yolo11"])
        self.assertEqual(production["channels"], ["R", "G", "B", "NDVI", "NDRE"])
        self.assertEqual(fusion["role"], "strict_yolo11_primary")
        self.assertEqual(fusion["required_model_indices"], [0])
        self.assertEqual(fusion["model_weights"], [1.0, 1.0])
        self.assertEqual(fusion["score_threshold"], production["confidence"])

    def test_sha256_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"frozen-production-artifact")
            self.assertEqual(report.sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_match_detail_agrees_with_counted_matcher(self) -> None:
        ground_truth = [square(20, 20, 40), square(100, 150, 40), square(30, 180, 30)]
        predictions = [square(21, 21, 40), square(100, 150, 40), square(160, 10, 25)]
        bboxes = [
            (int(np.nonzero(mask)[1].min()), int(np.nonzero(mask)[0].min()),
             int(np.nonzero(mask)[1].max() + 1), int(np.nonzero(mask)[0].max() + 1))
            for mask in predictions
        ]
        pairs, matched = report.match_detail(predictions, ground_truth, 0.5, bboxes)
        counted = match_instances(predictions, ground_truth, 0.5, prediction_bboxes=bboxes)
        self.assertEqual(len(pairs), counted["tp"])
        self.assertEqual(len(matched), counted["tp"])
        self.assertEqual(len(predictions) - len(pairs), counted["fp"])
        self.assertEqual(len(ground_truth) - len(matched), counted["fn"])
        self.assertEqual({target for _, target, _ in pairs}, matched)

    def test_select_showcase_covers_worst_best_and_error_extremes(self) -> None:
        rows = []
        for index, (f1, fp, fn) in enumerate(
            [(0.10, 5, 9), (0.90, 1, 2), (0.55, 40, 3), (0.60, 4, 30), (0.50, 7, 7), (0.45, 2, 4)]
        ):
            rows.append({
                "image": f"DJI_{index:04d}.JPG",
                "stem": f"DJI_{index:04d}",
                "ground_truth": 10,
                "fusion": {"f1": f1, "fp": fp, "fn": fn, "tp": 1, "precision": f1, "recall": f1, "predictions": 3},
            })
        picks = report.select_showcase(rows, limit=5)
        stems = [row["stem"] for row in picks]
        self.assertEqual(len(picks), 5)
        self.assertEqual(len(set(stems)), 5)
        self.assertEqual(stems[0], "DJI_0000")
        self.assertIn("DJI_0001", stems)
        self.assertIn("DJI_0002", stems)
        self.assertIn("DJI_0003", stems)
        self.assertEqual(len(report.select_showcase(rows, limit=2)), 2)

    def test_unlabelled_background_is_excluded_from_showcase(self) -> None:
        def row(stem: str, ground_truth: int, f1: float, fp: int, fn: int) -> dict:
            return {"image": f"{stem}.JPG", "stem": stem, "ground_truth": ground_truth,
                    "fusion": {"f1": f1, "fp": fp, "fn": fn, "tp": 0, "precision": f1,
                               "recall": f1, "predictions": fp}}

        rows = [row("DJI_0000", 0, 0.0, 120, 0), row("DJI_0001", 30, 0.55, 10, 12),
                row("DJI_0002", 40, 0.80, 2, 6), row("DJI_0003", 25, 0.45, 20, 3)]
        empty = report.zero_ground_truth(rows)
        self.assertEqual([item["stem"] for item in empty], ["DJI_0000"])
        picked = [item["stem"] for item in report.select_showcase(rows, limit=3)]
        self.assertNotIn("DJI_0000", picked)
        self.assertIn("DJI_0001", picked)
        self.assertIn("DJI_0002", picked)

    def test_showcase_still_returns_rows_when_every_label_is_empty(self) -> None:
        rows = [{"image": "DJI_0000.JPG", "stem": "DJI_0000", "ground_truth": 0,
                 "fusion": {"f1": 0.0, "fp": 5, "fn": 0, "tp": 0, "precision": 0.0,
                            "recall": 0.0, "predictions": 5}}]
        self.assertEqual(len(report.select_showcase(rows, limit=5)), 1)

    def test_error_focus_stays_inside_the_image(self) -> None:
        shape = (1300, 1600)
        ground_truth = [square(1250, 1560, 40, shape)]
        predictions = [square(10, 10, 30, shape)]
        pairs, matched = report.match_detail(predictions, ground_truth, 0.5, [(10, 10, 40, 40)])
        left, top, width, height = report.error_focus(pairs, ground_truth, predictions, shape, matched)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(left + width, shape[1])
        self.assertLessEqual(top + height, shape[0])
        self.assertEqual((width, height), (report.CROP_WIDTH, report.CROP_HEIGHT))

    def test_rendering_helpers_preserve_panel_geometry(self) -> None:
        image = np.full((260, 320, 3), 60, dtype=np.uint8)
        mask = square(20, 20, 40, (260, 320))
        panel, small = report.fit_panel(image, mask)
        self.assertEqual(panel.shape, (report.PANEL_HEIGHT, report.PANEL_WIDTH, 3))
        self.assertEqual(small.shape, (report.PANEL_HEIGHT, report.PANEL_WIDTH))
        blended = report.blend(panel, small, report.GREEN, 0.5)
        self.assertEqual(blended.shape, panel.shape)
        self.assertGreater(int(blended[small].sum()), int(panel[small].sum()))
        outlined = report.outline(panel.copy(), small, report.CYAN, 2)
        self.assertEqual(outlined.shape, panel.shape)
        self.assertTrue(np.array_equal(report.blend(panel, np.zeros_like(small), report.RED, 0.5), panel))

    def test_build_html_renders_metrics_and_missing_optional_logs(self) -> None:
        production = {
            "model_name": "v27_yolo11_primary_fusion_v1",
            "imgsz": 1024,
            "primary": {"architecture": "YOLO11n-seg", "sha256": "a" * 64},
            "auxiliary": {"architecture": "YOLOv8n-seg", "sha256": "b" * 64},
        }
        fusion = {
            "model_weights": [1.0, 1.0],
            "match_iou": 0.4,
            "mask_vote": 0.5,
            "score_threshold": 0.2,
            "unconfirmed_score_threshold": 0.225,
            "min_models": 1,
        }
        metrics = {"tp": 1855, "fp": 1020, "fn": 1045, "precision": 0.6452,
                   "recall": 0.6396, "f1": 0.6424, "mean_matched_iou": 0.8420}
        aggregate = {key: dict(metrics) for key in ("fusion", "primary_yolo11", "auxiliary_yolov8")}
        verification = {"recorded": dict(metrics), "deltas": {key: 0.0 for key in metrics},
                        "exact": True, "checks": {"check_a": True, "check_b": False}}
        with tempfile.TemporaryDirectory() as directory:
            page = report.build_html(Path(directory), production, fusion, aggregate, verification,
                                     [], {}, {}, ("yolo11_primary", "yolov8_auxiliary"),
                                     zero_gt=[{"image": "DJI_2330.JPG",
                                               "fusion": {"predictions": 132, "fp": 132}}])
        self.assertIn("项目进度与最新效果报告", page)
        self.assertIn("0.6424", page)
        self.assertIn("check_b", page)
        self.assertIn("未通过", page)
        self.assertIn("发现一处标注缺口", page)
        self.assertIn("DJI_2330.JPG", page)

    def test_read_json_returns_default_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(report._read_json(Path(directory) / "absent.json"))
            present = Path(directory) / "present.json"
            present.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(report._read_json(present), {"ok": True})


if __name__ == "__main__":
    unittest.main()