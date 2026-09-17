from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_data_v30_label_density_audit as audit


def write_label(path: Path, rects, class_id: str = "0") -> None:
    """Write YOLO-segmentation rectangles given as normalized (x0, y0, x1, y1)."""
    lines = []
    for x0, y0, x1, y1 in rects:
        lines.append(
            f"{class_id} {x0:.6f} {y0:.6f} {x1:.6f} {y0:.6f} "
            f"{x1:.6f} {y1:.6f} {x0:.6f} {y1:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def make_dataset(root: Path, images: dict[str, dict[str, list]]) -> None:
    """Create <split>/labels/<stem>.txt for each split in the mapping."""
    for split, stems in images.items():
        directory = root / split / "labels"
        directory.mkdir(parents=True, exist_ok=True)
        for stem, rects in stems.items():
            write_label(directory / f"{stem}.txt", rects)


def batch(polygons: float, radius: float, overlap: float, tiling: float) -> dict:
    return {
        "polygons_per_image_median": polygons,
        "median_radius": radius,
        "overlap_share_mean": overlap,
        "tiling_share_mean": tiling,
    }


class LabelDensityAuditTests(unittest.TestCase):
    def test_read_polygons_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DJI_0001.txt"
            path.write_text(
                "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
                "0 0.3 0.3 0.4 0.3\n"          # too few coordinates
                "0 0.5 0.5 0.6 0.5 0.6\n"       # odd coordinate count
                "0 0.7 0.7 0.8 0.7 0.8 0.8 0.7 0.8\n",
                encoding="utf-8",
            )
            polygons, malformed = audit.read_polygons(path)
            self.assertEqual(len(polygons), 2)
            self.assertEqual(malformed, 2)
            self.assertEqual(polygons[0].shape, (4, 2))

    def test_rasterized_iou_is_one_for_identical_instances(self) -> None:
        points = audit.normalized_to_grid(np.asarray([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]))
        first = audit.Instance(0, points)
        second = audit.Instance(1, points)
        roi = first.bbox
        self.assertGreater(first.area, 0)
        self.assertAlmostEqual(audit.polygon_iou(first.mask, second.mask), 1.0, places=6)
        self.assertEqual(roi, second.bbox)

    def test_overlap_indicator_flags_duplicated_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DJI_0002.txt"
            write_label(path, [(0.2, 0.2, 0.4, 0.4), (0.2, 0.2, 0.4, 0.4), (0.2, 0.2, 0.4, 0.4)])
            metrics = audit.image_metrics(path)
            self.assertEqual(metrics["polygons"], 3)
            self.assertEqual(metrics["overlap_pairs"], 3)
            self.assertAlmostEqual(float(metrics["overlap_share"]), 1.0, places=6)
            self.assertAlmostEqual(float(metrics["max_iou"]), 1.0, places=6)
            self.assertAlmostEqual(float(metrics["union_over_sum"]), 1.0 / 3.0, places=6)

    def test_tiling_indicator_flags_adjacent_pieces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DJI_0003.txt"
            pieces = [(0.10 + index * 0.04, 0.40, 0.14 + index * 0.04, 0.50) for index in range(6)]
            write_label(path, pieces)
            metrics = audit.image_metrics(path)
            self.assertEqual(metrics["polygons"], 6)
            self.assertEqual(metrics["overlap_pairs"], 0)
            self.assertAlmostEqual(float(metrics["overlap_share"]), 0.0, places=6)
            self.assertGreaterEqual(float(metrics["tiling_share"]), 0.5)

    def test_disjoint_instances_show_no_fragmentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DJI_0004.txt"
            write_label(path, [(0.05, 0.05, 0.15, 0.15), (0.40, 0.40, 0.50, 0.50), (0.80, 0.75, 0.90, 0.85)])
            metrics = audit.image_metrics(path)
            self.assertEqual(metrics["overlap_pairs"], 0)
            self.assertEqual(metrics["tiling_polygons"], 0)
            self.assertAlmostEqual(float(metrics["overlap_share"]), 0.0, places=6)
            self.assertAlmostEqual(float(metrics["tiling_share"]), 0.0, places=6)

    def test_decision_rule_detects_over_segmentation(self) -> None:
        decision = audit.decide(
            batch(25, 2.0, 0.20, 0.30),
            batch(10, 5.0, 0.02, 0.05),
        )
        self.assertEqual(decision["decision"], "over_segmentation")
        self.assertEqual(decision["indicators_fired"], 3)
        self.assertAlmostEqual(decision["polygon_ratio"], 2.5, places=6)

    def test_decision_rule_detects_real_density(self) -> None:
        decision = audit.decide(
            batch(25, 5.2, 0.02, 0.04),
            batch(10, 5.0, 0.02, 0.04),
        )
        self.assertEqual(decision["decision"], "real_density")
        self.assertEqual(decision["indicators_fired"], 0)

    def test_decision_rule_is_inconclusive_for_similar_batches(self) -> None:
        decision = audit.decide(
            batch(11, 5.0, 0.02, 0.04),
            batch(10, 5.0, 0.02, 0.04),
        )
        self.assertEqual(decision["decision"], "inconclusive")

    def test_rank_correlation_matches_monotone_orders(self) -> None:
        rising = [(index, index * 3.0) for index in range(6)]
        falling = [(index, -index) for index in range(6)]
        self.assertAlmostEqual(audit.rank_correlation(rising), 1.0, places=6)
        self.assertAlmostEqual(audit.rank_correlation(falling), -1.0, places=6)
        self.assertEqual(audit.rank_correlation([(1.0, 1.0)]), 0.0)

    def test_audit_reads_only_train_and_valid(self) -> None:
        self.assertEqual(audit.AUDITED_SPLITS, ("train", "valid"))
        self.assertNotIn("test", audit.AUDITED_SPLITS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(
                root,
                {
                    "train": {"DJI_0001": [(0.1, 0.1, 0.2, 0.2)]},
                    "valid": {"DJI_0002": [(0.1, 0.1, 0.2, 0.2)]},
                    "test": {"DJI_0003": [(0.1, 0.1, 0.2, 0.2)]},
                },
            )
            sam2_dir = root / "sam2"
            sam2_dir.mkdir()
            write_label(sam2_dir / "DJI_0002.txt", [(0.1, 0.1, 0.2, 0.2)])
            summary, rows = audit.build_summary(
                root, ("train", "valid"), {"DJI_0002"}, sam2_dir, root / "missing.json"
            )
            self.assertEqual({row["split"] for row in rows}, {"train", "valid"})
            self.assertEqual(sorted(summary["inputs"]["label_set_sha256"]), ["train", "valid"])
            self.assertFalse(summary["held_out_test_used"])
            self.assertFalse(summary["sealed_blind_used"])
            self.assertFalse(summary["recall_linkage"]["available"])

    def test_sam2_stems_are_attributed_to_the_sam2_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(
                root,
                {
                    "train": {
                        "DJI_0001": [(0.1, 0.1, 0.2, 0.2)],
                        "DJI_0002": [(0.1, 0.1, 0.2, 0.2), (0.4, 0.4, 0.5, 0.5)],
                    }
                },
            )
            summary, rows = audit.build_summary(
                root, ("train",), {"DJI_0002"}, root, root / "missing.json"
            )
            sources = {row["stem"]: row["source"] for row in rows}
            self.assertEqual(sources, {"DJI_0001": "base", "DJI_0002": "sam2_new"})
            self.assertIn("base", summary["per_split"]["train"]["batches"])
            self.assertIn("sam2_new", summary["per_split"]["train"]["batches"])

    def test_summary_and_html_are_written_with_the_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(
                root,
                {
                    "train": {
                        "DJI_0001": [(0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.6, 0.6)],
                        "DJI_0002": [(0.2, 0.2, 0.3, 0.3)] * 4,
                    }
                },
            )
            summary, _ = audit.build_summary(root, ("train",), {"DJI_0002"}, root, root / "missing.json")
            markup = audit.render_html(summary)
            self.assertIn("Data_v30", markup)
            self.assertIn(summary["decision"]["decision"], markup)
            self.assertIn("held_out_test_used = false", markup)
            csv_path = root / "per_image_density.csv"
            audit.write_csv(summary["densest_images"], csv_path)
            self.assertIn("overlap_share", csv_path.read_text(encoding="utf-8").splitlines()[0])

    def test_cli_rejects_the_frozen_test_split(self) -> None:
        import sys

        original = sys.argv
        try:
            sys.argv = ["build_data_v30_label_density_audit.py", "--splits", "test"]
            with self.assertRaises(SystemExit):
                audit.parse_args()
        finally:
            sys.argv = original

    def test_recall_linkage_requires_a_non_test_metrics_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"split": "valid", "stem": "DJI_0001", "source": "base", "polygons": 12}]
            blocked = root / "metrics.json"
            blocked.write_text(json.dumps({"held_out_test_used": True, "images": []}), encoding="utf-8")
            self.assertFalse(audit.recall_linkage(rows, blocked)["available"])

            metrics = root / "ok.json"
            metrics.write_text(
                json.dumps(
                    {
                        "held_out_test_used": False,
                        "images": [
                            {"stem": "DJI_0001", "ground_truth": 10, "fusion": {"fn": 1, "recall": 0.9}},
                            {"stem": "DJI_0002", "ground_truth": 10, "fusion": {"fn": 4, "recall": 0.6}},
                            {"stem": "DJI_0003", "ground_truth": 10, "fusion": {"fn": 9, "recall": 0.1}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dense_rows = rows + [
                {"split": "valid", "stem": "DJI_0002", "source": "sam2_new", "polygons": 40},
                {"split": "valid", "stem": "DJI_0003", "source": "sam2_new", "polygons": 90},
            ]
            linkage = audit.recall_linkage(dense_rows, metrics)
            self.assertTrue(linkage["available"])
            self.assertEqual(linkage["images"], 3)
            self.assertAlmostEqual(linkage["spearman_polygons_vs_fn_rate"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()