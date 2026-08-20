from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from label_review_app import ReviewStore


ORIGINAL_LINE = "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000"
SECOND_LINE = "0 0.600000 0.600000 0.800000 0.600000 0.800000 0.800000 0.600000 0.800000"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ReviewStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        dataset = root / "datasets" / "Data_v3"
        self.root = root
        self.dataset = dataset
        image_path = dataset / "train" / "images" / "DJI_0001.JPG"
        label_path = dataset / "train" / "labels" / "DJI_0001.txt"
        canonical = root / "待标注_100张" / "labels" / "DJI_0001.txt"
        image_path.parent.mkdir(parents=True)
        label_path.parent.mkdir(parents=True)
        canonical.parent.mkdir(parents=True)
        image = np.full((240, 320, 3), 180, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        content = f"{ORIGINAL_LINE}\n{SECOND_LINE}\n"
        label_path.write_text(content, encoding="utf-8")
        canonical.write_text(content, encoding="utf-8")

        quality_report = root / "reports" / "Data_v3_label_quality"
        self.quality_report = quality_report
        _write_csv(
            quality_report / "review_queue.csv",
            ["priority", "source", "split", "stem"],
            [{"priority": "P1-sam2-geometry", "source": "sam2_new", "split": "train", "stem": "DJI_0001"}],
        )
        self.geometry_csv = root / "reports" / "Data_v3_geometry_review" / "geometry_review.csv"
        _write_csv(
            self.geometry_csv,
            ["stem", "split", "line", "codes", "label_path", "crop_path"],
            [{
                "stem": "DJI_0001", "split": "train", "line": 1,
                "codes": "self_intersection", "label_path": str(label_path), "crop_path": "crop.jpg",
            }],
        )
        self.state_path = root / "reports" / "state.json"
        self.store = ReviewStore(
            project_root=root,
            dataset_root=dataset,
            report_dir=quality_report,
            state_path=self.state_path,
        )
        self.item_id = next(iter(self.store.items))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_redraw_syncs_both_labels_and_undo_restores(self) -> None:
        points = [[0.15, 0.15], [0.45, 0.15], [0.45, 0.45], [0.15, 0.45]]
        result = self.store.apply_action(self.item_id, "redrawn", points)
        self.assertEqual(result["status"], "redrawn")

        dataset_text = self.store.items[self.item_id].label_path.read_text(encoding="utf-8")
        canonical_text = self.store._canonical_label(self.store.items[self.item_id]).read_text(encoding="utf-8")
        self.assertEqual(dataset_text, canonical_text)
        self.assertIn("0.15000000", dataset_text)
        self.assertIn(SECOND_LINE, dataset_text)

        self.store.undo()
        self.assertEqual(
            self.store.items[self.item_id].label_path.read_text(encoding="utf-8"),
            f"{ORIGINAL_LINE}\n{SECOND_LINE}\n",
        )
        self.assertEqual(self.store.state["items"][self.item_id]["status"], "pending")

    def test_delete_syncs_both_labels_and_undo_restores(self) -> None:
        result = self.store.apply_action(self.item_id, "deleted")
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(self.store.items[self.item_id].label_path.read_text(encoding="utf-8"), f"{SECOND_LINE}\n")
        self.assertEqual(
            self.store._canonical_label(self.store.items[self.item_id]).read_text(encoding="utf-8"),
            f"{SECOND_LINE}\n",
        )
        self.store.undo()
        self.assertEqual(
            self.store.items[self.item_id].label_path.read_text(encoding="utf-8"),
            f"{ORIGINAL_LINE}\n{SECOND_LINE}\n",
        )

    def test_redraw_can_split_one_row_into_multiple_contours(self) -> None:
        contours = [
            [[0.10, 0.10], [0.25, 0.10], [0.25, 0.25], [0.10, 0.25]],
            [[0.30, 0.30], [0.45, 0.30], [0.45, 0.45], [0.30, 0.45]],
        ]
        result = self.store.apply_action(self.item_id, "redrawn", raw_contours=contours)
        self.assertEqual(result["status"], "redrawn")
        self.assertEqual(len(result["target_contours"]), 2)
        dataset_lines = self.store.items[self.item_id].label_path.read_text(encoding="utf-8").splitlines()
        canonical_lines = self.store._canonical_label(self.store.items[self.item_id]).read_text(encoding="utf-8").splitlines()
        self.assertEqual(dataset_lines, canonical_lines)
        self.assertEqual(len(dataset_lines), 3)
        self.assertEqual(dataset_lines[-1], SECOND_LINE)

        self.store.undo()
        self.assertEqual(
            self.store.items[self.item_id].label_path.read_text(encoding="utf-8"),
            f"{ORIGINAL_LINE}\n{SECOND_LINE}\n",
        )

    def test_add_contour_keeps_review_pending_and_undo_restores(self) -> None:
        contour = [[0.20, 0.55], [0.35, 0.55], [0.35, 0.70], [0.20, 0.70]]
        result = self.store.apply_action(self.item_id, "added", raw_contours=[contour])

        self.assertEqual(result["status"], "pending")
        self.assertEqual(len(result["polygons"]), 3)
        dataset_text = self.store.items[self.item_id].label_path.read_text(encoding="utf-8")
        canonical_text = self.store._canonical_label(self.store.items[self.item_id]).read_text(encoding="utf-8")
        self.assertEqual(dataset_text, canonical_text)
        self.assertIn("0.20000000 0.55000000", dataset_text)

        self.store.undo()
        self.assertEqual(
            self.store.items[self.item_id].label_path.read_text(encoding="utf-8"),
            f"{ORIGINAL_LINE}\n{SECOND_LINE}\n",
        )
        self.assertEqual(self.store.state["items"][self.item_id]["status"], "pending")

    def test_delete_selected_non_target_polygon_and_undo_restores(self) -> None:
        item = self.store.get_item(self.item_id)
        second_polygon = next(polygon for polygon in item["polygons"] if not polygon["target"])
        result = self.store.apply_action(
            self.item_id,
            "deleted_selected",
            raw_polygon_hashes=[second_polygon["hash"]],
        )

        self.assertEqual(result["status"], "pending")
        self.assertEqual(self.store.items[self.item_id].label_path.read_text(encoding="utf-8"), f"{ORIGINAL_LINE}\n")
        self.assertEqual(
            self.store._canonical_label(self.store.items[self.item_id]).read_text(encoding="utf-8"),
            f"{ORIGINAL_LINE}\n",
        )

        self.store.undo()
        self.assertEqual(
            self.store.items[self.item_id].label_path.read_text(encoding="utf-8"),
            f"{ORIGINAL_LINE}\n{SECOND_LINE}\n",
        )

    def test_delete_selected_rejects_target_polygon(self) -> None:
        item = self.store.get_item(self.item_id)
        target_polygon = next(polygon for polygon in item["polygons"] if polygon["target"])
        with self.assertRaisesRegex(ValueError, "当前异常轮廓"):
            self.store.apply_action(
                self.item_id,
                "deleted_selected",
                raw_polygon_hashes=[target_polygon["hash"]],
            )

    def test_keep_and_undo_change_only_review_state(self) -> None:
        before = self.store.items[self.item_id].label_path.read_text(encoding="utf-8")
        self.store.apply_action(self.item_id, "kept")
        self.assertEqual(self.store.state["items"][self.item_id]["status"], "kept")
        self.assertEqual(self.store.items[self.item_id].label_path.read_text(encoding="utf-8"), before)
        self.store.undo()
        self.assertEqual(self.store.state["items"][self.item_id]["status"], "pending")

    def test_draft_survives_restart_and_action_clears_it(self) -> None:
        workspace = {
            "mode": "redraw",
            "vertex_tool": "insert",
            "draft": [[0.2, 0.2], [0.3, 0.2]],
            "draft_contours": [[[0.1, 0.1], [0.4, 0.4], [0.1, 0.4], [0.4, 0.1]]],
            "selected_hash": None,
            "freehand": True,
            "snap": False,
        }
        result = self.store.save_draft(self.item_id, workspace)
        self.assertEqual(result["workspace"]["vertex_tool"], "insert")
        self.assertEqual(result["summary"]["drafts"], 1)

        restarted = ReviewStore(
            project_root=self.root,
            dataset_root=self.dataset,
            report_dir=self.quality_report,
            geometry_csv=self.geometry_csv,
            state_path=self.state_path,
        )
        item = restarted.get_item(self.item_id)
        self.assertTrue(item["has_draft"])
        self.assertEqual(item["workspace"]["draft"], workspace["draft"])
        self.assertTrue(item["workspace"]["freehand"])

        restarted.apply_action(self.item_id, "kept")
        self.assertFalse(restarted.get_item(self.item_id)["has_draft"])
        self.assertEqual(restarted.list_items()["drafts"], 0)

    def test_invalid_draft_coordinates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "坐标"):
            self.store.save_draft(
                self.item_id,
                {
                    "mode": "add", "vertex_tool": "move", "draft": [[1.5, 0.2]],
                    "draft_contours": [], "selected_hash": None, "freehand": False, "snap": True,
                },
            )

    def test_keep_restores_a_target_deleted_by_another_edit(self) -> None:
        item = self.store.items[self.item_id]
        dataset_path = item.label_path
        canonical_path = self.store._canonical_label(item)
        dataset_path.write_text(f"{SECOND_LINE}\n", encoding="utf-8")
        canonical_path.write_text(f"{SECOND_LINE}\n", encoding="utf-8")

        result = self.store.apply_action(self.item_id, "kept")

        self.assertEqual(result["status"], "kept")
        self.assertTrue(result["target_points"])
        self.assertEqual(dataset_path.read_text(encoding="utf-8"), f"{SECOND_LINE}\n{ORIGINAL_LINE}\n")
        self.assertEqual(canonical_path.read_text(encoding="utf-8"), f"{SECOND_LINE}\n{ORIGINAL_LINE}\n")

        undo_result = self.store.undo()
        self.assertEqual(undo_result["undone"], "restored_kept")
        self.assertEqual(dataset_path.read_text(encoding="utf-8"), f"{SECOND_LINE}\n")
        self.assertEqual(self.store.state["items"][self.item_id]["status"], "pending")

    def test_delete_other_rejects_another_pending_review_target(self) -> None:
        _write_csv(
            self.geometry_csv,
            ["stem", "split", "line", "codes", "label_path", "crop_path"],
            [
                {
                    "stem": "DJI_0001", "split": "train", "line": 1,
                    "codes": "self_intersection", "label_path": "unused", "crop_path": "one.jpg",
                },
                {
                    "stem": "DJI_0001", "split": "train", "line": 2,
                    "codes": "duplicate_vertex", "label_path": "unused", "crop_path": "two.jpg",
                },
            ],
        )
        store = ReviewStore(
            project_root=self.root,
            dataset_root=self.dataset,
            report_dir=self.quality_report,
            geometry_csv=self.geometry_csv,
            state_path=self.root / "reports" / "two-items-state.json",
        )
        first_id = next(item_id for item_id, item in store.items.items() if item.initial_line == 1)
        second_polygon = next(polygon for polygon in store.get_item(first_id)["polygons"] if not polygon["target"])

        with self.assertRaisesRegex(ValueError, "待复核项"):
            store.apply_action(
                first_id,
                "deleted_selected",
                raw_polygon_hashes=[second_polygon["hash"]],
            )

    def test_redraw_rejects_self_intersection(self) -> None:
        bow_tie = [[0.1, 0.1], [0.4, 0.4], [0.1, 0.4], [0.4, 0.1]]
        with self.assertRaisesRegex(ValueError, "自交"):
            self.store.apply_action(self.item_id, "redrawn", bow_tie)


if __name__ == "__main__":
    unittest.main()
