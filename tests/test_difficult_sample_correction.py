from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_difficult_sample_correction import build_queue
from difficult_sample_correction_app import create_app


ORIGINAL = "0 0.10000000 0.10000000 0.30000000 0.10000000 0.20000000 0.30000000\n"


class DifficultSampleCorrectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        review = self.root / "review"
        review.mkdir()
        with (review / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["rank", "image", "stem", "split", "batch_index", "mask_f1"],
            )
            writer.writeheader()
            writer.writerow(
                {"rank": 1, "image": "DJI_0001.JPG", "stem": "DJI_0001", "split": "valid", "batch_index": 0, "mask_f1": 0.2}
            )
            writer.writerow(
                {"rank": 2, "image": "DJI_0002.JPG", "stem": "DJI_0002", "split": "valid", "batch_index": 1, "mask_f1": 0.4}
            )
        (review / "review_state.json").write_text(
            json.dumps(
                {
                    "decisions": {
                        "DJI_0001.JPG": {"decision": "needs_label_fix", "note": "missing"},
                        "DJI_0002.JPG": {"decision": "label_ok_hard", "note": ""},
                    }
                }
            ),
            encoding="utf-8",
        )
        self.report = self.root / "correction"
        summary = build_queue(review, self.report)
        self.assertEqual(summary["selected_for_correction"], 1)
        self.assertFalse(summary["held_out_test_used"])

        dataset_label = self.root / "datasets" / "Data_v3" / "valid" / "labels" / "DJI_0001.txt"
        dataset_image = self.root / "datasets" / "Data_v3" / "valid" / "images" / "DJI_0001.JPG"
        canonical = self.root / "datasets" / "Data_v2" / "train" / "labels" / "DJI_0001.txt"
        dataset_label.parent.mkdir(parents=True)
        dataset_image.parent.mkdir(parents=True)
        canonical.parent.mkdir(parents=True)
        dataset_label.write_text(ORIGINAL, encoding="utf-8")
        canonical.write_text(ORIGINAL, encoding="utf-8")
        dataset_image.write_bytes(b"fake-jpeg")
        reference = self.root / "reference"
        reference.mkdir()
        (reference / "val_batch0_pred.jpg").write_bytes(b"fake-jpeg")
        self.dataset_label = dataset_label
        self.canonical = canonical
        self.state = self.report / "correction_state.json"
        self.app = create_app(
            self.report / "correction_queue.csv",
            self.state,
            self.root / "datasets" / "Data_v3",
            reference,
            self.root,
        )
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_save_syncs_both_labels_and_undo_restores_them(self) -> None:
        polygons = [
            {
                "class_id": "0",
                "points": [[0.2, 0.2], [0.5, 0.2], [0.4, 0.6]],
            }
        ]
        response = self.client.post("/api/item/DJI_0001.JPG/save", json={"polygons": polygons})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"]["completed"], 1)
        self.assertEqual(self.dataset_label.read_text(encoding="utf-8"), self.canonical.read_text(encoding="utf-8"))
        self.assertNotEqual(self.dataset_label.read_text(encoding="utf-8"), ORIGINAL)

        undo = self.client.post("/api/undo", json={})
        self.assertEqual(undo.status_code, 200)
        self.assertEqual(self.dataset_label.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.canonical.read_text(encoding="utf-8"), ORIGINAL)

    def test_self_intersection_is_rejected_without_writes(self) -> None:
        crossing = [{"class_id": "0", "points": [[0.1, 0.1], [0.8, 0.8], [0.1, 0.8], [0.8, 0.1]]}]
        response = self.client.post("/api/item/DJI_0001.JPG/save", json={"polygons": crossing})
        self.assertEqual(response.status_code, 400)
        self.assertIn("第 1 个轮廓", response.get_json()["error"])
        self.assertEqual(self.dataset_label.read_text(encoding="utf-8"), ORIGINAL)
        self.assertEqual(self.canonical.read_text(encoding="utf-8"), ORIGINAL)

    def test_unfinished_draft_survives_app_restart(self) -> None:
        workspace = {
            "polygons": [{"class_id": "0", "points": [[0.2, 0.2], [0.5, 0.2], [0.4, 0.6]]}],
            "draft": [[0.1, 0.1], [0.2, 0.1]],
            "mode": "add",
            "selected": 0,
        }
        response = self.client.post("/api/item/DJI_0001.JPG/draft", json=workspace)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"]["drafts"], 1)

        restarted = create_app(
            self.report / "correction_queue.csv",
            self.state,
            self.root / "datasets" / "Data_v3",
            self.root / "reference",
            self.root,
        )
        restarted.config["TESTING"] = True
        restored = restarted.test_client().get("/api/item/DJI_0001.JPG").get_json()
        self.assertTrue(restored["has_draft"])
        self.assertEqual(restored["workspace"]["mode"], "add")
        self.assertEqual(len(restored["workspace"]["draft"]), 2)

        cleared = restarted.test_client().delete("/api/item/DJI_0001.JPG/draft")
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(cleared.get_json()["status"]["drafts"], 0)


if __name__ == "__main__":
    unittest.main()
