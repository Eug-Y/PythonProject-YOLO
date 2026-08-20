from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from difficult_sample_review_app import create_app


class DifficultSampleReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.queue = root / "review_queue.csv"
        with self.queue.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rank", "image", "batch_index", "mask_f1"])
            writer.writeheader()
            writer.writerow({"rank": 1, "image": "DJI_0001.JPG", "batch_index": 0, "mask_f1": 0.25})
        self.state = root / "review_state.json"
        self.app = create_app(self.queue, self.state, root, root)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_decision_is_persisted_and_undoable(self) -> None:
        initial = self.client.get("/api/queue").get_json()
        self.assertEqual(initial["status"]["remaining"], 1)
        response = self.client.post(
            "/api/decision",
            json={"image": "DJI_0001.JPG", "decision": "needs_label_fix", "note": "road edge"},
        )
        self.assertEqual(response.status_code, 200)
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(saved["decisions"]["DJI_0001.JPG"]["note"], "road edge")
        self.assertEqual(self.client.get("/api/queue").get_json()["status"]["completed"], 1)
        self.assertEqual(self.client.post("/api/undo", json={"image": "DJI_0001.JPG"}).status_code, 200)
        self.assertEqual(self.client.get("/api/queue").get_json()["status"]["remaining"], 1)

    def test_invalid_decision_is_rejected(self) -> None:
        response = self.client.post(
            "/api/decision", json={"image": "DJI_0001.JPG", "decision": "invalid"}
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
