import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from release_blind_annotation import (
    ATTESTATION,
    BlindAnnotationStore,
    annotation_warnings,
    create_app,
    verify_selection,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseBlindAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("test/images", "test/features", "test/labels"):
            (self.root / folder).mkdir(parents=True)
        records = []
        self.stems = [f"DJI_{number:04d}" for number in range(10, 310, 10)]
        for stem in self.stems:
            image = self.root / "test/images" / f"{stem}.JPG"
            feature = self.root / "test/features" / f"{stem}.npy"
            image.write_bytes(("image-" + stem).encode())
            feature.write_bytes(("feature-" + stem).encode())
            records.append({"stem": stem, "image_sha256": digest(image), "feature_sha256": digest(feature)})
        signature = hashlib.sha256("\n".join(self.stems).encode()).hexdigest()
        manifest = {"version": "release_blind_holdout_v1", "selected_images": 30,
                    "selected_stem_signature": signature, "records": records}
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_verify_selection_detects_hash_drift(self) -> None:
        result = verify_selection(self.root)
        self.assertEqual(result["images_verified"], 30)
        (self.root / "test/features/DJI_0010.npy").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "特征哈希漂移"):
            verify_selection(self.root)

    def test_empty_needs_confirmation_and_complete_set_can_seal(self) -> None:
        store = BlindAnnotationStore(self.root)
        with self.assertRaisesRegex(ValueError, "空场景必须"):
            store.save_item(self.stems[0], [], "human", False)
        polygon = [{"class_id": "0", "points": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}]
        store.save_item(self.stems[0], polygon, "human")
        for stem in self.stems[1:]:
            store.save_item(stem, [], "human", True)
        sealed = store.seal("reviewer", ATTESTATION)
        self.assertEqual(sealed["images"], 30)
        self.assertFalse(sealed["model_predictions_exposed"])
        self.assertTrue(store.status()["sealed"])
        with self.assertRaisesRegex(ValueError, "已封存"):
            store.save_item(self.stems[0], polygon, "human")

    def test_web_app_has_no_prediction_endpoint(self) -> None:
        app = create_app(self.root)
        client = app.test_client()
        self.assertEqual(client.get("/api/items").status_code, 200)
        self.assertEqual(client.get("/api/prediction/DJI_0010").status_code, 404)
        response = client.post("/api/item/DJI_0010/save", json={"polygons": [], "actor": "h"})
        self.assertEqual(response.status_code, 400)

    def test_completed_item_exposes_tiny_polygon_warning(self) -> None:
        store = BlindAnnotationStore(self.root)
        tiny = [{"class_id": "0", "points": [[0.1, 0.1], [0.102, 0.1], [0.1, 0.102]]}]
        saved = store.save_item(self.stems[0], tiny, "human")
        self.assertTrue(saved["has_warning"])
        self.assertIn("tiny_polygon", {warning["code"] for warning in saved["warnings"]})
        listed = store.list_items()
        self.assertTrue(listed["items"][0]["has_warning"])
        self.assertEqual(listed["status"]["warning_images"], 1)
        self.assertGreaterEqual(listed["status"]["warning_count"], 1)

    def test_duplicate_completed_polygons_are_flagged_for_review(self) -> None:
        polygon = {"class_id": "0", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]}
        warnings = annotation_warnings([polygon, polygon])
        self.assertIn("duplicate_polygon", {warning["code"] for warning in warnings})

    def test_normal_completed_polygon_has_no_warning(self) -> None:
        store = BlindAnnotationStore(self.root)
        polygon = [{"class_id": "0", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]}]
        saved = store.save_item(self.stems[0], polygon, "human")
        self.assertFalse(saved["has_warning"])
        self.assertEqual(store.status()["warning_images"], 0)


if __name__ == "__main__":
    unittest.main()
