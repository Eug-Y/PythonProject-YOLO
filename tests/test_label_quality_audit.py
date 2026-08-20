from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from label_quality_audit import _review_item_id, load_review_acceptances, parse_label


BOW_TIE = "0 0.100000 0.100000 0.400000 0.400000 0.100000 0.400000 0.400000 0.100000"


class ReviewAcceptanceTest(unittest.TestCase):
    def test_empty_label_is_a_valid_background_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            label = Path(temporary) / "background.txt"
            label.write_text("", encoding="utf-8")
            issues = []
            polygons = parse_label(
                label, "Data_v3", "valid", "background", issues,
                expected_class=0, min_area=1e-5, max_points=200,
            )
            self.assertEqual(polygons, [])
            self.assertEqual([(issue.severity, issue.code) for issue in issues], [("warning", "empty_label")])

    def test_kept_warning_is_suppressed_by_exact_row_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label = root / "DJI_0001.txt"
            geometry = root / "geometry.csv"
            state = root / "state.json"
            label.write_text(BOW_TIE + "\n", encoding="utf-8")
            with geometry.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["stem", "split", "line", "codes"])
                writer.writeheader()
                writer.writerow({
                    "stem": "DJI_0001",
                    "split": "train",
                    "line": 1,
                    "codes": "self_intersection",
                })
            item_id = _review_item_id("train", "DJI_0001", 1, "self_intersection")
            line_hash = hashlib.sha256(BOW_TIE.encode("utf-8")).hexdigest()
            state.write_text(json.dumps({
                "version": 1,
                "items": {item_id: {
                    "status": "kept",
                    "current_hash": line_hash,
                    "current_hashes": [line_hash],
                }},
            }), encoding="utf-8")

            accepted, item_count = load_review_acceptances(state, geometry)
            issues = []
            polygons = parse_label(
                label, "Data_v3", "train", "DJI_0001", issues,
                expected_class=0, min_area=1e-5, max_points=200, accepted_issues=accepted,
            )
            self.assertEqual(item_count, 1)
            self.assertEqual(len(polygons), 1)
            self.assertEqual([issue.code for issue in issues], ["tiny_polygon"])

            label.write_text(BOW_TIE.replace("0.400000 0.100000", "0.410000 0.100000") + "\n", encoding="utf-8")
            changed_issues = []
            parse_label(
                label, "Data_v3", "train", "DJI_0001", changed_issues,
                expected_class=0, min_area=1e-5, max_points=200, accepted_issues=accepted,
            )
            self.assertEqual([issue.code for issue in changed_issues], ["self_intersection"])


if __name__ == "__main__":
    unittest.main()
