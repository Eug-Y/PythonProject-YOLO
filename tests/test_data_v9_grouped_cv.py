from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CV_ROOT = ROOT / "configs" / "generated" / "data_v9_grouped_cv"
CONTRACT = CV_ROOT / "contract.json"


class GroupedCVContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.records = cls.payload["records"]
        cls.all_stems = {row["stem"] for row in cls.records}

    def test_contract_is_train_valid_only(self) -> None:
        self.assertEqual(self.payload["source_splits"], ["train", "valid"])
        self.assertFalse(self.payload["held_out_test_used"])
        self.assertFalse(self.payload["test_paths_enumerated"])
        self.assertEqual(self.payload["source_images"], 270)
        for row in self.records:
            self.assertIn(row["source_split"], {"train", "valid"})
            self.assertNotIn("/test/", Path(row["image"]).as_posix().lower())
            self.assertNotIn("/test/", Path(row["human_image"]).as_posix().lower())

    def test_validation_folds_are_disjoint_and_cover_every_image(self) -> None:
        validation_sets = [set(fold["validation_stems"]) for fold in self.payload["fold_contracts"]]
        self.assertEqual(set().union(*validation_sets), self.all_stems)
        for index, stems in enumerate(validation_sets):
            for other in validation_sets[index + 1 :]:
                self.assertTrue(stems.isdisjoint(other))

    def test_each_train_list_is_the_validation_complement(self) -> None:
        for fold in self.payload["fold_contracts"]:
            train_stems = {Path(line).stem for line in Path(fold["train_list"]).read_text(encoding="utf-8").splitlines()}
            human_train_stems = {
                Path(line).stem for line in Path(fold["human_train_list"]).read_text(encoding="utf-8").splitlines()
            }
            val_stems = {Path(line).stem for line in Path(fold["validation_list"]).read_text(encoding="utf-8").splitlines()}
            self.assertTrue(train_stems.isdisjoint(val_stems))
            self.assertEqual(train_stems | val_stems, self.all_stems)
            self.assertEqual(train_stems, human_train_stems)
            self.assertEqual(len(train_stems), fold["train_images"])
            self.assertEqual(len(val_stems), fold["validation_images"])
            self.assertNotIn("test:", Path(fold["data_yaml"]).read_text(encoding="utf-8").lower())
            self.assertNotIn("test:", Path(fold["human_data_yaml"]).read_text(encoding="utf-8").lower())

    def test_every_validation_path_uses_human_labels(self) -> None:
        human_path_by_stem = {row["stem"]: Path(row["human_image"]).resolve() for row in self.records}
        for fold in self.payload["fold_contracts"]:
            for line in Path(fold["validation_list"]).read_text(encoding="utf-8").splitlines():
                path = Path(line).resolve()
                self.assertEqual(path, human_path_by_stem[path.stem])

    def test_sequence_groups_never_cross_a_fold_boundary(self) -> None:
        group_by_stem = {row["stem"]: row["sequence_group"] for row in self.records}
        for fold in self.payload["fold_contracts"]:
            val_stems = set(fold["validation_stems"])
            train_stems = self.all_stems - val_stems
            val_groups = {group_by_stem[stem] for stem in val_stems}
            train_groups = {group_by_stem[stem] for stem in train_stems}
            self.assertTrue(val_groups.isdisjoint(train_groups))

    def test_no_numeric_neighbor_within_gap_crosses_train_and_validation(self) -> None:
        rows = sorted(self.records, key=lambda row: row["number"])
        for fold in self.payload["fold_contracts"]:
            val_stems = set(fold["validation_stems"])
            for left in rows:
                for right in rows:
                    if right["number"] <= left["number"]:
                        continue
                    gap = right["number"] - left["number"]
                    if gap > self.payload["sequence_group_max_gap"]:
                        break
                    self.assertEqual(left["stem"] in val_stems, right["stem"] in val_stems)

    def test_all_referenced_artifacts_exist(self) -> None:
        for row in self.records:
            self.assertTrue(Path(row["image"]).is_file())
            self.assertTrue(Path(row["label"]).is_file())
            self.assertTrue(Path(row["human_image"]).is_file())
            self.assertTrue(Path(row["human_label"]).is_file())
            self.assertTrue(Path(row["feature"]).is_file())

    def test_paired_variants_change_labels_not_image_bytes(self) -> None:
        for row in self.records:
            self.assertEqual(row["image_sha256"], row["human_image_sha256"])


if __name__ == "__main__":
    unittest.main()
