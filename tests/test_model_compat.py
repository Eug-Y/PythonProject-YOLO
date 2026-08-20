from __future__ import annotations

import unittest
import json
from pathlib import Path

from train_ca_5ch import (
    SUPPORTED_SEGMENTATION_MODELS,
    resolve_base_checkpoint,
    validate_attention_compatibility,
)


class ModelCompatibilityTest(unittest.TestCase):
    def test_legacy_architecture_flags_still_resolve(self) -> None:
        self.assertEqual(resolve_base_checkpoint(None, "yolo11", "n"), "yolo11n-seg.pt")
        self.assertEqual(resolve_base_checkpoint(None, "yolov8", "s"), "yolov8s-seg.pt")

    def test_explicit_modern_model_overrides_legacy_flags(self) -> None:
        self.assertEqual(
            resolve_base_checkpoint("yolo26n-seg.pt", "yolo11", "s"),
            "yolo26n-seg.pt",
        )

    def test_all_expected_families_are_available(self) -> None:
        for prefix in ("yolo11", "yolov8", "yolo12", "yolo26", "yolov9"):
            self.assertTrue(any(model.startswith(prefix) for model in SUPPORTED_SEGMENTATION_MODELS))

    def test_yolov9_requires_native_attention(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention none"):
            validate_attention_compatibility("yolov9c-seg.pt", "ca_eca")
        validate_attention_compatibility("yolov9c-seg.pt", "none")

    def test_model_matrix_matches_cli_choices(self) -> None:
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads((root / "configs" / "yolo_segmentation_model_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(set(matrix["models"]), set(SUPPORTED_SEGMENTATION_MODELS))


if __name__ == "__main__":
    unittest.main()
