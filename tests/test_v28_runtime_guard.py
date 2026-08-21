from __future__ import annotations

import unittest
from pathlib import Path

from benchmark_v28_runtime import PROJECT_ROOT, validate_image_path


class V28RuntimeGuardTests(unittest.TestCase):
    def test_valid_image_is_allowed(self) -> None:
        path = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images" / "DJI_0010.JPG"
        self.assertEqual(validate_image_path(path), path.resolve())

    def test_old_test_is_rejected(self) -> None:
        path = PROJECT_ROOT / "datasets" / "Data_v3" / "test" / "images" / "DJI_1960.JPG"
        with self.assertRaisesRegex(ValueError, "test/blind"):
            validate_image_path(path)

    def test_release_blind_is_rejected(self) -> None:
        path = PROJECT_ROOT / "datasets" / "Data_release_blind_v1" / "test" / "images" / "DJI_1960.JPG"
        with self.assertRaisesRegex(ValueError, "test/blind"):
            validate_image_path(path)


if __name__ == "__main__":
    unittest.main()
