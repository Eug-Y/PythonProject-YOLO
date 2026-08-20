import sys
import unittest
from unittest.mock import patch

from error_analysis_multichannel import parse_args


class ErrorAnalysisMultichannelGuardTests(unittest.TestCase):
    def test_validation_is_default_and_test_requires_explicit_flag(self) -> None:
        argv = [
            "error_analysis_multichannel.py",
            "--checkpoint", "model.pt",
            "--data", "data.yaml",
            "--manifest", "manifest.json",
            "--feature-root", "features",
            "--channels", "R", "G", "B", "NDVI", "NDRE",
            "--output", "output",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.split, "val")
        self.assertFalse(args.allow_test)

    def test_explicit_test_flags_parse_together(self) -> None:
        argv = [
            "error_analysis_multichannel.py",
            "--checkpoint", "model.pt",
            "--data", "data.yaml",
            "--manifest", "manifest.json",
            "--feature-root", "features",
            "--channels", "R", "G", "B", "NDVI", "NDRE",
            "--output", "output",
            "--split", "test",
            "--allow-test",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.split, "test")
        self.assertTrue(args.allow_test)


if __name__ == "__main__":
    unittest.main()
