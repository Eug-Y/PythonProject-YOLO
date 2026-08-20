from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from audit_terrain_sources import audit_terrain_sources
from run_terrain_model_screen import training_command


class TerrainSourceAuditTest(unittest.TestCase):
    def test_reports_ready_missing_and_mismatched_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "dataset" / "train" / "images"
            terrain_dir = root / "terrain" / "train"
            image_dir.mkdir(parents=True)
            terrain_dir.mkdir(parents=True)
            image = np.full((12, 16, 3), 120, dtype=np.uint8)
            for stem in ("DJI_0010", "DJI_0020", "DJI_0030"):
                self.assertTrue(cv2.imwrite(str(image_dir / f"{stem}.JPG"), image))
            np.save(terrain_dir / "DJI_0010.npy", np.arange(192, dtype=np.float32).reshape(12, 16))
            np.save(terrain_dir / "DJI_0030.npy", np.ones((6, 8), dtype=np.float32))

            summary, rows = audit_terrain_sources(
                root / "dataset", root / "terrain", splits=("train",), pixel_size_m=0.1
            )

        self.assertEqual(summary["status_counts"], {"ready": 1, "missing": 1, "shape_mismatch": 1})
        self.assertFalse(summary["can_prepare_candidates"])
        self.assertEqual({row["status"] for row in rows}, {"ready", "missing", "shape_mismatch"})


class TerrainModelRunnerTest(unittest.TestCase):
    def test_command_uses_locked_configs_and_requested_model(self) -> None:
        command = training_command(
            group="slope_6ch",
            model="yolov8n-seg.pt",
            attention="none",
            epochs=3,
            imgsz=640,
            device="0",
            seed=2,
            batch=None,
        )
        text = " ".join(command)
        self.assertIn("--base-model yolov8n-seg.pt", text)
        self.assertIn("--group slope_6ch", text)
        self.assertIn("feature_ablation_groups_v6_terrain.json", text)
        self.assertNotIn("--batch", command)
        self.assertNotIn("test", [part.lower() for part in command])

    def test_optional_batch_is_forwarded(self) -> None:
        command = training_command(
            group="current_5ch",
            model="yolov9c-seg.pt",
            attention="none",
            epochs=1,
            imgsz=320,
            device="cpu",
            seed=0,
            batch=1,
        )
        self.assertEqual(command[-2:], ["--batch", "1"])


if __name__ == "__main__":
    unittest.main()

