from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from terrain_factors import TERRAIN_CHANNELS, TerrainDataError, compute_terrain_factors, load_elevation
from train_feature_ablation import resolve_selection_yaml


class TerrainFactorTest(unittest.TestCase):
    def test_flat_surface_has_zero_slope_and_neutral_aspect(self) -> None:
        elevation = np.full((9, 11), 100.0, dtype=np.float32)
        factors = compute_terrain_factors(elevation, pixel_size_m=0.1)
        self.assertEqual(tuple(factors), TERRAIN_CHANNELS)
        np.testing.assert_allclose(factors["SLOPE"], 0.0, atol=1e-6)
        np.testing.assert_allclose(factors["ASPECT_SIN"], 0.5, atol=1e-6)
        np.testing.assert_allclose(factors["ASPECT_COS"], 0.5, atol=1e-6)
        np.testing.assert_allclose(factors["TRI"], 0.0, atol=1e-6)
        np.testing.assert_allclose(factors["TPI"], 0.5, atol=1e-6)
        np.testing.assert_allclose(factors["RELIEF5"], 0.0, atol=1e-6)

    def test_planar_slope_matches_expected_angle(self) -> None:
        y, x = np.mgrid[:20, :30]
        elevation = (x * 0.1).astype(np.float32)
        factors = compute_terrain_factors(elevation, pixel_size_m=0.1)
        expected = np.arctan(1.0) / (np.pi / 2.0)
        np.testing.assert_allclose(factors["SLOPE"][2:-2, 2:-2], expected, atol=1e-5)
        for values in factors.values():
            self.assertEqual(values.dtype, np.float32)
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)

    def test_pixel_size_changes_slope_not_relative_relief(self) -> None:
        elevation = np.tile(np.arange(12, dtype=np.float32), (10, 1))
        fine = compute_terrain_factors(elevation, pixel_size_m=0.5)
        coarse = compute_terrain_factors(elevation, pixel_size_m=2.0)
        self.assertGreater(float(fine["SLOPE"].mean()), float(coarse["SLOPE"].mean()))
        np.testing.assert_allclose(fine["TRI"], coarse["TRI"])

    def test_loader_repairs_nonfinite_and_resizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "height.npy"
            elevation = np.arange(20, dtype=np.float32).reshape(4, 5)
            elevation[1, 2] = np.nan
            np.save(path, elevation)
            loaded = load_elevation(path, target_hw=(8, 10))
        self.assertEqual(loaded.shape, (8, 10))
        self.assertTrue(np.isfinite(loaded).all())

    def test_invalid_pixel_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(TerrainDataError, "positive"):
            compute_terrain_factors(np.ones((5, 5), dtype=np.float32), pixel_size_m=0)

    def test_data_v6_manifest_resolves_matching_channel_yaml(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs" / "data_v6_terrain_channels.json").read_text(encoding="utf-8"))
        for channel_count in (5, 6, 8, 10):
            resolved = resolve_selection_yaml(root / "datasets" / "missing", config, channel_count)
            self.assertEqual(resolved, root / "configs" / f"data_v6_selection_{channel_count}ch.yaml")


if __name__ == "__main__":
    unittest.main()
