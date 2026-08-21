from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from spectral_indices import compute_indices, normalize_dn
from web_viewer import (
    DEFAULT_YOLO_CONF,
    PREFERRED_MODEL_NAME,
    app,
    build_project_status,
    compute_web_indices,
    discover_models,
    make_index_heatmap,
    make_thermal_view,
)


class WebSpectralLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_num = 100
        self.raw = {
            101: np.asarray([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=np.uint16),  # Blue
            102: np.asarray([[3000, 3500, 4000], [4500, 5000, 5500]], dtype=np.uint16),  # Green
            103: np.asarray([[2000, 2500, 3000], [3500, 4000, 4500]], dtype=np.uint16),  # Red
            104: np.asarray([[2500, 3000, 3500], [4000, 4500, 5000]], dtype=np.uint16),  # RedEdge
            105: np.asarray([[5000, 5500, 6000], [6500, 7000, 7500]], dtype=np.uint16),  # NIR
        }

    def test_web_uses_shared_registry_values(self) -> None:
        names = ("NDVI", "GNDVI", "EVI2", "RE_BSI_PROXY")
        with patch("web_viewer._load_tif", side_effect=lambda number: self.raw.get(number)):
            actual = compute_web_indices(self.img_num, 2, 3, names)
        bands = {
            "blue": normalize_dn(self.raw[101]),
            "green": normalize_dn(self.raw[102]),
            "red": normalize_dn(self.raw[103]),
            "rededge": normalize_dn(self.raw[104]),
            "nir": normalize_dn(self.raw[105]),
        }
        expected = compute_indices(list(names), bands, encoded=True)
        self.assertEqual(set(actual), set(names))
        for name in names:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0.0, atol=1e-7)

    def test_missing_required_band_returns_no_partial_layers(self) -> None:
        with patch("web_viewer._load_tif", return_value=None):
            self.assertEqual(compute_web_indices(self.img_num, 2, 3, ("NDVI",)), {})

    def test_request_band_cache_reuses_nir_for_thermal_layer(self) -> None:
        cache = {}
        with patch("web_viewer._load_tif", side_effect=lambda number: self.raw.get(number)) as loader:
            actual = compute_web_indices(self.img_num, 2, 3, ("NDVI", "NDRE", "ExG"), band_cache=cache)
            thermal = make_thermal_view(self.img_num, 2, 3, nir=cache["nir"])
        self.assertEqual(set(actual), {"NDVI", "NDRE", "ExG"})
        self.assertEqual(loader.call_count, 5)
        self.assertEqual(thermal.shape, (42, 3, 3))

    def test_heatmap_contract(self) -> None:
        values = np.linspace(0.0, 1.0, 120, dtype=np.float32).reshape(10, 12)
        for name in ("NDVI", "RE_BSI_PROXY"):
            with self.subTest(name=name):
                result = make_index_heatmap(values, name)
                self.assertEqual(result.shape, (58, 12, 3))
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(np.isfinite(result).all())


class WebProjectStatusTest(unittest.TestCase):
    def test_status_keeps_deployment_model_after_failed_candidate(self) -> None:
        status = build_project_status()
        self.assertEqual(status["deployment_model"], PREFERRED_MODEL_NAME)
        self.assertEqual(status["deployment_confidence"], DEFAULT_YOLO_CONF)
        self.assertTrue(status["decision_zh"])
        self.assertGreaterEqual(len(status["reports"]), 5)

    def test_research_multichannel_models_are_not_selectable(self) -> None:
        names = {item["name"] for item in discover_models()}
        self.assertIn(PREFERRED_MODEL_NAME, names)
        self.assertFalse(any(name.startswith("v4_") for name in names))
        self.assertFalse(any(name.startswith("v5_") for name in names))

    def test_status_and_report_routes(self) -> None:
        client = app.test_client()
        status_response = client.get("/api/project-status")
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.get_json()["deployment_model"], PREFERRED_MODEL_NAME)
        report_response = client.get("/reports/Data_v5_final_model_comparison/index.html")
        self.assertEqual(report_response.status_code, 200)
        report_response.close()
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("V27 YOLO11 主导融合生产状态".encode("utf-8"), page.data)
        self.assertIn("杂草候选".encode("utf-8"), page.data)
        self.assertNotIn(b"v5_final_rgb_mcari_1024_seed0_v1</option>", page.data)
        self.assertIn("RE-BSI土壤代理".encode("utf-8"), page.data)
        page.close()


if __name__ == "__main__":
    unittest.main()
