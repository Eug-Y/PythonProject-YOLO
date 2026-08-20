from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from deployment_fusion import PROJECT_ROOT, build_web_five_channel, load_shadow_config, shadow_status
from web_viewer import (
    PREFERRED_MODEL_NAME,
    SHADOW_FUSION_MODEL_NAME,
    app,
    discover_models,
    yolo_predict,
)
from yolo_mask_ensemble import FusedMaskPrediction


class ShadowDeploymentV27Tests(unittest.TestCase):
    def test_manifest_hashes_roles_and_rollback_are_locked(self) -> None:
        config = load_shadow_config()
        status = shadow_status()
        production = json.loads((PROJECT_ROOT / "configs" / "deployment.json").read_text(encoding="utf-8"))
        rollback = json.loads((PROJECT_ROOT / "configs" / "deployment.pre_v27.json").read_text(encoding="utf-8"))
        self.assertEqual(config["primary"]["architecture"], "YOLO11n-seg")
        self.assertEqual(config["auxiliary"]["architecture"], "YOLOv8n-seg")
        self.assertEqual(config["fusion"]["required_model_indices"], [0])
        self.assertEqual(config["fusion"]["model_weights"], [1.0, 1.0])
        self.assertTrue(status["every_output_requires_yolo11"])
        self.assertFalse(status["production_switch_allowed"])
        self.assertEqual(production["model_name"], config["model_name"])
        self.assertEqual(production["primary"]["sha256"], config["primary"]["sha256"])
        self.assertEqual(config["rollback"]["sha256"], rollback["sha256"])
        self.assertEqual(config["rollback"]["checkpoint"], rollback["checkpoint"])

    def test_web_channel_contract_encodes_indices_as_uint8(self) -> None:
        rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        rgb[..., 0] = 17
        ndvi = np.full((8, 10), 0.5, dtype=np.float32)
        ndre = np.full((8, 10), 1.0, dtype=np.float32)
        result = build_web_five_channel(rgb, ndvi, ndre)
        self.assertEqual(result.shape, (8, 10, 5))
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[0, 0, 0]), 17)
        self.assertEqual(int(result[0, 0, 3]), 128)
        self.assertEqual(int(result[0, 0, 4]), 255)

    def test_promoted_fusion_is_the_default_production_model(self) -> None:
        models = discover_models()
        self.assertEqual(models[0]["name"], PREFERRED_MODEL_NAME)
        shadow = next(item for item in models if item["name"] == SHADOW_FUSION_MODEL_NAME)
        self.assertFalse(shadow["shadow"])
        self.assertTrue(shadow["recommended"])
        self.assertEqual(shadow["primary"], "YOLO11")
        self.assertEqual(shadow["auxiliary"], "YOLOv8")

    def test_shadow_renderer_uses_fused_predictions(self) -> None:
        image = np.zeros((32, 40, 3), dtype=np.uint8)
        mask = np.zeros((32, 40), dtype=bool)
        mask[5:20, 8:30] = True
        prediction = FusedMaskPrediction(mask, 0.83, 0, ("yolo11_primary",), (8, 5, 30, 20))
        layers = np.full((32, 40), 0.5, dtype=np.float32)
        with patch("deployment_fusion.infer_shadow_fusion", return_value=([prediction], {"total_ms": 1.0})):
            overlay, count, used = yolo_predict(SHADOW_FUSION_MODEL_NAME, image, layers, layers)
        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(count, 1)
        self.assertEqual(used, SHADOW_FUSION_MODEL_NAME)
        self.assertTrue(overlay.any())

    def test_shadow_status_route_verifies_artifacts(self) -> None:
        response = app.test_client().get("/api/shadow-status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["model_name"], SHADOW_FUSION_MODEL_NAME)
        self.assertTrue(payload["validation_gate_passed"])
        self.assertTrue(payload["every_output_requires_yolo11"])


if __name__ == "__main__":
    unittest.main()
