"""Validated runtime for the frozen YOLO11-primary V27 shadow candidate."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from export_multichannel_predictions import infer_masks
from yolo_mask_ensemble import FusedMaskPrediction, MaskPrediction, fuse_mask_predictions


PROJECT_ROOT = Path(__file__).resolve().parent
SHADOW_CONFIG_PATH = PROJECT_ROOT / "configs" / "shadow_deployment_v27.json"
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Shadow deployment path leaves project root: {value}") from exc
    return path


def load_shadow_config(path: Path = SHADOW_CONFIG_PATH, *, verify_hashes: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "shadow" or payload.get("validation_gate_passed") is not True:
        raise RuntimeError("Shadow candidate is not validation-gated")
    if payload.get("every_output_requires_yolo11") is not True:
        raise RuntimeError("Shadow candidate does not guarantee YOLO11 ownership")
    if payload.get("production_switch_allowed") is not False:
        raise RuntimeError("Unreviewed shadow config cannot authorize a production switch")
    if payload.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError("Unexpected shadow feature contract")
    primary_path = _resolve_project_path(payload["primary"]["checkpoint"])
    auxiliary_path = _resolve_project_path(payload["auxiliary"]["checkpoint"])
    fusion_path = _resolve_project_path(payload["fusion_config"])
    for item in (primary_path, auxiliary_path, fusion_path):
        if not item.is_file():
            raise FileNotFoundError(item)
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    if fusion.get("role") != "strict_yolo11_primary":
        raise RuntimeError("V27 runtime requires the strict YOLO11-primary role")
    if fusion.get("required_model_indices") != [0] or fusion.get("model_weights") != [1.0, 1.0]:
        raise RuntimeError("V27 runtime requires YOLO11 and exact 1:1 fusion weights")
    if verify_hashes:
        expected = (
            (primary_path, payload["primary"]["sha256"]),
            (auxiliary_path, payload["auxiliary"]["sha256"]),
            (fusion_path, payload["fusion_config_sha256"]),
        )
        for item, expected_hash in expected:
            actual = sha256(item)
            if actual != expected_hash:
                raise RuntimeError(f"Shadow artifact hash drifted: {item} expected={expected_hash} actual={actual}")
    return {
        **payload,
        "primary_path": primary_path,
        "auxiliary_path": auxiliary_path,
        "fusion_path": fusion_path,
        "fusion": fusion,
    }


def shadow_status(*, verify_hashes: bool = True) -> dict[str, Any]:
    config = load_shadow_config(verify_hashes=verify_hashes)
    return {
        "state": config["state"],
        "model_name": config["model_name"],
        "release_candidate": config["release_candidate"],
        "primary_architecture": config["primary"]["architecture"],
        "primary_sha256": config["primary"]["sha256"],
        "auxiliary_architecture": config["auxiliary"]["architecture"],
        "auxiliary_sha256": config["auxiliary"]["sha256"],
        "fusion_config_sha256": config["fusion_config_sha256"],
        "every_output_requires_yolo11": config["every_output_requires_yolo11"],
        "validation_gate_passed": config["validation_gate_passed"],
        "release_blind_holdout_used": config["release_blind_holdout_used"],
        "production_switch_allowed": config["production_switch_allowed"],
        "rollback": config["rollback"],
    }


def build_web_five_channel(image_rgb: np.ndarray, ndvi: np.ndarray, ndre: np.ndarray) -> np.ndarray:
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("Web RGB image must be uint8 HxWx3")
    height, width = image_rgb.shape[:2]
    encoded = []
    for name, layer in (("NDVI", ndvi), ("NDRE", ndre)):
        values = np.asarray(layer, dtype=np.float32)
        if values.shape != (height, width) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite HxW layer matching RGB")
        encoded.append(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.dstack((image_rgb, encoded[0], encoded[1]))


def _get_checkpoint_model(cache_key: str, checkpoint: Path, device):
    key = (cache_key, str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    # Imports register the custom attention modules referenced by both frozen checkpoints.
    from custom_modules import CoordAtt, ECA  # noqa: F401
    from ultralytics.nn.tasks import load_checkpoint

    model, _ = load_checkpoint(str(checkpoint), device=device)
    if int(model.model[0].conv.in_channels) != 5:
        raise RuntimeError(f"Shadow checkpoint does not accept five channels: {checkpoint}")
    model.float().to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def infer_shadow_components(
    image_5ch: np.ndarray, *, device_name: str | None = None
) -> tuple[list[list[MaskPrediction]], list[FusedMaskPrediction], dict[str, Any]]:
    import torch

    config = load_shadow_config()
    image = np.asarray(image_5ch)
    if image.ndim != 3 or image.shape[2] != 5:
        raise ValueError(f"V27 shadow inference expects HxWx5 input, got {image.shape}")
    if image.dtype != np.uint8:
        values = image.astype(np.float32)
        if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
            raise ValueError("Non-uint8 shadow input must be finite and normalized to [0, 1]")
        image = np.rint(values * 255.0).astype(np.uint8)
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    primary = _get_checkpoint_model("v27_yolo11_primary", config["primary_path"], device)
    auxiliary = _get_checkpoint_model("v27_yolov8_auxiliary", config["auxiliary_path"], device)
    fusion = config["fusion"]
    started = time.perf_counter()
    outputs = []
    per_model_ms = []
    for name, model in (("yolo11_primary", primary), ("yolov8_auxiliary", auxiliary)):
        stage_started = time.perf_counter()
        result = infer_masks(
            model,
            image,
            imgsz=int(config["imgsz"]),
            conf=float(fusion["score_threshold"]),
            iou=float(config["nms_iou"]),
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_model_ms.append((time.perf_counter() - stage_started) * 1000.0)
        outputs.append([
            MaskPrediction(mask, float(score), int(class_id), name)
            for mask, score, class_id in zip(result["masks"], result["scores"], result["class_ids"])
        ])
    fused = fuse_mask_predictions(
        outputs,
        model_weights=tuple(fusion["model_weights"]),
        match_iou=float(fusion["match_iou"]),
        mask_vote=float(fusion["mask_vote"]),
        score_threshold=float(fusion["score_threshold"]),
        min_models=int(fusion["min_models"]),
        required_model_indices=tuple(fusion["required_model_indices"]),
        unconfirmed_score_threshold=float(fusion["unconfirmed_score_threshold"]),
    )
    if any("yolo11_primary" not in item.sources for item in fused):
        raise RuntimeError("YOLO11 ownership invariant failed at runtime")
    metadata = {
        "model_name": config["model_name"],
        "primary_detections": len(outputs[0]),
        "auxiliary_detections": len(outputs[1]),
        "fused_detections": len(fused),
        "primary_ms": per_model_ms[0],
        "auxiliary_ms": per_model_ms[1],
        "total_ms": (time.perf_counter() - started) * 1000.0,
        "every_output_requires_yolo11": True,
    }
    return outputs, fused, metadata


def infer_shadow_fusion(
    image_5ch: np.ndarray, *, device_name: str | None = None
) -> tuple[list[FusedMaskPrediction], dict[str, Any]]:
    _components, fused, metadata = infer_shadow_components(image_5ch, device_name=device_name)
    return fused, metadata
