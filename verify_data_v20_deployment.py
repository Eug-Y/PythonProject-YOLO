"""Run post-deployment model, inference, and Flask route regression checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
DEPLOYMENT_CONFIG = ROOT / "configs" / "deployment.json"
DEPLOYMENT_AUDIT = ROOT / "logs" / "data_v20_deployment.json"
SAMPLE_FEATURE = ROOT / "datasets" / "Data_v4_candidates" / "valid" / "features" / "DJI_0010.npy"
EXPERIMENT_LABEL = "V20"
RELEASE_VERSION = "v20_sparse_context_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_deployment_config(payload: dict[str, object]) -> Path:
    if payload.get("release_version") != RELEASE_VERSION:
        raise RuntimeError(f"Active config is not the {EXPERIMENT_LABEL} release")
    if payload.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError("Deployment channel contract drifted")
    if float(payload.get("confidence", 0.0)) != 0.371:
        raise RuntimeError("Deployment confidence is not locked to 0.371")
    checkpoint = ROOT / str(payload["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != payload.get("sha256"):
        raise RuntimeError("Deployment checkpoint/hash mismatch")
    return checkpoint


def main() -> int:
    if not DEPLOYMENT_CONFIG.is_file() or not DEPLOYMENT_AUDIT.is_file() or not SAMPLE_FEATURE.is_file():
        raise FileNotFoundError("Deployment config, audit, or regression feature is missing")
    config = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    audit = json.loads(DEPLOYMENT_AUDIT.read_text(encoding="utf-8"))
    checkpoint = validate_deployment_config(config)
    try:
        from web_viewer import (
            DEFAULT_YOLO_CONF,
            PREFERRED_MODEL_NAME,
            app,
            discover_models,
            yolo_predict,
        )

        if PREFERRED_MODEL_NAME != config["model_name"] or DEFAULT_YOLO_CONF != 0.371:
            raise RuntimeError(f"Web process did not load the {EXPERIMENT_LABEL} deployment config")
        models = discover_models()
        preferred = [row for row in models if row["name"] == PREFERRED_MODEL_NAME and row["recommended"]]
        if len(preferred) != 1:
            raise RuntimeError(f"{EXPERIMENT_LABEL} release is not the unique recommended web model")
        source = np.load(SAMPLE_FEATURE, mmap_mode="r", allow_pickle=False)
        if source.shape[2] < 7 or source.dtype != np.uint8:
            raise RuntimeError("Regression feature contract drifted")
        image_rgb = np.ascontiguousarray(source[..., [0, 1, 2]])
        ndvi = np.asarray(source[..., 5], dtype=np.float32) / 255.0
        ndre = np.asarray(source[..., 6], dtype=np.float32) / 255.0
        overlay, detections, used_model = yolo_predict(
            PREFERRED_MODEL_NAME,
            image_rgb,
            ndvi,
            ndre,
            conf=DEFAULT_YOLO_CONF,
            imgsz=1024,
        )
        if overlay is None or overlay.shape != image_rgb.shape or not isinstance(detections, int):
            raise RuntimeError(f"{EXPERIMENT_LABEL} web inference regression failed: {used_model}")
        if used_model != PREFERRED_MODEL_NAME:
            raise RuntimeError("Inference returned the wrong model name")
        client = app.test_client()
        status_response = client.get("/api/project-status")
        models_response = client.get("/api/models")
        page_response = client.get("/")
        if status_response.status_code != 200 or models_response.status_code != 200 or page_response.status_code != 200:
            raise RuntimeError("Flask deployment routes failed")
        status = status_response.get_json()
        if status.get("deployment_model") != PREFERRED_MODEL_NAME or float(status.get("deployment_confidence")) != 0.371:
            raise RuntimeError(f"Flask status does not expose the active {EXPERIMENT_LABEL} release")
        completed_at = __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")
        audit.update(
            inference_regression_completed=True,
            inference_regression_at=completed_at,
            inference_sample=str(SAMPLE_FEATURE.resolve()),
            inference_detections=detections,
            flask_routes_verified=["/", "/api/models", "/api/project-status"],
            release_checkpoint_hash_verified=sha256(checkpoint) == config["sha256"],
            rollback_available=True,
            web_restart_required=True,
        )
        write_json_atomic(DEPLOYMENT_AUDIT, audit)
        print(json.dumps({"verified": True, "model": PREFERRED_MODEL_NAME, "detections": detections}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        rollback = config.get("rollback_config")
        if isinstance(rollback, dict):
            write_json_atomic(DEPLOYMENT_CONFIG, rollback)
            audit.update(
                inference_regression_completed=False,
                inference_regression_error=f"{type(exc).__name__}: {exc}",
                automatic_rollback_applied=True,
                rollback_model=rollback.get("model_name"),
                web_restart_required=True,
            )
            write_json_atomic(DEPLOYMENT_AUDIT, audit)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
