"""Run direct and Flask end-to-end smoke gates for the frozen V27 shadow release."""

from __future__ import annotations

import io
import json
import statistics
import time
from pathlib import Path

from deployment_fusion import infer_shadow_fusion, load_shadow_config, sha256, shadow_status
from run_data_v9_grouped_cv import now_iso
from run_feature_ablations import write_json_atomic
from web_viewer import SHADOW_FUSION_MODEL_NAME, app
from yolo_multichannel_data import _load_selected_array, feature_path_for_image, validate_channel_selection


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "logs" / "v27_shadow_deployment_smoke.summary.json"


def main() -> int:
    shadow = load_shadow_config()
    production_path = ROOT / "configs" / "deployment.json"
    production_before = production_path.read_bytes()
    production = json.loads(production_before.decode("utf-8"))
    if sha256(ROOT / production["checkpoint"]) != production["sha256"]:
        raise RuntimeError("Production rollback checkpoint hash drifted before smoke test")

    feature_manifest = json.loads(
        (ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json").read_text(encoding="utf-8")
    )
    images = sorted((ROOT / "datasets" / "Data_v3" / "valid" / "images").glob("*.JPG"))
    if not images:
        raise FileNotFoundError("No valid image for shadow smoke test")
    image_path = images[0]
    indices = validate_channel_selection(
        tuple(feature_manifest["channels"]), tuple(shadow["channels"])
    )
    image_5ch = _load_selected_array(
        feature_path_for_image(image_path, ROOT / "datasets" / "Data_v4_candidates"),
        indices,
        len(feature_manifest["channels"]),
    )

    direct_runs = []
    for _ in range(3):
        fused, metadata = infer_shadow_fusion(image_5ch)
        if any("yolo11_primary" not in item.sources for item in fused):
            raise RuntimeError("Direct smoke produced an output without YOLO11")
        direct_runs.append(metadata)

    web_started = time.perf_counter()
    with image_path.open("rb") as stream:
        response = app.test_client().post(
            "/api/extract",
            data={
                "mode": "predict",
                "model": SHADOW_FUSION_MODEL_NAME,
                "yolo_conf": str(shadow["fusion"]["score_threshold"]),
                "threshold": "0.005",
                "gsd": "0.028",
                "ndvi_threshold": "0",
                "images": (io.BytesIO(stream.read()), image_path.name),
            },
            content_type="multipart/form-data",
        )
    web_ms = (time.perf_counter() - web_started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(f"Shadow Web smoke failed HTTP {response.status_code}: {response.get_data(as_text=True)[:500]}")
    body = response.get_json()
    item = body["results"][0]
    result = item.get("results", {})
    if item.get("status") != "ok" or result.get("yolo_model") != SHADOW_FUSION_MODEL_NAME:
        raise RuntimeError(f"Shadow Web response contract failed: {result}")
    if "yolo" not in item.get("images", {}) or "yolo_error" in result:
        raise RuntimeError(f"Shadow Web did not return a rendered YOLO result: {result}")

    if production_path.read_bytes() != production_before:
        raise RuntimeError("Production deployment config changed during shadow smoke")
    rollback = shadow["rollback"]
    rollback_ready = bool(
        rollback["checkpoint"] == production["checkpoint"]
        and rollback["sha256"] == production["sha256"]
        and sha256(ROOT / rollback["checkpoint"]) == rollback["sha256"]
    )
    warm_totals = [float(row["total_ms"]) for row in direct_runs[1:]]
    latency = {
        "cold_process_model_loading_excluded": True,
        "direct_runs_ms": [round(float(row["total_ms"]), 3) for row in direct_runs],
        "warm_median_ms": round(statistics.median(warm_totals), 3),
        "warm_max_ms": round(max(warm_totals), 3),
        "web_end_to_end_ms": round(web_ms, 3),
        "gate": {"warm_max_ms_max": 3000.0, "web_end_to_end_ms_max": 10000.0},
    }
    latency["passed"] = bool(
        latency["warm_max_ms"] <= latency["gate"]["warm_max_ms_max"]
        and latency["web_end_to_end_ms"] <= latency["gate"]["web_end_to_end_ms_max"]
    )
    summary = {
        "version": "v27_shadow_deployment_smoke_v1",
        "created_at": now_iso(),
        "state": "passed" if latency["passed"] and rollback_ready else "failed",
        "model_name": SHADOW_FUSION_MODEL_NAME,
        "sample_image": image_path.name,
        "sample_shape": list(image_5ch.shape),
        "direct_runs": direct_runs,
        "all_outputs_require_yolo11": True,
        "web": {
            "status_code": response.status_code,
            "model": result.get("yolo_model"),
            "detections": result.get("yolo_detections"),
            "rendered_image_returned": "yolo" in item.get("images", {}),
        },
        "latency": latency,
        "shadow_status": shadow_status(),
        "production_unchanged": True,
        "rollback_ready": rollback_ready,
        "release_blind_holdout_used": False,
        "production_switch_allowed": False,
    }
    write_json_atomic(SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
