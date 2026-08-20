"""Run the preregistered full-resolution boundary-preserving V25 matrix."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from calibrate_yolo_ensemble import evaluate_config_matrix_full_resolution
from run_data_v9_grouped_cv import now_iso, sha256
from run_data_v23_yolo11_fusion_valid import AUXILIARY_EXPORT, DATASET, PRIMARY, PRIMARY_EXPORT, gate
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
V24_SUMMARY = ROOT / "logs" / "v24_yolo11_hybrid_fusion_valid.summary.json"
STATUS = ROOT / "logs" / "v25_yolo11_fullres_fusion_matrix.status.json"
SUMMARY = ROOT / "logs" / "v25_yolo11_fullres_fusion_matrix.summary.json"
LOCKED = ROOT / "reports" / "Data_v25_yolo11_fullres_fusion_matrix" / "locked_fusion_config.json"


def protocol() -> dict[str, object]:
    return {
        "version": "data_v25_yolo11_fullres_fusion_matrix_protocol_v1",
        "preregistered_at": now_iso(),
        "prerequisite": "V24 detection gates passed but downsample-selected mask_vote failed full-resolution IoU",
        "single_change": "select mask vote directly on full-resolution valid metrics",
        "strict_primary_matrix": {
            "model_weights": [[1.0, 0.5], [1.0, 1.0]], "match_iou": [0.3, 0.5],
            "mask_vote": [0.4, 0.5, 0.6], "unconfirmed_score_threshold": [0.3],
            "required_model_indices": [0],
        },
        "equal_fallback_matrix": {
            "model_weights": [[1.0, 1.0]], "match_iou": [0.3, 0.5],
            "mask_vote": [0.4, 0.5, 0.6], "unconfirmed_score_threshold": [0.4, 0.5],
            "required_model_indices": [],
        },
        "common_score_threshold": 0.2, "min_models": 1,
        "selection_rule": "highest F1 passing strict YOLO11-primary gate; only if none pass, highest passing exact 1:1",
        "gate_unchanged_from_v23_v24": True,
        "selection_split": "valid", "held_out_test_used": False,
        "release_blind_holdout_used": False, "deployment_write_allowed": False,
    }


def configs() -> list[dict]:
    rows = []
    for weight, match_iou, mask_vote in itertools.product((0.5, 1.0), (0.3, 0.5), (0.4, 0.5, 0.6)):
        rows.append({
            "role": "strict_yolo11_primary", "model_weights": [1.0, weight],
            "match_iou": match_iou, "mask_vote": mask_vote, "score_threshold": 0.2,
            "min_models": 1, "required_model_indices": [0], "unconfirmed_score_threshold": 0.3,
        })
    for unconfirmed, match_iou, mask_vote in itertools.product((0.4, 0.5), (0.3, 0.5), (0.4, 0.5, 0.6)):
        rows.append({
            "role": "equal_yolo11_yolov8", "model_weights": [1.0, 1.0],
            "match_iou": match_iou, "mask_vote": mask_vote, "score_threshold": 0.2,
            "min_models": 1, "required_model_indices": [], "unconfirmed_score_threshold": unconfirmed,
        })
    return rows


def main() -> int:
    v24 = json.loads(V24_SUMMARY.read_text(encoding="utf-8"))
    if v24.get("decision") != "reject_v24_continue_yolo11_optimization":
        raise RuntimeError("V25 requires the frozen V24 boundary-IoU failure")
    locked_protocol = protocol()
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        if previous.get("protocol", {}).get("version") == locked_protocol["version"]:
            locked_protocol = previous["protocol"]
    state = {"version": "data_v25_yolo11_fullres_fusion_matrix_status_v1", "state": "running",
             "stage": "full_resolution_matrix", "started_at": now_iso(), "updated_at": now_iso(),
             "protocol": locked_protocol}
    write_json_atomic(STATUS, state)
    try:
        images = sorted((DATASET / "valid" / "images").glob("*.JPG"))
        if len(images) != 60:
            raise RuntimeError("V25 valid image contract drifted")
        matrix = evaluate_config_matrix_full_resolution(
            images, DATASET, (PRIMARY_EXPORT, AUXILIARY_EXPORT),
            ("yolo11_primary", "yolov8_auxiliary"), configs(),
        )
        rows = []
        for row in matrix:
            result = gate({"locked_full_resolution_metrics": row["metrics"]})
            rows.append({**row, "gate": result})
        strict = [row for row in rows if row["config"]["role"] == "strict_yolo11_primary" and row["gate"]["passed"]]
        equal = [row for row in rows if row["config"]["role"] == "equal_yolo11_yolov8" and row["gate"]["passed"]]
        pool = strict or equal
        selected = max(pool, key=lambda row: float(row["metrics"]["fusion"]["f1"]), default=None)
        if selected:
            LOCKED.parent.mkdir(parents=True, exist_ok=True)
            locked = {
                "version": "locked_yolo11_yolov8_mask_fusion_v25",
                "created_at": now_iso(), "calibrated_on": "valid_full_resolution",
                "held_out_test_used": False, "release_blind_holdout_used": False,
                "primary_model": "yolo11_primary", "auxiliary_model": "yolov8_auxiliary",
                **selected["config"], "validation_metrics": selected["metrics"],
            }
            write_json_atomic(LOCKED, locked)
        summary = {
            "version": "data_v25_yolo11_fullres_fusion_matrix_summary_v1", "created_at": now_iso(),
            "protocol": locked_protocol, "matrix_rows": rows, "gate_passed": selected is not None,
            "selected": selected, "locked_fusion_config": str(LOCKED.resolve()) if selected else None,
            "decision": "freeze_v25_for_multiseed_confirmation" if selected else "reject_v25_continue_yolo11_optimization",
            "yolo11_checkpoint": str(PRIMARY.resolve()), "yolo11_sha256": sha256(PRIMARY),
            "held_out_test_used": False, "release_blind_holdout_used": False,
            "deployment_write_allowed": False,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(state="complete", stage="complete", updated_at=now_iso(), decision=summary["decision"])
        write_json_atomic(STATUS, state)
        print(json.dumps({"decision": summary["decision"], "selected": selected}, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        state.update(state="failed", updated_at=now_iso(), error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
