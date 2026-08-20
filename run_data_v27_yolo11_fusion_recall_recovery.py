"""Recover multiseed recall without weakening the YOLO11-primary contract.

V27 uses only cached validation predictions.  It varies the confirmation
matching IoU and the confidence threshold for YOLO11-only clusters, then
applies the unchanged V26 three-seed aggregate gate at full resolution.
"""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

from calibrate_yolo_ensemble import evaluate_config_matrix_full_resolution
from run_data_v9_grouped_cv import now_iso, sha256
from run_data_v23_yolo11_fusion_valid import AUXILIARY_EXPORT, DATASET
from run_data_v26_yolo11_fusion_multiseed import CHECKPOINTS, aggregate, seed_export
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
V26_SUMMARY = ROOT / "logs" / "v26_yolo11_fusion_multiseed.summary.json"
OUTPUT = ROOT / "reports" / "Data_v27_yolo11_fusion_recall_recovery"
STATUS = ROOT / "logs" / "v27_yolo11_fusion_recall_recovery.status.json"
SUMMARY = ROOT / "logs" / "v27_yolo11_fusion_recall_recovery.summary.json"
LOCKED = OUTPUT / "locked_fusion_config.json"


def protocol() -> dict[str, object]:
    return {
        "version": "data_v27_yolo11_fusion_recall_recovery_protocol_v1",
        "preregistered_at": now_iso(),
        "prerequisite": "V26 passed all aggregate gates except mean recall guard",
        "single_change": "recover YOLO11-only clusters by lowering their confidence threshold",
        "matrix": {
            "model_weights": [[1.0, 1.0]],
            "match_iou": [0.3, 0.4, 0.5],
            "mask_vote": [0.5],
            "score_threshold": [0.2],
            "min_models": [1],
            "required_model_indices": [0],
            "unconfirmed_score_threshold": [0.2, 0.225, 0.25, 0.275, 0.3],
        },
        "seeds": [0, 1, 2],
        "selection_rule": "highest mean fusion F1 among configs passing the unchanged V26 aggregate gate",
        "architecture_requirement": "every output requires YOLO11; YOLOv8 has equal fusion weight but cannot emit alone",
        "selection_split": "valid_full_resolution",
        "configuration_search_allowed": True,
        "held_out_test_used": False,
        "release_blind_holdout_used": False,
        "deployment_write_allowed": False,
    }


def configs() -> list[dict[str, object]]:
    return [
        {
            "role": "strict_yolo11_primary",
            "model_weights": [1.0, 1.0],
            "match_iou": match_iou,
            "mask_vote": 0.5,
            "score_threshold": 0.2,
            "min_models": 1,
            "required_model_indices": [0],
            "unconfirmed_score_threshold": unconfirmed,
        }
        for match_iou, unconfirmed in itertools.product(
            (0.3, 0.4, 0.5), (0.2, 0.225, 0.25, 0.275, 0.3)
        )
    ]


def row_for_gate(seed: int, metrics: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    fusion = metrics["fusion"]
    primary = metrics["primary_yolo11"]
    auxiliary = metrics["auxiliary_yolov8"]
    return {
        "seed": seed,
        "f1_delta_vs_yolo11": float(fusion["f1"]) - float(primary["f1"]),
        "f1_delta_vs_yolov8": float(fusion["f1"]) - float(auxiliary["f1"]),
        "precision_delta_vs_better": float(fusion["precision"]) - max(
            float(primary["precision"]), float(auxiliary["precision"])
        ),
        "recall_delta_vs_better": float(fusion["recall"]) - max(
            float(primary["recall"]), float(auxiliary["recall"])
        ),
        "iou_delta_vs_better": float(fusion["mean_matched_iou"]) - max(
            float(primary["mean_matched_iou"]), float(auxiliary["mean_matched_iou"])
        ),
    }


def seed_cache(seed: int) -> Path:
    return OUTPUT / f"seed{seed}_matrix.json"


def main() -> int:
    v26 = json.loads(V26_SUMMARY.read_text(encoding="utf-8"))
    checks = v26.get("aggregate_gate", {}).get("checks", {})
    failed_checks = {name for name, passed in checks.items() if not passed}
    if v26.get("decision") != "reject_v26_continue_yolo11_optimization" or failed_checks != {"mean_recall_guard"}:
        raise RuntimeError("V27 requires the isolated V26 mean-recall failure")
    locked_protocol = protocol()
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        if previous.get("protocol", {}).get("version") == locked_protocol["version"]:
            locked_protocol = previous["protocol"]
    state = {
        "version": "data_v27_yolo11_fusion_recall_recovery_status_v1",
        "state": "running",
        "stage": "preregistered",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": locked_protocol,
    }
    write_json_atomic(STATUS, state)
    candidates = configs()
    images = sorted((DATASET / "valid" / "images").glob("*.JPG"))
    if len(images) != 60:
        raise RuntimeError("V27 valid image contract drifted")
    try:
        by_seed: dict[int, list[dict]] = {}
        for seed in (0, 1, 2):
            cache_path = seed_cache(seed)
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    cached.get("version") == "data_v27_seed_matrix_v1"
                    and cached.get("configs") == candidates
                    and cached.get("checkpoint_sha256") == sha256(CHECKPOINTS[seed])
                    and cached.get("auxiliary_manifest_sha256") == sha256(AUXILIARY_EXPORT / "manifest.json")
                ):
                    by_seed[seed] = cached["rows"]
                    continue
            state.update(stage=f"full_resolution_seed{seed}", updated_at=now_iso())
            write_json_atomic(STATUS, state)
            by_seed[seed] = evaluate_config_matrix_full_resolution(
                images,
                DATASET,
                (seed_export(seed), AUXILIARY_EXPORT),
                (f"yolo11_seed{seed}", "yolov8_auxiliary"),
                candidates,
            )
            write_json_atomic(cache_path, {
                "version": "data_v27_seed_matrix_v1",
                "created_at": now_iso(),
                "seed": seed,
                "checkpoint_sha256": sha256(CHECKPOINTS[seed]),
                "auxiliary_manifest_sha256": sha256(AUXILIARY_EXPORT / "manifest.json"),
                "configs": candidates,
                "rows": by_seed[seed],
                "held_out_test_used": False,
                "release_blind_holdout_used": False,
            })
        rows = []
        for index, config in enumerate(candidates):
            seed_rows = []
            fusion_f1s = []
            for seed in (0, 1, 2):
                metrics = by_seed[seed][index]["metrics"]
                seed_rows.append({
                    **row_for_gate(seed, metrics),
                    "checkpoint": str(CHECKPOINTS[seed].resolve()),
                    "checkpoint_sha256": sha256(CHECKPOINTS[seed]),
                    "metrics": metrics,
                })
                fusion_f1s.append(float(metrics["fusion"]["f1"]))
            gate = aggregate(seed_rows)
            rows.append({
                "config": config,
                "seed_rows": seed_rows,
                "aggregate_gate": gate,
                "mean_fusion_f1": statistics.fmean(fusion_f1s),
            })
        passing = [row for row in rows if row["aggregate_gate"]["passed"]]
        selected = max(
            passing,
            key=lambda row: (
                float(row["mean_fusion_f1"]),
                float(row["aggregate_gate"]["metrics"]["mean_recall_delta_vs_better"]),
            ),
            default=None,
        )
        if selected:
            LOCKED.parent.mkdir(parents=True, exist_ok=True)
            locked = {
                "version": "locked_yolo11_yolov8_mask_fusion_v27",
                "created_at": now_iso(),
                "calibrated_on": "valid_full_resolution_three_seed",
                "held_out_test_used": False,
                "release_blind_holdout_used": False,
                "primary_model": "yolo11_primary",
                "auxiliary_model": "yolov8_auxiliary",
                **selected["config"],
                "aggregate_gate": selected["aggregate_gate"],
                "mean_fusion_f1": selected["mean_fusion_f1"],
            }
            write_json_atomic(LOCKED, locked)
        summary = {
            "version": "data_v27_yolo11_fusion_recall_recovery_summary_v1",
            "created_at": now_iso(),
            "protocol": locked_protocol,
            "matrix_rows": rows,
            "gate_passed": selected is not None,
            "selected": selected,
            "locked_fusion_config": str(LOCKED.resolve()) if selected else None,
            "locked_fusion_config_sha256": sha256(LOCKED) if selected else None,
            "decision": "freeze_v27_for_shadow_deployment_and_blind_test" if selected else "reject_v27_continue_yolo11_optimization",
            "held_out_test_used": False,
            "release_blind_holdout_used": False,
            "deployment_write_allowed": False,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(state="complete", stage="complete", updated_at=now_iso(), decision=summary["decision"])
        write_json_atomic(STATUS, state)
        print(json.dumps({
            "decision": summary["decision"],
            "selected": selected,
            "locked_fusion_config_sha256": summary["locked_fusion_config_sha256"],
        }, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        state.update(state="failed", updated_at=now_iso(), error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
