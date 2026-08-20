"""Confirm the frozen V25 YOLO11-primary fusion across three YOLO11 seeds."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from calibrate_yolo_ensemble import evaluate_locked_full_resolution
from run_data_v9_grouped_cv import now_iso, sha256
from run_data_v23_yolo11_fusion_valid import (
    AUXILIARY_EXPORT, DATASET, FEATURE_MANIFEST, FEATURE_ROOT, PRIMARY_EXPORT, command_export, export_complete,
)
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
V25_SUMMARY = ROOT / "logs" / "v25_yolo11_fullres_fusion_matrix.summary.json"
V25_LOCKED = ROOT / "reports" / "Data_v25_yolo11_fullres_fusion_matrix" / "locked_fusion_config.json"
OUTPUT = ROOT / "reports" / "Data_v26_yolo11_fusion_multiseed"
STATUS = ROOT / "logs" / "v26_yolo11_fusion_multiseed.status.json"
SUMMARY = ROOT / "logs" / "v26_yolo11_fusion_multiseed.summary.json"
LOG = ROOT / "logs" / "v26_yolo11_fusion_multiseed.log"
CHECKPOINTS = tuple(
    ROOT / "runs" / "segment" / "land_seg" / f"v19_yolo11_c0_nooverlap_b1_seed{seed}_v1" / "weights" / "best.pt"
    for seed in (0, 1, 2)
)


def protocol() -> dict[str, object]:
    return {
        "version": "data_v26_yolo11_fusion_multiseed_protocol_v1",
        "preregistered_at": now_iso(),
        "frozen_config": str(V25_LOCKED.resolve()),
        "configuration_search_allowed": False,
        "seeds": [0, 1, 2], "selection_split": "valid",
        "aggregate_gate": {
            "mean_f1_delta_vs_yolo11_min": 0.0, "worst_f1_delta_vs_yolo11_min": -0.005,
            "mean_f1_delta_vs_yolov8_min": 0.0, "worst_f1_delta_vs_yolov8_min": -0.005,
            "mean_precision_delta_vs_better_min": -0.01,
            "mean_recall_delta_vs_better_min": -0.01,
            "mean_iou_delta_vs_better_min": -0.01,
            "positive_f1_vs_yolov8_seeds_min": 2,
        },
        "architecture_requirement": "every output requires YOLO11; YOLOv8 is auxiliary",
        "held_out_test_used": False, "release_blind_holdout_used": False,
        "deployment_write_allowed": False,
    }


def seed_export(seed: int) -> Path:
    return PRIMARY_EXPORT if seed == 0 else OUTPUT / f"seed{seed}_yolo11_predictions"


def aggregate(rows: list[dict]) -> dict[str, object]:
    fields = (
        "f1_delta_vs_yolo11", "f1_delta_vs_yolov8", "precision_delta_vs_better",
        "recall_delta_vs_better", "iou_delta_vs_better",
    )
    values = {field: [float(row[field]) for row in rows] for field in fields}
    metrics = {
        **{f"mean_{field}": statistics.fmean(items) for field, items in values.items()},
        **{f"worst_{field}": min(items) for field, items in values.items()},
        "positive_f1_vs_yolov8_seeds": sum(item > 0 for item in values["f1_delta_vs_yolov8"]),
    }
    checks = {
        "mean_f1_vs_yolo11": metrics["mean_f1_delta_vs_yolo11"] >= 0.0,
        "worst_f1_vs_yolo11": metrics["worst_f1_delta_vs_yolo11"] >= -0.005,
        "mean_f1_vs_yolov8": metrics["mean_f1_delta_vs_yolov8"] >= 0.0,
        "worst_f1_vs_yolov8": metrics["worst_f1_delta_vs_yolov8"] >= -0.005,
        "mean_precision_guard": metrics["mean_precision_delta_vs_better"] >= -0.01,
        "mean_recall_guard": metrics["mean_recall_delta_vs_better"] >= -0.01,
        "mean_iou_guard": metrics["mean_iou_delta_vs_better"] >= -0.01,
        "positive_yolov8_seeds": metrics["positive_f1_vs_yolov8_seeds"] >= 2,
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def main() -> int:
    v25 = json.loads(V25_SUMMARY.read_text(encoding="utf-8"))
    locked = json.loads(V25_LOCKED.read_text(encoding="utf-8"))
    if v25.get("decision") != "freeze_v25_for_multiseed_confirmation":
        raise RuntimeError("V26 requires the frozen V25 valid gate")
    if locked.get("required_model_indices") != [0] or locked.get("primary_model") != "yolo11_primary":
        raise RuntimeError("V25 is not a strict YOLO11-primary configuration")
    locked_protocol = protocol()
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        if previous.get("protocol", {}).get("version") == locked_protocol["version"]:
            locked_protocol = previous["protocol"]
    state = {"version": "data_v26_yolo11_fusion_multiseed_status_v1", "state": "running",
             "stage": "preregistered", "started_at": now_iso(), "updated_at": now_iso(),
             "protocol": locked_protocol}
    write_json_atomic(STATUS, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            for seed in (1, 2):
                checkpoint, output = CHECKPOINTS[seed], seed_export(seed)
                if export_complete(output, checkpoint):
                    continue
                state.update(stage=f"export_seed{seed}", updated_at=now_iso())
                write_json_atomic(STATUS, state)
                command = command_export(checkpoint, output, "0")
                log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"V26 Seed-{seed} export failed; see {LOG}")
        images = sorted((DATASET / "valid" / "images").glob("*.JPG"))
        rows = []
        for seed in (0, 1, 2):
            state.update(stage=f"evaluate_seed{seed}", updated_at=now_iso())
            write_json_atomic(STATUS, state)
            metrics = evaluate_locked_full_resolution(
                images, DATASET, (seed_export(seed), AUXILIARY_EXPORT),
                (f"yolo11_seed{seed}", "yolov8_auxiliary"), locked,
            )
            fusion, primary, auxiliary = metrics["fusion"], metrics["primary_yolo11"], metrics["auxiliary_yolov8"]
            rows.append({
                "seed": seed, "checkpoint": str(CHECKPOINTS[seed].resolve()),
                "checkpoint_sha256": sha256(CHECKPOINTS[seed]), "metrics": metrics,
                "f1_delta_vs_yolo11": fusion["f1"] - primary["f1"],
                "f1_delta_vs_yolov8": fusion["f1"] - auxiliary["f1"],
                "precision_delta_vs_better": fusion["precision"] - max(primary["precision"], auxiliary["precision"]),
                "recall_delta_vs_better": fusion["recall"] - max(primary["recall"], auxiliary["recall"]),
                "iou_delta_vs_better": fusion["mean_matched_iou"] - max(primary["mean_matched_iou"], auxiliary["mean_matched_iou"]),
            })
        result = aggregate(rows)
        summary = {
            "version": "data_v26_yolo11_fusion_multiseed_summary_v1", "created_at": now_iso(),
            "protocol": locked_protocol, "locked_fusion_config": str(V25_LOCKED.resolve()),
            "locked_fusion_config_sha256": sha256(V25_LOCKED), "rows": rows,
            "aggregate_gate": result, "gate_passed": result["passed"],
            "frozen_release_candidate": {
                "yolo11_checkpoint": str(CHECKPOINTS[0].resolve()),
                "yolo11_sha256": sha256(CHECKPOINTS[0]),
                "yolov8_role": "auxiliary", "every_output_requires_yolo11": True,
            } if result["passed"] else None,
            "decision": "freeze_v26_for_blind_test_and_shadow_deployment" if result["passed"] else "reject_v26_continue_yolo11_optimization",
            "held_out_test_used": False, "release_blind_holdout_used": False,
            "deployment_write_allowed": False,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(state="complete", stage="complete", updated_at=now_iso(), decision=summary["decision"])
        write_json_atomic(STATUS, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        state.update(state="failed", updated_at=now_iso(), error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
