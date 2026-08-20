"""Lock and audit the V15 Seed-2 operating threshold without new inference.

V17 deliberately derives the threshold from the Seed-2 control curve and then
applies that exact threshold to the already-frozen V15 Seed-2 candidate.  It
uses valid data only, does not retrain or modify a checkpoint, and never makes
the candidate deployment-eligible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "logs" / "v17_seed2_threshold_calibration.status.json"
SUMMARY = ROOT / "logs" / "v17_seed2_threshold_calibration.summary.json"
V15_SUMMARY = ROOT / "logs" / "v15_yolo11_standardized_multiseed.summary.json"
CONTROL_SUMMARY = (
    ROOT
    / "reports"
    / "Data_v15_yolo11_standardized_multiseed"
    / "seed2"
    / "control"
    / "confidence_sweep.summary.json"
)
CANDIDATE_SUMMARY = (
    ROOT
    / "reports"
    / "Data_v15_yolo11_standardized_multiseed"
    / "seed2"
    / "candidate"
    / "confidence_sweep.summary.json"
)
CANDIDATE_CURVE = CANDIDATE_SUMMARY.with_name("confidence_sweep.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_row(path: Path, threshold: float, *, tolerance: float = 1e-12) -> dict[str, float | int]:
    """Return the unique sweep row at an already-locked confidence threshold."""

    # PowerShell/exported historical curves may carry a UTF-8 BOM.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if abs(float(row["confidence"]) - threshold) <= tolerance]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row at confidence={threshold}, found {len(matches)} in {path}")
    row = matches[0]
    return {
        "confidence": float(row["confidence"]),
        "tp": int(row["tp"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
    }


def gate(
    control_ultralytics: dict[str, float],
    control_fixed: dict[str, float | int],
    candidate_fixed: dict[str, float | int],
) -> dict[str, Any]:
    """Apply the preregistered V17 precision, recall, and F1 protections."""

    values = {
        "locked_confidence": float(candidate_fixed["confidence"]),
        "control_ultralytics_precision": float(control_ultralytics["precision"]),
        "control_ultralytics_recall": float(control_ultralytics["recall"]),
        "control_ultralytics_f1": float(control_ultralytics["f1"]),
        "control_fixed_precision": float(control_fixed["precision"]),
        "control_fixed_recall": float(control_fixed["recall"]),
        "control_fixed_f1": float(control_fixed["f1"]),
        "candidate_fixed_precision": float(candidate_fixed["precision"]),
        "candidate_fixed_recall": float(candidate_fixed["recall"]),
        "candidate_fixed_f1": float(candidate_fixed["f1"]),
    }
    values.update(
        precision_delta_vs_control_ultralytics=values["candidate_fixed_precision"]
        - values["control_ultralytics_precision"],
        recall_delta_vs_control_ultralytics=values["candidate_fixed_recall"]
        - values["control_ultralytics_recall"],
        f1_delta_vs_control_fixed=values["candidate_fixed_f1"] - values["control_fixed_f1"],
        precision_delta_vs_control_fixed=values["candidate_fixed_precision"]
        - values["control_fixed_precision"],
        recall_delta_vs_control_fixed=values["candidate_fixed_recall"] - values["control_fixed_recall"],
    )
    checks = {
        "candidate_precision_at_least_control_ultralytics": values["precision_delta_vs_control_ultralytics"] >= 0.0,
        "candidate_recall_at_least_control_ultralytics": values["recall_delta_vs_control_ultralytics"] >= 0.0,
        "candidate_f1_at_least_control_fixed_best": values["f1_delta_vs_control_fixed"] >= 0.0,
    }
    return {"metrics": values, "checks": checks, "passed": all(checks.values())}


def validate_contract(v15: dict[str, Any], control: dict[str, Any], candidate: dict[str, Any]) -> None:
    if v15.get("gate_passed") is not True or v15.get("decision") != "freeze_candidate_for_review_no_auto_deployment":
        raise RuntimeError("V15 did not authorize Seed-2 candidate review")
    if v15.get("held_out_test_used") is not False or v15.get("validation_overlap_mask") is not False:
        raise RuntimeError("V15 valid-only standardized-overlap contract drifted")
    for name, payload in (("control", control), ("candidate", candidate)):
        if payload.get("split") != "val" or payload.get("held_out_test_used") is not False:
            raise RuntimeError(f"V17 {name} input is not valid-only")
        if payload.get("validation_overlap_mask") is not False:
            raise RuntimeError(f"V17 {name} input did not use validation_overlap_mask=False")
    if control.get("target_instances") != candidate.get("target_instances"):
        raise RuntimeError("Control and candidate target-instance counts differ")


def main() -> int:
    args = parse_args()
    v15 = load_json(V15_SUMMARY)
    control = load_json(CONTROL_SUMMARY)
    candidate = load_json(CANDIDATE_SUMMARY)
    validate_contract(v15, control, candidate)

    deployment_hash_before = sha256(DEPLOYMENT)
    if deployment_hash_before != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V17")

    # This is intentionally control-derived.  The candidate curve is not
    # searched for another threshold after this value is locked.
    control_fixed = control["best_fixed_iou50_operating_point"]
    locked_threshold = float(control_fixed["confidence"])
    protocol = {
        "version": "v17_seed2_control_derived_threshold_protocol_v1",
        "locked_at": now_iso(),
        "candidate": "V15 YOLO11 Seed-2 frozen review checkpoint",
        "threshold_source": "Seed-2 control best fixed-IoU50 F1 operating point",
        "locked_confidence": locked_threshold,
        "candidate_threshold_search_allowed": False,
        "new_training": False,
        "new_inference": False,
        "validation_overlap_mask": False,
        "gate_thresholds": {
            "candidate_precision_min": float(control["ultralytics_operating_point"]["precision"]),
            "candidate_recall_min": float(control["ultralytics_operating_point"]["recall"]),
            "candidate_f1_min": float(control_fixed["f1"]),
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_write_allowed": False,
    }
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0

    state = {
        "version": "v17_seed2_threshold_calibration_status_v1",
        "state": "running",
        "stage": "fixed_threshold_audit",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol,
        "deployment_sha256_before": deployment_hash_before,
    }
    write_json_atomic(STATUS, state)
    try:
        candidate_fixed = fixed_row(CANDIDATE_CURVE, locked_threshold)
        result = gate(control["ultralytics_operating_point"], control_fixed, candidate_fixed)
        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash_before:
            raise RuntimeError("Deployment checkpoint changed during V17")

        checkpoint = Path(v15["candidate_checkpoint"])
        summary = {
            "version": "v17_seed2_threshold_calibration_summary_v1",
            "created_at": now_iso(),
            "protocol": protocol,
            "candidate_checkpoint": str(checkpoint.resolve()),
            "candidate_checkpoint_sha256": sha256(checkpoint),
            "control_fixed_operating_point": control_fixed,
            "candidate_fixed_operating_point": candidate_fixed,
            "candidate_ultralytics_operating_point": candidate["ultralytics_operating_point"],
            "result": result,
            "decision": (
                "lock_valid_operating_threshold_for_review_no_auto_deployment"
                if result["passed"]
                else "do_not_lock_threshold_retain_v15_seed2_for_review"
            ),
            "scope_note": "Valid-only operating-threshold contract; mAP and checkpoint weights are unchanged.",
            "selection_splits": ["train", "valid"],
            "held_out_test_used": False,
            "deployment_eligible": False,
            "deployment_write_allowed": False,
            "deployment_sha256_expected": DEPLOYMENT_SHA256,
            "deployment_sha256_current": deployment_hash_after,
            "deployment_unchanged": True,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(
            state="complete",
            stage="complete",
            updated_at=now_iso(),
            gate_passed=result["passed"],
            decision=summary["decision"],
            summary=str(SUMMARY.resolve()),
        )
        write_json_atomic(STATUS, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        state.update(state="failed", updated_at=now_iso(), error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
