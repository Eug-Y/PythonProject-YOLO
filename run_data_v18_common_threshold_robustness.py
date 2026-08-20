"""Audit the locked V17 confidence threshold across all three V15 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from run_data_v17_seed2_threshold_calibration import fixed_row, load_json
from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "logs" / "v18_common_threshold_robustness.status.json"
SUMMARY = ROOT / "logs" / "v18_common_threshold_robustness.summary.json"
V17_SUMMARY = ROOT / "logs" / "v17_seed2_threshold_calibration.summary.json"
REPORT_ROOT = ROOT / "reports" / "Data_v15_yolo11_standardized_multiseed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def compare_rows(seed: int, control: dict[str, float | int], candidate: dict[str, float | int]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "control": control,
        "candidate": candidate,
    }
    for metric in ("precision", "recall", "f1"):
        delta = float(candidate[metric]) - float(control[metric])
        row[f"{metric}_delta"] = delta
        row[f"{metric}_delta_pp"] = delta * 100.0
    row["tp_delta"] = int(candidate["tp"]) - int(control["tp"])
    row["fp_delta"] = int(candidate["fp"]) - int(control["fp"])
    row["fn_delta"] = int(candidate["fn"]) - int(control["fn"])
    return row


def gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3 or {int(row["seed"]) for row in rows} != {0, 1, 2}:
        raise ValueError("V18 requires exactly Seed 0/1/2")
    metrics = {
        "mean_precision_delta": mean(float(row["precision_delta"]) for row in rows),
        "mean_recall_delta": mean(float(row["recall_delta"]) for row in rows),
        "mean_f1_delta": mean(float(row["f1_delta"]) for row in rows),
        "worst_f1_delta": min(float(row["f1_delta"]) for row in rows),
        "positive_f1_seeds": sum(float(row["f1_delta"]) > 0.0 for row in rows),
    }
    metrics.update(
        mean_precision_delta_pp=metrics["mean_precision_delta"] * 100.0,
        mean_recall_delta_pp=metrics["mean_recall_delta"] * 100.0,
        mean_f1_delta_pp=metrics["mean_f1_delta"] * 100.0,
        worst_f1_delta_pp=metrics["worst_f1_delta"] * 100.0,
    )
    checks = {
        "all_three_seed_f1_deltas_nonnegative": metrics["worst_f1_delta"] >= 0.0,
        "mean_precision_delta_not_below_minus_0_01": metrics["mean_precision_delta"] >= -0.01,
        "mean_recall_delta_not_below_minus_0_01": metrics["mean_recall_delta"] >= -0.01,
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def validate_sweep(payload: dict[str, Any], *, seed: int, role: str) -> None:
    if payload.get("split") != "val" or payload.get("held_out_test_used") is not False:
        raise RuntimeError(f"Seed-{seed} {role} sweep is not valid-only")
    if payload.get("validation_overlap_mask") is not False:
        raise RuntimeError(f"Seed-{seed} {role} sweep did not use overlap_mask=False")


def main() -> int:
    args = parse_args()
    v17 = load_json(V17_SUMMARY)
    if v17.get("result", {}).get("passed") is not True:
        raise RuntimeError("V17 threshold gate did not pass")
    if v17.get("decision") != "lock_valid_operating_threshold_for_review_no_auto_deployment":
        raise RuntimeError("V17 did not lock a review threshold")
    if v17.get("held_out_test_used") is not False or v17.get("deployment_eligible") is not False:
        raise RuntimeError("V17 safety contract drifted")

    threshold = float(v17["protocol"]["locked_confidence"])
    protocol = {
        "version": "v18_common_threshold_robustness_protocol_v1",
        "created_at": now_iso(),
        "locked_confidence": threshold,
        "threshold_source": "V17 control-derived Seed-2 threshold",
        "threshold_search_allowed": False,
        "seeds": [0, 1, 2],
        "reference": "same-seed V15 control at identical fixed confidence",
        "gate_thresholds": {
            "worst_seed_f1_delta_min": 0.0,
            "mean_precision_delta_min": -0.01,
            "mean_recall_delta_min": -0.01,
        },
        "new_training": False,
        "new_inference": False,
        "validation_overlap_mask": False,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_write_allowed": False,
    }
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0

    deployment_hash_before = sha256(DEPLOYMENT)
    if deployment_hash_before != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V18")
    state = {
        "version": "v18_common_threshold_robustness_status_v1",
        "state": "running",
        "stage": "three_seed_fixed_threshold_audit",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol,
        "deployment_sha256_before": deployment_hash_before,
    }
    write_json_atomic(STATUS, state)
    try:
        rows = []
        for seed in (0, 1, 2):
            control_dir = REPORT_ROOT / f"seed{seed}" / "control"
            candidate_dir = REPORT_ROOT / f"seed{seed}" / "candidate"
            control_summary = load_json(control_dir / "confidence_sweep.summary.json")
            candidate_summary = load_json(candidate_dir / "confidence_sweep.summary.json")
            validate_sweep(control_summary, seed=seed, role="control")
            validate_sweep(candidate_summary, seed=seed, role="candidate")
            if control_summary.get("target_instances") != candidate_summary.get("target_instances"):
                raise RuntimeError(f"Seed-{seed} target-instance counts differ")
            control = fixed_row(control_dir / "confidence_sweep.csv", threshold)
            candidate = fixed_row(candidate_dir / "confidence_sweep.csv", threshold)
            rows.append(compare_rows(seed, control, candidate))

        result = gate(rows)
        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash_before:
            raise RuntimeError("Deployment checkpoint changed during V18")
        summary = {
            "version": "v18_common_threshold_robustness_summary_v1",
            "created_at": now_iso(),
            "protocol": protocol,
            "rows": rows,
            "result": result,
            "decision": (
                "retain_v17_threshold_as_robust_review_contract"
                if result["passed"]
                else "withdraw_v17_common_threshold_contract"
            ),
            "scope_note": "Descriptive cross-seed valid audit; no checkpoint, mAP, or deployment state changed.",
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
