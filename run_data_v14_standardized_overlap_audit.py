"""Re-evaluate V12 grouped-CV weights under one independent-overlap validation contract."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from run_data_v9_grouped_cv import (
    CONTRACT_PATH,
    CV_ROOT,
    DEPLOYMENT,
    DEPLOYMENT_SHA256,
    FEATURE_ROOT,
    MANIFEST,
    PROJECT,
    experiment_name as control_name,
    now_iso,
    sha256,
)
from run_data_v12_overlap_mask_cv import experiment_name as candidate_name
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v14_standardized_overlap_cv"
STATUS = ROOT / "logs" / "v14_standardized_overlap_audit.status.json"
SUMMARY = ROOT / "logs" / "v14_standardized_overlap_audit.summary.json"
LOG_DIR = ROOT / "logs" / "v14_standardized_overlap_audit"
FOLDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def data_yaml(fold: int) -> Path:
    return CV_ROOT / f"fold{fold}" / "data_5ch.yaml"


def output_dir(fold: int, variant: str) -> Path:
    return OUTPUT / f"fold{fold}" / variant


def checkpoint(fold: int, variant: str) -> Path:
    name = control_name(fold, "h0_refined") if variant == "control" else candidate_name(fold)
    return PROJECT / name / "weights" / "best.pt"


def sweep_command(fold: int, variant: str, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "sweep_multichannel_confidence.py"),
        "--checkpoint", str(checkpoint(fold, variant)),
        "--data", str(data_yaml(fold)),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--output", str(output_dir(fold, variant)),
        "--imgsz", "640",
        "--conf-min", "0.001",
        "--step", "0.005",
        "--device", device,
        "--no-overlap-mask",
    ]


def validate_inputs() -> str:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if contract.get("held_out_test_used") is not False or contract.get("source_splits") != ["train", "valid"]:
        raise RuntimeError("Grouped-CV contract does not preserve the sealed test")
    if manifest.get("held_out_split_used") is not False or manifest.get("selection_splits") != ["train", "valid"]:
        raise RuntimeError("Feature manifest does not preserve the sealed test")
    for fold in FOLDS:
        text = data_yaml(fold).read_text(encoding="utf-8").lower()
        if "test:" in text or "channels: 5" not in text:
            raise RuntimeError(f"Invalid grouped-CV YAML: {data_yaml(fold)}")
        for variant in ("control", "candidate"):
            if not checkpoint(fold, variant).is_file():
                raise FileNotFoundError(checkpoint(fold, variant))
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V14 audit")
    return deployment_hash


def read_result(fold: int, variant: str) -> dict:
    path = output_dir(fold, variant) / "confidence_sweep.summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("held_out_test_used") is not False:
        raise RuntimeError(f"Non-validation result in V14: {path}")
    if payload.get("validation_overlap_mask") is not False:
        raise RuntimeError(f"Validation overlap contract drifted in V14: {path}")
    if payload.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError(f"Channel contract drifted in V14: {path}")
    return payload


def paired_row(fold: int) -> dict[str, float | int | str]:
    control = read_result(fold, "control")
    candidate = read_result(fold, "candidate")
    control_metrics = control["aggregate_metrics"]
    candidate_metrics = candidate["aggregate_metrics"]
    keys = {
        "map": "metrics/mAP50-95(M)",
        "map50": "metrics/mAP50(M)",
        "precision": "metrics/precision(M)",
        "recall": "metrics/recall(M)",
    }
    row: dict[str, float | int | str] = {
        "fold": fold,
        "validation_images": int(json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["fold_contracts"][fold]["validation_images"]),
        "control_checkpoint": str(checkpoint(fold, "control").resolve()),
        "candidate_checkpoint": str(checkpoint(fold, "candidate").resolve()),
    }
    for short, key in keys.items():
        base = float(control_metrics[key])
        trial = float(candidate_metrics[key])
        row[f"control_{short}"] = base
        row[f"candidate_{short}"] = trial
        row[f"{short}_delta"] = trial - base
        row[f"{short}_delta_pp"] = (trial - base) * 100.0
    return row


def gate_decision(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    map_deltas = [float(row["map_delta"]) for row in rows]
    precision_deltas = [float(row["precision_delta"]) for row in rows]
    recall_deltas = [float(row["recall_delta"]) for row in rows]
    checks = {
        "three_folds_complete": len(rows) == 3,
        "mean_map_delta_at_least_0_002": statistics.fmean(map_deltas) >= 0.002,
        "worst_map_delta_not_below_minus_0_005": min(map_deltas) >= -0.005,
        "at_least_two_positive_map_folds": sum(delta > 0 for delta in map_deltas) >= 2,
        "mean_precision_delta_not_below_minus_0_01": statistics.fmean(precision_deltas) >= -0.01,
        "mean_recall_delta_not_below_minus_0_01": statistics.fmean(recall_deltas) >= -0.01,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "mean_map_delta": statistics.fmean(map_deltas),
        "worst_map_delta": min(map_deltas),
        "positive_map_folds": sum(delta > 0 for delta in map_deltas),
        "mean_precision_delta": statistics.fmean(precision_deltas),
        "mean_recall_delta": statistics.fmean(recall_deltas),
    }


def main() -> int:
    args = parse_args()
    deployment_hash = validate_inputs()
    protocol = {
        "version": "v14_standardized_overlap_validation_protocol_v1",
        "preregistered_at": now_iso(),
        "purpose": "correct the V12 control/candidate validation-mask mismatch before any new YOLO11 training",
        "model": "YOLOv8n-seg existing grouped-CV weights",
        "folds": 3,
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "imgsz": 640,
        "validation_overlap_mask": False,
        "inference_confidence_min": 0.001,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "weight_training_reused": True,
        "gate_thresholds": {
            "mean_map_delta_min": 0.002,
            "worst_map_delta_min": -0.005,
            "positive_folds_min": 2,
            "mean_precision_delta_min": -0.01,
            "mean_recall_delta_min": -0.01,
        },
        "v13_gate_override_allowed": False,
        "deployment_write_allowed": False,
    }
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        previous_protocol = previous.get("protocol", {})
        if previous_protocol.get("version") == protocol["version"]:
            protocol = previous_protocol
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        for fold in FOLDS:
            for variant in ("control", "candidate"):
                print(subprocess.list2cmdline(sweep_command(fold, variant, args.device)))
        return 0
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "state": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol,
        "current": None,
        "completed": [],
        "deployment_sha256_before": deployment_hash,
    }
    write_json_atomic(STATUS, state)
    try:
        for fold in FOLDS:
            for variant in ("control", "candidate"):
                result_path = output_dir(fold, variant) / "confidence_sweep.summary.json"
                state["current"] = {"fold": fold, "variant": variant}
                state["updated_at"] = now_iso()
                write_json_atomic(STATUS, state)
                log_path = LOG_DIR / f"fold{fold}_{variant}.log"
                if not result_path.is_file():
                    command = sweep_command(fold, variant, args.device)
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        capture_output=True,
                        check=False,
                    )
                    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
                    if completed.returncode != 0 or not result_path.is_file():
                        raise RuntimeError(f"V14 validation failed for fold={fold}, variant={variant}; see {log_path}")
                read_result(fold, variant)
                state["completed"].append({"fold": fold, "variant": variant})
                state["updated_at"] = now_iso()
                write_json_atomic(STATUS, state)
        rows = [paired_row(fold) for fold in FOLDS]
        gate = gate_decision(rows)
        current_hash = sha256(DEPLOYMENT)
        if current_hash != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V14 audit")
        summary = {
            "version": "v14_standardized_overlap_audit_summary_v1",
            "created_at": now_iso(),
            "protocol": protocol,
            "rows": rows,
            "gate": gate,
            "decision": "eligible_for_fresh_yolo11_seed0" if gate["passed"] else "retain_current_overlap_training",
            "v12_historical_gate_reinterpreted": "validation_semantics_mismatch_detected",
            "v13_historical_gate_status": "failed_and_not_overridden",
            "deployment_write_allowed": False,
            "deployment_sha256_expected": DEPLOYMENT_SHA256,
            "deployment_sha256_current": current_hash,
            "deployment_unchanged": current_hash == DEPLOYMENT_SHA256,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(state="complete", current=None, updated_at=now_iso(), gate=gate, summary=str(SUMMARY.resolve()))
        write_json_atomic(STATUS, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        state.update(state="failed", updated_at=now_iso(), error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
