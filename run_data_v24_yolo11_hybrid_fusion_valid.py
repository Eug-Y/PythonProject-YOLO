"""Validate hybrid confidence-aware YOLO11-led fusion on valid only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_data_v9_grouped_cv import now_iso, sha256
from run_data_v23_yolo11_fusion_valid import (
    AUXILIARY_EXPORT,
    DATASET,
    PRIMARY,
    PRIMARY_EXPORT,
    gate,
)
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
V23_SUMMARY = ROOT / "logs" / "v23_yolo11_fusion_valid.summary.json"
OUTPUT = ROOT / "reports" / "Data_v24_yolo11_hybrid_fusion_valid"
STATUS = ROOT / "logs" / "v24_yolo11_hybrid_fusion_valid.status.json"
SUMMARY = ROOT / "logs" / "v24_yolo11_hybrid_fusion_valid.summary.json"
LOG = ROOT / "logs" / "v24_yolo11_hybrid_fusion_valid.log"
THRESHOLDS = (0.3, 0.4, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def protocol() -> dict[str, object]:
    return {
        "version": "data_v24_yolo11_hybrid_fusion_valid_protocol_v1",
        "preregistered_at": now_iso(),
        "prerequisite": "V23 improved F1 and precision but failed the recall guard",
        "architecture_requirement": "YOLO11 primary; YOLOv8 assistance only, with exact 1:1 fallback allowed",
        "single_change": "retain high-confidence unconfirmed outputs while low-confidence outputs require agreement",
        "unconfirmed_score_thresholds": list(THRESHOLDS),
        "matched_score_threshold": 0.2,
        "strict_primary_grid": {
            "yolov8_weights": [0.5, 1.0], "match_ious": [0.3, 0.5],
            "mask_votes": [0.4, 0.5, 0.6], "min_models": [1], "require_yolo11": True,
        },
        "equal_fallback_grid": {
            "yolov8_weights": [1.0], "match_ious": [0.3, 0.5],
            "mask_votes": [0.4, 0.5, 0.6], "min_models": [1], "require_yolo11": False,
        },
        "selection_rule": "highest full-resolution F1 among passing strict-primary configs; only if none pass, use passing 1:1 configs",
        "gate_unchanged_from_v23": True,
        "selection_split": "valid", "held_out_test_used": False,
        "release_blind_holdout_used": False, "deployment_write_allowed": False,
    }


def calibration_command(output: Path, threshold: float, strict: bool) -> list[str]:
    return [
        sys.executable, str(ROOT / "calibrate_yolo_ensemble.py"),
        "--prediction", f"yolo11_primary={PRIMARY_EXPORT}",
        "--prediction", f"yolov8_auxiliary={AUXILIARY_EXPORT}",
        "--dataset", str(DATASET), "--output", str(output),
        "--second-model-weights", *(("0.5", "1.0") if strict else ("1.0",)),
        "--match-ious", "0.3", "0.5", "--mask-votes", "0.4", "0.5", "0.6",
        "--score-thresholds", "0.2", "--min-model-options", "1",
        "--unconfirmed-score-thresholds", str(threshold), "--calibration-max-side", "320",
        *(("--require-primary",) if strict else ()),
    ]


def evaluate_outputs(roots: list[tuple[str, float, Path]]) -> list[dict]:
    rows = []
    for role, threshold, root in roots:
        payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        result = gate(payload)
        rows.append({
            "role": role, "unconfirmed_score_threshold": threshold,
            "summary": str((root / "summary.json").resolve()),
            "locked_fusion_config": str((root / "locked_fusion_config.json").resolve()),
            "full_resolution_metrics": payload["locked_full_resolution_metrics"],
            "gate": result,
        })
    return rows


def main() -> int:
    args = parse_args()
    v23 = json.loads(V23_SUMMARY.read_text(encoding="utf-8"))
    if v23.get("decision") != "reject_fusion_retain_current_deployment":
        raise RuntimeError("V24 requires the locked V23 recall-gate failure")
    for export in (PRIMARY_EXPORT, AUXILIARY_EXPORT):
        manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("split") != "valid" or manifest.get("held_out_test_used") is not False:
            raise RuntimeError("V24 prediction export is not valid-only")
    locked_protocol = protocol()
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        if previous.get("protocol", {}).get("version") == locked_protocol["version"]:
            locked_protocol = previous["protocol"]
    if args.dry_run:
        print(json.dumps(locked_protocol, ensure_ascii=False, indent=2))
        return 0
    state = {"version": "data_v24_yolo11_hybrid_fusion_valid_status_v1", "state": "running",
             "stage": "preregistered", "started_at": now_iso(), "updated_at": now_iso(),
             "protocol": locked_protocol}
    write_json_atomic(STATUS, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    strict_roots = [("strict_yolo11_primary", value, OUTPUT / f"strict_primary_unconfirmed_{value:.1f}") for value in THRESHOLDS]
    equal_roots = [("equal_yolo11_yolov8", value, OUTPUT / f"equal_unconfirmed_{value:.1f}") for value in THRESHOLDS]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            for role, threshold, root in strict_roots:
                state.update(stage=f"{role}_{threshold:.1f}", updated_at=now_iso())
                write_json_atomic(STATUS, state)
                command = calibration_command(root, threshold, strict=True)
                log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"V24 strict threshold {threshold} failed; see {LOG}")
        strict_rows = evaluate_outputs(strict_roots)
        strict_passing = [row for row in strict_rows if row["gate"]["passed"]]
        equal_rows: list[dict] = []
        if not strict_passing:
            with LOG.open("a", encoding="utf-8", buffering=1) as log:
                for role, threshold, root in equal_roots:
                    state.update(stage=f"{role}_{threshold:.1f}", updated_at=now_iso())
                    write_json_atomic(STATUS, state)
                    command = calibration_command(root, threshold, strict=False)
                    log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
                    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    if completed.returncode:
                        raise RuntimeError(f"V24 equal threshold {threshold} failed; see {LOG}")
            equal_rows = evaluate_outputs(equal_roots)
        pool = strict_passing or [row for row in equal_rows if row["gate"]["passed"]]
        selected = max(
            pool,
            key=lambda row: float(row["full_resolution_metrics"]["fusion"]["f1"]),
            default=None,
        )
        summary = {
            "version": "data_v24_yolo11_hybrid_fusion_valid_summary_v1", "created_at": now_iso(),
            "protocol": locked_protocol, "strict_rows": strict_rows, "equal_rows": equal_rows,
            "gate_passed": selected is not None, "selected": selected,
            "decision": "freeze_fusion_config_for_multiseed_confirmation" if selected else "reject_v24_continue_yolo11_optimization",
            "yolo11_checkpoint": str(PRIMARY.resolve()), "yolo11_sha256": sha256(PRIMARY),
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
