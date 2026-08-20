"""Preregister and validate a YOLO11-led YOLOv8-assisted release candidate.

This runner only reads the train/valid feature contract.  It first evaluates
a strict YOLO11-primary fusion, then a 1:1 co-primary fallback.  The old test
and the new release blind holdout are never read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_data_v9_grouped_cv import now_iso, sha256
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "Data_v3"
FEATURE_ROOT = ROOT / "datasets" / "Data_v4_candidates"
FEATURE_MANIFEST = ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json"
DEPLOYMENT_CONFIG = ROOT / "configs" / "deployment.json"
PRIMARY = ROOT / "runs" / "segment" / "land_seg" / "v19_yolo11_c0_nooverlap_b1_seed0_v1" / "weights" / "best.pt"
PRIMARY_EXPECTED_SHA256 = "eea619abdc0ba0e71404942895c15b045d0d63bf6fbc9ad999a4874c35061181"
OUTPUT = ROOT / "reports" / "Data_v23_yolo11_fusion_valid"
PRIMARY_EXPORT = OUTPUT / "predictions_yolo11_primary"
AUXILIARY_EXPORT = OUTPUT / "predictions_yolov8_auxiliary"
PRIMARY_FUSION = OUTPUT / "strict_yolo11_primary"
EQUAL_FUSION = OUTPUT / "equal_yolo11_yolov8"
STATUS = ROOT / "logs" / "v23_yolo11_fusion_valid.status.json"
SUMMARY = ROOT / "logs" / "v23_yolo11_fusion_valid.summary.json"
LOG = ROOT / "logs" / "v23_yolo11_fusion_valid.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def protocol() -> dict[str, object]:
    return {
        "version": "data_v23_yolo11_fusion_valid_protocol_v1",
        "preregistered_at": now_iso(),
        "architecture_requirement": "YOLO11 must be primary; YOLOv8 may assist, or both may be exactly 1:1",
        "primary_model": "YOLO11n-seg 5-channel CA+ECA V19 Seed-0",
        "auxiliary_model": "deployed YOLOv8n-seg 5-channel CA incumbent",
        "candidate_priority": [
            "strict_yolo11_primary_with_yolov8_weight_0.5_or_1.0_and_every_output_requires_yolo11",
            "exact_1_to_1_yolo11_yolov8_co_primary_fallback",
        ],
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "imgsz": 1024,
        "prediction_export_confidence": 0.2,
        "selection_split": "valid",
        "held_out_test_used": False,
        "release_blind_holdout_used": False,
        "gate": {
            "fusion_f1_delta_vs_yolo11_min": 0.0,
            "fusion_f1_delta_vs_yolov8_min": 0.0,
            "precision_delta_vs_better_standalone_min": -0.01,
            "recall_delta_vs_better_standalone_min": -0.01,
            "mean_matched_iou_delta_vs_better_standalone_min": -0.01,
        },
        "deployment_write_allowed": False,
    }


def command_export(checkpoint: Path, output: Path, device: str) -> list[str]:
    return [
        sys.executable, str(ROOT / "export_multichannel_predictions.py"),
        "--checkpoint", str(checkpoint),
        "--feature-root", str(FEATURE_ROOT),
        "--manifest", str(FEATURE_MANIFEST),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--dataset", str(DATASET), "--split", "valid",
        "--output", str(output), "--imgsz", "1024", "--conf", "0.2", "--device", device,
    ]


def command_calibrate(output: Path, *, strict_primary: bool) -> list[str]:
    command = [
        sys.executable, str(ROOT / "calibrate_yolo_ensemble.py"),
        "--prediction", f"yolo11_primary={PRIMARY_EXPORT}",
        "--prediction", f"yolov8_auxiliary={AUXILIARY_EXPORT}",
        "--dataset", str(DATASET), "--output", str(output),
        "--second-model-weights", *(("0.5", "1.0") if strict_primary else ("1.0",)),
        "--calibration-max-side", "320",
    ]
    if strict_primary:
        command.append("--require-primary")
    return command


def export_complete(root: Path, checkpoint: Path) -> bool:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        payload.get("checkpoint") == str(checkpoint.resolve())
        and payload.get("split") == "valid"
        and payload.get("held_out_test_used") is False
        and payload.get("image_count") == 60
        and not payload.get("failures")
        and len(list((root / "predictions").glob("*.npz"))) == 60
    )


def gate(payload: dict) -> dict[str, object]:
    metrics = payload["locked_full_resolution_metrics"]
    fusion = metrics["fusion"]
    primary = metrics["primary_yolo11"]
    auxiliary = metrics["auxiliary_yolov8"]
    better_precision = max(float(primary["precision"]), float(auxiliary["precision"]))
    better_recall = max(float(primary["recall"]), float(auxiliary["recall"]))
    better_iou = max(float(primary["mean_matched_iou"]), float(auxiliary["mean_matched_iou"]))
    values = {
        "fusion_f1_delta_vs_yolo11": float(fusion["f1"]) - float(primary["f1"]),
        "fusion_f1_delta_vs_yolov8": float(fusion["f1"]) - float(auxiliary["f1"]),
        "precision_delta_vs_better_standalone": float(fusion["precision"]) - better_precision,
        "recall_delta_vs_better_standalone": float(fusion["recall"]) - better_recall,
        "mean_matched_iou_delta_vs_better_standalone": float(fusion["mean_matched_iou"]) - better_iou,
    }
    checks = {
        "f1_not_below_yolo11": values["fusion_f1_delta_vs_yolo11"] >= 0.0,
        "f1_not_below_yolov8": values["fusion_f1_delta_vs_yolov8"] >= 0.0,
        "precision_regression_within_0_01": values["precision_delta_vs_better_standalone"] >= -0.01,
        "recall_regression_within_0_01": values["recall_delta_vs_better_standalone"] >= -0.01,
        "matched_iou_regression_within_0_01": values["mean_matched_iou_delta_vs_better_standalone"] >= -0.01,
    }
    return {"values": values, "checks": checks, "passed": all(checks.values())}


def main() -> int:
    args = parse_args()
    deployment = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    auxiliary = ROOT / deployment["checkpoint"]
    for path in (DATASET, FEATURE_ROOT, FEATURE_MANIFEST, PRIMARY, auxiliary):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(PRIMARY) != PRIMARY_EXPECTED_SHA256:
        raise RuntimeError("Frozen YOLO11 primary checkpoint hash drifted")
    if sha256(auxiliary) != deployment.get("sha256"):
        raise RuntimeError("Deployed YOLOv8 auxiliary checkpoint hash drifted")
    deployment_hash_before = sha256(auxiliary)
    locked_protocol = protocol()
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        previous_protocol = previous.get("protocol", {})
        if previous_protocol.get("version") == locked_protocol["version"]:
            locked_protocol = previous_protocol
    if args.dry_run:
        print(json.dumps(locked_protocol, ensure_ascii=False, indent=2))
        return 0
    state = {
        "version": "data_v23_yolo11_fusion_valid_status_v1",
        "state": "running", "stage": "preregistered", "started_at": now_iso(),
        "updated_at": now_iso(), "protocol": locked_protocol,
        "deployment_sha256_before": deployment_hash_before,
    }
    write_json_atomic(STATUS, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    steps: list[tuple[str, list[str]]] = []
    if not export_complete(PRIMARY_EXPORT, PRIMARY):
        steps.append(("export_yolo11_primary_valid", command_export(PRIMARY, PRIMARY_EXPORT, args.device)))
    if not export_complete(AUXILIARY_EXPORT, auxiliary):
        steps.append(("export_yolov8_auxiliary_valid", command_export(auxiliary, AUXILIARY_EXPORT, args.device)))
    steps.extend([
        ("calibrate_strict_yolo11_primary", command_calibrate(PRIMARY_FUSION, strict_primary=True)),
        ("calibrate_equal_fallback", command_calibrate(EQUAL_FUSION, strict_primary=False)),
    ])
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            for stage, command in steps:
                state.update(stage=stage, updated_at=now_iso())
                write_json_atomic(STATUS, state)
                log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"{stage} failed with exit code {completed.returncode}; see {LOG}")
        strict_payload = json.loads((PRIMARY_FUSION / "summary.json").read_text(encoding="utf-8"))
        equal_payload = json.loads((EQUAL_FUSION / "summary.json").read_text(encoding="utf-8"))
        strict_gate, equal_gate = gate(strict_payload), gate(equal_payload)
        if strict_gate["passed"]:
            selected_role, selected_root = "strict_yolo11_primary", PRIMARY_FUSION
        elif equal_gate["passed"]:
            selected_role, selected_root = "equal_yolo11_yolov8", EQUAL_FUSION
        else:
            selected_role, selected_root = None, None
        if sha256(auxiliary) != deployment_hash_before:
            raise RuntimeError("Deployment checkpoint changed during V23 valid evaluation")
        summary = {
            "version": "data_v23_yolo11_fusion_valid_summary_v1",
            "created_at": now_iso(), "protocol": locked_protocol,
            "models": {
                "yolo11_primary": {"checkpoint": str(PRIMARY.resolve()), "sha256": sha256(PRIMARY)},
                "yolov8_auxiliary": {"checkpoint": str(auxiliary.resolve()), "sha256": sha256(auxiliary)},
            },
            "strict_yolo11_primary": {"gate": strict_gate, "summary": str((PRIMARY_FUSION / 'summary.json').resolve())},
            "equal_yolo11_yolov8": {"gate": equal_gate, "summary": str((EQUAL_FUSION / 'summary.json').resolve())},
            "gate_passed": selected_root is not None,
            "selected_role": selected_role,
            "locked_fusion_config": str((selected_root / "locked_fusion_config.json").resolve()) if selected_root else None,
            "decision": "continue_multiseed_and_latency_confirmation" if selected_root else "reject_fusion_retain_current_deployment",
            "held_out_test_used": False, "release_blind_holdout_used": False,
            "deployment_write_allowed": False, "deployment_unchanged": True,
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
