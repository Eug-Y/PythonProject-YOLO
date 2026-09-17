"""Run V29 recall-priority fine-tuning and the locked V27 valid gate.

V29 starts from the frozen V19 YOLO11 checkpoint and uses only conservative
train-split false-negative sampling weights.  YOLOv8 and the deployed V27
fusion configuration remain frozen.  The runner never reads test or sealed
blind data and never writes the deployment configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from calibrate_yolo_ensemble import evaluate_locked_full_resolution
from run_data_v23_yolo11_fusion_valid import AUXILIARY_EXPORT, DATASET, FEATURE_MANIFEST, FEATURE_ROOT
from run_data_v9_grouped_cv import now_iso, sha256
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
SOURCE_DATASET = ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed"
DATA = ROOT / "configs" / "data_v8_fusion_selection_5ch.yaml"
GROUPS = ROOT / "configs" / "data_v8_yolo11_early_fusion_groups.json"
WEIGHTS = ROOT / "configs" / "generated" / "data_v29_train_recall_weights.json"
V19_CHECKPOINT = PROJECT / "v19_yolo11_c0_nooverlap_b1_seed0_v1" / "weights" / "best.pt"
V19_SHA256 = "eea619abdc0ba0e71404942895c15b045d0d63bf6fbc9ad999a4874c35061181"
V28_SUMMARY = ROOT / "logs" / "v28_yolo11_hard_examples.summary.json"
BASELINE_FREEZE = ROOT / "logs" / "v28_baseline_freeze.summary.json"
V27_SUMMARY = ROOT / "logs" / "v27_yolo11_fusion_recall_recovery.summary.json"
V27_FUSION = ROOT / "reports" / "Data_v27_yolo11_fusion_recall_recovery" / "locked_fusion_config.json"
DEPLOYMENT = ROOT / "configs" / "deployment.json"
ROLLBACK = ROOT / "configs" / "deployment.pre_v27.json"
RUN_NAME = "v29_yolo11_recall_finetune_seed0_v1"
RUN_DIR = PROJECT / RUN_NAME
CHECKPOINT = RUN_DIR / "weights" / "best.pt"
SMOKE = PROJECT / "v29_yolo11_recall_finetune_smoke_seed0_v1" / "weights" / "best.pt"
OUTPUT = ROOT / "reports" / "Data_v29_yolo11_recall_finetune"
CANDIDATE_EXPORT = OUTPUT / "candidate_yolo11_valid_predictions"
STATUS = ROOT / "logs" / "v29_yolo11_recall_finetune.status.json"
SUMMARY = ROOT / "logs" / "v29_yolo11_recall_finetune.summary.json"
LOG = ROOT / "logs" / "v29_yolo11_recall_finetune.log"
CHANNELS = ["R", "G", "B", "NDVI", "NDRE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def protocol() -> dict[str, object]:
    return {
        "version": "data_v29_yolo11_recall_finetune_protocol_v1",
        "preregistered_at": now_iso(),
        "objective": "recover recall without weakening the existing V27 production gates",
        "initialization": "frozen V19 Seed-0 YOLO11 checkpoint; optimizer and epoch state reset",
        "initialization_checkpoint_sha256": V19_SHA256,
        "training_change": "train-only false-negative weighted sampling; no negative examples added",
        "primary_model": "YOLO11n-seg + 3x CoordAtt + 3x ECA",
        "auxiliary_model": "frozen YOLOv8; training disabled",
        "channels": CHANNELS,
        "epochs": 12,
        "patience": 0,
        "imgsz": 1024,
        "batch": 1,
        "seed": 0,
        "optimizer": "SGD",
        "lr0": 0.0002,
        "lrf": 0.1,
        "warmup_epochs": 0.5,
        "augmentation": {"mosaic": 0.0, "mixup": 0.0, "close_mosaic": 0},
        "training_overlap_mask": False,
        "fusion_config_search_allowed": False,
        "fusion_weights": [1.0, 1.0],
        "every_output_requires_yolo11": True,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "sealed_blind_used": False,
        "seed0_gate_relative_to_v27": {
            "f1_delta_min": 0.015,
            "precision_delta_min": 0.0,
            "recall_delta_min": 0.01,
            "mean_matched_iou_delta_min": -0.005,
        },
        "deployment_write_allowed": False,
    }


def training_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", "c0_current_indices_early",
        "--groups-config", str(GROUPS),
        "--manifest", str(FEATURE_MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(DATA),
        "--base-model", "yolo11n-seg.pt",
        "--attention", "ca_eca",
        "--extra-init", "mean",
        "--init-checkpoint", str(V19_CHECKPOINT),
        "--sampling-weights", str(WEIGHTS),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--warmup-epochs", "0.5",
        "--lr0", "0.0002",
        "--lrf", "0.1",
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--device", args.device,
        "--seed", "0",
        "--mask-ratio", "4",
        "--no-overlap-mask",
        "--no-cache",
        "--deterministic",
        "--mosaic", "0.0",
        "--mixup", "0.0",
        "--close-mosaic", "0",
        "--name", RUN_NAME,
    ]


def export_command(device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "export_multichannel_predictions.py"),
        "--checkpoint", str(CHECKPOINT),
        "--feature-root", str(FEATURE_ROOT),
        "--manifest", str(FEATURE_MANIFEST),
        "--channels", *CHANNELS,
        "--dataset", str(DATASET),
        "--split", "valid",
        "--output", str(CANDIDATE_EXPORT),
        "--imgsz", "1024",
        "--conf", "0.2",
        "--iou", "0.45",
        "--device", device,
    ]


def export_complete() -> bool:
    path = CANDIDATE_EXPORT / "manifest.json"
    if not path.is_file() or not CHECKPOINT.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        payload.get("checkpoint") == str(CHECKPOINT.resolve())
        and payload.get("split") == "valid"
        and payload.get("held_out_test_used") is False
        and payload.get("channels") == CHANNELS
        and payload.get("image_count") == 60
        and payload.get("conf") == 0.2
        and not payload.get("failures")
        and len(list((CANDIDATE_EXPORT / "predictions").glob("*.npz"))) == 60
    )


def validate_inputs(*, require_smoke: bool = True) -> tuple[dict[str, object], dict[str, object]]:
    required_paths = [
        SOURCE_DATASET, DATA, GROUPS, FEATURE_ROOT, FEATURE_MANIFEST, WEIGHTS, V19_CHECKPOINT,
        V28_SUMMARY, BASELINE_FREEZE, V27_SUMMARY, V27_FUSION, DEPLOYMENT, ROLLBACK,
        AUXILIARY_EXPORT,
    ]
    if require_smoke:
        required_paths.append(SMOKE)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(V19_CHECKPOINT) != V19_SHA256:
        raise RuntimeError("Frozen V19 initialization checkpoint drifted")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text or "data_release_blind_v1" in data_text or "channels: 5" not in data_text:
        raise RuntimeError("V29 training YAML exposed forbidden data or changed channels")
    weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    values = list(weights.get("weights_by_image", {}).values())
    if (
        weights.get("version") != "data_v29_train_recall_weights_v1"
        or weights.get("primary_model") != "YOLO11"
        or weights.get("source_split") != "train"
        or weights.get("validation_split_used") is not False
        or weights.get("held_out_split_used") is not False
        or weights.get("sealed_blind_used") is not False
        or weights.get("selection", {}).get("selected_count") != 24
        or weights.get("selection", {}).get("selected_groups") != 24
        or weights.get("selection", {}).get("negative_examples_added") != 0
        or len(values) != 210
        or min(values) != 1.0
        or max(values) != 1.4
    ):
        raise RuntimeError("V29 train-only recall weights contract drifted")
    v28 = json.loads(V28_SUMMARY.read_text(encoding="utf-8"))
    if v28.get("decision") != "reject_v28_retain_v27_production" or v28.get("held_out_test_used") is not False:
        raise RuntimeError("V29 requires the rejected V28 valid-only result")
    freeze = json.loads(BASELINE_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("state") != "frozen" or freeze.get("production_model") != "v27_yolo11_primary_fusion_v1":
        raise RuntimeError("V27 baseline is not frozen")
    for key, path in (("deployment_config", DEPLOYMENT), ("rollback_config", ROLLBACK), ("fusion", V27_FUSION)):
        if sha256(path) != freeze["artifacts"][key]["sha256"]:
            raise RuntimeError(f"Frozen V27 artifact changed before V29: {key}")
    locked = json.loads(V27_FUSION.read_text(encoding="utf-8"))
    if (
        locked.get("model_weights") != [1.0, 1.0]
        or locked.get("required_model_indices") != [0]
        or locked.get("held_out_test_used") is not False
        or locked.get("release_blind_holdout_used") is not False
    ):
        raise RuntimeError("Locked V27 fusion contract drifted")
    return freeze, locked


def run_logged(command: list[str], state: dict[str, object]) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        state["current_process_pid"] = process.pid
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS, state)
        return_code = process.wait()
    state["current_process_pid"] = None
    state["updated_at"] = now_iso()
    write_json_atomic(STATUS, state)
    if return_code:
        raise RuntimeError(f"Command failed ({return_code}); see {LOG}")


def v27_seed0_metrics() -> dict[str, float | int]:
    payload = json.loads(V27_SUMMARY.read_text(encoding="utf-8"))
    rows = payload.get("selected", {}).get("seed_rows", [])
    seed0 = next((row for row in rows if row.get("seed") == 0), None)
    if payload.get("decision") != "freeze_v27_for_shadow_deployment_and_blind_test" or seed0 is None:
        raise RuntimeError("Frozen V27 Seed-0 metrics are missing")
    return seed0["metrics"]["fusion"]


def recall_gate(candidate: dict[str, float | int], baseline: dict[str, float | int]) -> dict[str, object]:
    deltas = {
        "f1": float(candidate["f1"]) - float(baseline["f1"]),
        "precision": float(candidate["precision"]) - float(baseline["precision"]),
        "recall": float(candidate["recall"]) - float(baseline["recall"]),
        "mean_matched_iou": float(candidate["mean_matched_iou"]) - float(baseline["mean_matched_iou"]),
    }
    tolerance = 1e-12
    checks = {
        "f1_delta_at_least_0_015": deltas["f1"] >= 0.015 - tolerance,
        "precision_not_lower": deltas["precision"] >= -tolerance,
        "recall_delta_at_least_0_01": deltas["recall"] >= 0.01 - tolerance,
        "mean_matched_iou_delta_at_least_minus_0_005": deltas["mean_matched_iou"] >= -0.005 - tolerance,
        "every_output_requires_yolo11": True,
    }
    return {"deltas": deltas, "checks": checks, "passed": all(checks.values())}


def main() -> int:
    args = parse_args()
    if (args.epochs, args.patience, args.imgsz, args.batch) != (12, 0, 1024, 1):
        raise ValueError("V29 formal protocol is locked to 12 epochs, patience=0, imgsz=1024, batch=1")
    freeze, locked = validate_inputs(require_smoke=not args.dry_run)
    if args.dry_run:
        print(json.dumps({"protocol": protocol(), "training_command": training_command(args)}, ensure_ascii=False, indent=2))
        return 0
    state: dict[str, object] = {
        "version": "data_v29_yolo11_recall_finetune_status_v1",
        "state": "running",
        "stage": "preregistered",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol(),
        "current_process_pid": None,
        "production_deployment_sha256_before": freeze["artifacts"]["deployment_config"]["sha256"],
        "production_write_allowed": False,
    }
    write_json_atomic(STATUS, state)
    try:
        progress = read_progress(RUN_DIR / "results.csv")
        if not (progress and int(progress["epochs_recorded"]) >= 12 and CHECKPOINT.is_file()):
            if RUN_DIR.exists():
                raise RuntimeError(f"Refusing an incomplete V29 formal run: {RUN_DIR}")
            state["stage"] = "recall_finetune_seed0"
            write_json_atomic(STATUS, state)
            run_logged(training_command(args), state)
        progress = read_progress(RUN_DIR / "results.csv")
        if not progress or int(progress["epochs_recorded"]) < 12 or not CHECKPOINT.is_file():
            raise RuntimeError("V29 formal fine-tuning did not complete")
        state["training_progress"] = progress
        if sha256(DEPLOYMENT) != freeze["artifacts"]["deployment_config"]["sha256"]:
            raise RuntimeError("Production deployment changed during V29 training")
        if not export_complete():
            state["stage"] = "export_locked_valid_predictions"
            write_json_atomic(STATUS, state)
            run_logged(export_command(args.device), state)
        if not export_complete():
            raise RuntimeError("V29 valid prediction export is incomplete")
        state["stage"] = "locked_v27_fusion_recall_gate"
        write_json_atomic(STATUS, state)
        images = sorted((DATASET / "valid" / "images").glob("*.JPG"))
        if len(images) != 60:
            raise RuntimeError(f"Expected 60 valid images, got {len(images)}")
        metrics = evaluate_locked_full_resolution(
            images,
            DATASET,
            (CANDIDATE_EXPORT, AUXILIARY_EXPORT),
            ("v29_yolo11_primary", "frozen_yolov8_auxiliary"),
            locked,
        )
        baseline = v27_seed0_metrics()
        gate = recall_gate(metrics["fusion"], baseline)
        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != freeze["artifacts"]["deployment_config"]["sha256"]:
            raise RuntimeError("Production deployment changed during V29 gate")
        summary = {
            "version": "data_v29_yolo11_recall_finetune_summary_v1",
            "created_at": now_iso(),
            "protocol": state["protocol"],
            "candidate_checkpoint": str(CHECKPOINT.resolve()),
            "candidate_checkpoint_sha256": sha256(CHECKPOINT),
            "training_progress": progress,
            "sampling_weights": str(WEIGHTS.resolve()),
            "sampling_weights_sha256": sha256(WEIGHTS),
            "locked_fusion_config": str(V27_FUSION.resolve()),
            "locked_fusion_config_sha256": sha256(V27_FUSION),
            "v27_seed0_baseline_fusion": baseline,
            "v29_locked_valid_metrics": metrics,
            "recall_priority_gate": gate,
            "gate_passed": gate["passed"],
            "decision": "freeze_v29_seed0_for_multiseed_confirmation" if gate["passed"] else "reject_v29_retain_v27_production",
            "held_out_test_used": False,
            "sealed_blind_used": False,
            "deployment_eligible": False,
            "deployment_write_allowed": False,
            "production_deployment_sha256_before": freeze["artifacts"]["deployment_config"]["sha256"],
            "production_deployment_sha256_after": deployment_hash_after,
            "production_deployment_unchanged": True,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(
            state="complete",
            stage="complete",
            completed_at=now_iso(),
            updated_at=now_iso(),
            decision=summary["decision"],
            gate_passed=summary["gate_passed"],
            candidate_checkpoint_sha256=summary["candidate_checkpoint_sha256"],
            production_deployment_sha256_after=deployment_hash_after,
            production_deployment_unchanged=True,
            current_process_pid=None,
        )
        write_json_atomic(STATUS, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        current_hash = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state.update(
            state="failed",
            failed_at=now_iso(),
            updated_at=now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            production_deployment_sha256_after=current_hash,
            production_deployment_unchanged=current_hash == freeze["artifacts"]["deployment_config"]["sha256"],
            current_process_pid=None,
        )
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
