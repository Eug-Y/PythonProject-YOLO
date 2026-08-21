"""Train and gate the V28 YOLO11 hard-example candidate on locked valid data.

The runner keeps the deployed V27 artifacts immutable.  It trains only the
YOLO11 primary, exports valid predictions, applies the already-locked V27 1:1
YOLO11-primary fusion configuration, and compares it with the frozen V27
Seed-0 production baseline.  Test and sealed blind data are never read.
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
DATASET_ROOT = ROOT / "datasets" / "Data_v28_yolo11_hard_examples_train"
DATASET_MANIFEST = DATASET_ROOT / "manifest.json"
DATA = ROOT / "configs" / "generated" / "data_v28_yolo11_hard_examples_5ch.yaml"
PLAN = ROOT / "configs" / "v28_optimization_plan.json"
BASELINE_FREEZE = ROOT / "logs" / "v28_baseline_freeze.summary.json"
V27_SUMMARY = ROOT / "logs" / "v27_yolo11_fusion_recall_recovery.summary.json"
V27_FUSION = ROOT / "reports" / "Data_v27_yolo11_fusion_recall_recovery" / "locked_fusion_config.json"
DEPLOYMENT = ROOT / "configs" / "deployment.json"
ROLLBACK = ROOT / "configs" / "deployment.pre_v27.json"
SMOKE = ROOT / "runs" / "segment" / "land_seg" / "v28_yolo11_hard_examples_batch1_smoke_seed0_v1" / "weights" / "best.pt"
RUNTIME_INTERVENTION = ROOT / "configs" / "v28_training_runtime_intervention.json"
RUN_NAME = "v28_yolo11_hard_examples_batch1_seed0_v2"
RUN_DIR = ROOT / "runs" / "segment" / "land_seg" / RUN_NAME
CHECKPOINT = RUN_DIR / "weights" / "best.pt"
OUTPUT = ROOT / "reports" / "Data_v28_yolo11_hard_examples"
CANDIDATE_EXPORT = OUTPUT / "candidate_yolo11_valid_predictions"
STATUS = ROOT / "logs" / "v28_yolo11_hard_examples.status.json"
SUMMARY = ROOT / "logs" / "v28_yolo11_hard_examples.summary.json"
LOG = ROOT / "logs" / "v28_yolo11_hard_examples.log"
CHANNELS = ["R", "G", "B", "NDVI", "NDRE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def protocol() -> dict[str, object]:
    return {
        "version": "data_v28_yolo11_hard_examples_protocol_v1",
        "preregistered_at": now_iso(),
        "single_training_change": "4 train-only YOLO11 hard-negative crops plus 8 train-only hard-positive replays",
        "architecture": "YOLO11n-seg + 3x CoordAtt + 3x ECA",
        "primary_model": "YOLO11",
        "auxiliary_model": "frozen YOLOv8; training disabled",
        "channels": CHANNELS,
        "epochs": 60,
        "imgsz": 1024,
        "batch": 1,
        "effective_batch_control": "Ultralytics nbs=64 gradient accumulation",
        "runtime_intervention": "batch=2 exceeded the 6 GB dedicated VRAM budget on dense Mosaic batches; batch=1 is preregistered before the formal candidate run",
        "seed": 0,
        "training_overlap_mask": False,
        "fusion_config_search_allowed": False,
        "fusion_weights": [1.0, 1.0],
        "every_output_requires_yolo11": True,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "sealed_blind_used": False,
        "promotion_gate_relative_to_v27_seed0": {
            "f1_delta_min": 0.015,
            "precision_delta_min": 0.0,
            "recall_delta_min": -0.01,
            "mean_matched_iou_delta_min": -0.005,
        },
        "deployment_write_allowed": False,
    }


def training_command(args: argparse.Namespace, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", "current_5ch",
        "--groups-config", str(ROOT / "configs" / "feature_ablation_groups.json"),
        "--manifest", str(DATASET_MANIFEST),
        "--feature-root", str(DATASET_ROOT),
        "--data", str(DATA),
        "--name", RUN_NAME,
        "--base-model", "yolo11n-seg.pt",
        "--attention", "ca_eca",
        "--extra-init", "mean",
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--warmup-epochs", "1",
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--device", args.device,
        "--seed", "0",
        "--mask-ratio", "4",
        "--no-overlap-mask",
        "--no-cache",
        "--deterministic",
        "--mosaic", "0.5",
        "--mixup", "0.0",
        "--close-mosaic", "15",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


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
    manifest_path = CANDIDATE_EXPORT / "manifest.json"
    if not manifest_path.is_file() or not CHECKPOINT.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def validate_inputs() -> tuple[dict[str, object], dict[str, object]]:
    for path in (
        DATASET_ROOT, DATASET_MANIFEST, DATA, PLAN, BASELINE_FREEZE, V27_SUMMARY, V27_FUSION,
        DEPLOYMENT, ROLLBACK, SMOKE, RUNTIME_INTERVENTION, FEATURE_ROOT, FEATURE_MANIFEST, AUXILIARY_EXPORT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (
        plan.get("state") != "active"
        or plan.get("architecture_policy", {}).get("required_primary") != "YOLO11"
        or plan.get("architecture_policy", {}).get("fusion_weights") != [1.0, 1.0]
        or plan.get("architecture_policy", {}).get("every_output_requires_yolo11") is not True
    ):
        raise RuntimeError("V28 architecture plan contract drifted")
    intervention = json.loads(RUNTIME_INTERVENTION.read_text(encoding="utf-8"))
    if (
        intervention.get("state") != "applied_before_formal_candidate_completion"
        or intervention.get("replacement_run") != RUN_NAME
        or intervention.get("replacement_batch") != 1
        or intervention.get("data_or_model_selection_changed") is not False
        or intervention.get("held_out_test_used") is not False
        or intervention.get("sealed_blind_used") is not False
        or intervention.get("deployment_changed") is not False
    ):
        raise RuntimeError("V28 batch=1 runtime intervention contract drifted")
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if (
        dataset.get("version") != "data_v28_yolo11_hard_examples_train_v1"
        or dataset.get("selection_basis_splits") != ["train"]
        or dataset.get("validation_labels_used_for_selection") is not False
        or dataset.get("held_out_test_used") is not False
        or dataset.get("sealed_blind_used") is not False
        or dataset.get("primary_model") != "YOLO11"
        or dataset.get("auxiliary_model_training") is not False
        or dataset.get("hard_negative_count") != 4
        or dataset.get("hard_positive_replay_count") != 8
        or dataset.get("positive_to_negative_ratio") != 2.0
        or dataset.get("total_train_images") != 222
    ):
        raise RuntimeError("V28 train-only dataset contract drifted")
    yaml_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in yaml_text or "data_release_blind_v1" in yaml_text or "channels: 5" not in yaml_text:
        raise RuntimeError("V28 training YAML exposed forbidden data or changed channel count")
    for relative, expected in (
        ("train/images", 222), ("train/labels", 222), ("train/features", 222), ("valid/features", 60)
    ):
        if len([path for path in (DATASET_ROOT / relative).iterdir() if path.is_file()]) != expected:
            raise RuntimeError(f"V28 artifact count drifted: {relative}")
    freeze = json.loads(BASELINE_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("state") != "frozen" or freeze.get("production_model") != "v27_yolo11_primary_fusion_v1":
        raise RuntimeError("V27 production baseline is not frozen")
    for key, path in (("deployment_config", DEPLOYMENT), ("rollback_config", ROLLBACK), ("fusion", V27_FUSION)):
        if sha256(path) != freeze["artifacts"][key]["sha256"]:
            raise RuntimeError(f"Frozen V27 artifact changed before V28: {key}")
    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    if (
        deployment.get("model_name") != "v27_yolo11_primary_fusion_v1"
        or deployment.get("primary", {}).get("architecture") != "YOLO11n-seg"
        or deployment.get("auxiliary", {}).get("architecture") != "YOLOv8n-seg"
        or deployment.get("every_output_requires_yolo11") is not True
    ):
        raise RuntimeError("Current deployment is no longer the protected V27 baseline")
    locked = json.loads(V27_FUSION.read_text(encoding="utf-8"))
    if (
        locked.get("model_weights") != [1.0, 1.0]
        or locked.get("required_model_indices") != [0]
        or locked.get("held_out_test_used") is not False
        or locked.get("release_blind_holdout_used") is not False
    ):
        raise RuntimeError("V27 fusion contract drifted")
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


def baseline_seed0_metrics() -> dict[str, float | int]:
    summary = json.loads(V27_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("decision") != "freeze_v27_for_shadow_deployment_and_blind_test":
        raise RuntimeError("V27 summary is not the frozen production candidate")
    rows = summary.get("selected", {}).get("seed_rows", [])
    seed0 = next((row for row in rows if row.get("seed") == 0), None)
    if seed0 is None:
        raise RuntimeError("V27 Seed-0 baseline metrics are missing")
    return seed0["metrics"]["fusion"]


def promotion_gate(candidate: dict[str, float | int], baseline: dict[str, float | int]) -> dict[str, object]:
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
        "recall_delta_at_least_minus_0_01": deltas["recall"] >= -0.01 - tolerance,
        "mean_matched_iou_delta_at_least_minus_0_005": deltas["mean_matched_iou"] >= -0.005 - tolerance,
        "every_output_requires_yolo11": True,
    }
    return {"deltas": deltas, "checks": checks, "passed": all(checks.values())}


def main() -> int:
    args = parse_args()
    if args.epochs != 60 or args.imgsz != 1024 or args.batch != 1:
        raise ValueError("V28 formal protocol is locked to 60 epochs, imgsz=1024, batch=1")
    freeze, locked = validate_inputs()
    if args.dry_run:
        print(json.dumps({"protocol": protocol(), "training_command": training_command(args)}, ensure_ascii=False, indent=2))
        return 0
    state: dict[str, object] = {
        "version": "data_v28_yolo11_hard_examples_status_v1",
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
        if not (progress and int(progress["epochs_recorded"]) >= 60 and CHECKPOINT.is_file()):
            last = RUN_DIR / "weights" / "last.pt"
            resume = last if progress and last.is_file() else None
            if RUN_DIR.exists() and resume is None:
                raise RuntimeError(f"Incomplete non-resumable formal run exists: {RUN_DIR}")
            state["stage"] = "training_yolo11_seed0"
            write_json_atomic(STATUS, state)
            run_logged(training_command(args, resume), state)
        progress = read_progress(RUN_DIR / "results.csv")
        if not progress or int(progress["epochs_recorded"]) < 60 or not CHECKPOINT.is_file():
            raise RuntimeError("V28 formal YOLO11 training did not complete")
        state["training_progress"] = progress
        if sha256(DEPLOYMENT) != freeze["artifacts"]["deployment_config"]["sha256"]:
            raise RuntimeError("Production deployment changed during V28 training")
        if not export_complete():
            state["stage"] = "export_locked_valid_predictions"
            write_json_atomic(STATUS, state)
            run_logged(export_command(args.device), state)
        if not export_complete():
            raise RuntimeError("V28 valid prediction export is incomplete")
        state["stage"] = "locked_v27_fusion_valid_gate"
        write_json_atomic(STATUS, state)
        images = sorted((DATASET / "valid" / "images").glob("*.JPG"))
        if len(images) != 60:
            raise RuntimeError(f"Expected 60 locked valid images, got {len(images)}")
        metrics = evaluate_locked_full_resolution(
            images,
            DATASET,
            (CANDIDATE_EXPORT, AUXILIARY_EXPORT),
            ("v28_yolo11_primary", "frozen_yolov8_auxiliary"),
            locked,
        )
        baseline = baseline_seed0_metrics()
        gate = promotion_gate(metrics["fusion"], baseline)
        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != freeze["artifacts"]["deployment_config"]["sha256"]:
            raise RuntimeError("Production deployment changed during V28 valid gate")
        summary = {
            "version": "data_v28_yolo11_hard_examples_summary_v1",
            "created_at": now_iso(),
            "protocol": state["protocol"],
            "candidate_checkpoint": str(CHECKPOINT.resolve()),
            "candidate_checkpoint_sha256": sha256(CHECKPOINT),
            "training_progress": progress,
            "locked_fusion_config": str(V27_FUSION.resolve()),
            "locked_fusion_config_sha256": sha256(V27_FUSION),
            "v27_seed0_baseline_fusion": baseline,
            "v28_locked_valid_metrics": metrics,
            "promotion_gate": gate,
            "gate_passed": gate["passed"],
            "decision": "freeze_v28_for_new_one_time_blind_gate" if gate["passed"] else "reject_v28_retain_v27_production",
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
