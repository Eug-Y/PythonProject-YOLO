"""Staged YOLO11n confirmation of B1 augmentation on the V15 no-overlap protocol."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_data_v13_yolo11_overlap_multiseed import (
    BASE_MODEL,
    DATA,
    FEATURE_ROOT,
    GROUP,
    GROUPS_CONFIG,
    MANIFEST,
    PROJECT,
)
from run_data_v15_yolo11_standardized_multiseed import candidate_name as v15_candidate_name
from run_data_v17_seed2_threshold_calibration import fixed_row
from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
V15_SUMMARY = ROOT / "logs" / "v15_yolo11_standardized_multiseed.summary.json"
V19A_SUMMARY = ROOT / "logs" / "v19a_yolov8_boundary_nooverlap_cv.summary.json"
V15_REPORT_ROOT = ROOT / "reports" / "Data_v15_yolo11_standardized_multiseed"
STATUS = ROOT / "logs" / "v19b_yolo11_boundary_nooverlap_multiseed.status.json"
SUMMARY = ROOT / "logs" / "v19b_yolo11_boundary_nooverlap_multiseed.summary.json"
LOG_DIR = ROOT / "logs" / "v19b_yolo11_boundary_nooverlap_multiseed"
OUTPUT = ROOT / "reports" / "Data_v19b_yolo11_boundary_nooverlap_multiseed"
SEEDS = (0, 1, 2)
LOCKED_CONFIDENCE = 0.371


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=4300)
    parser.add_argument("--no-wait-for-gpu", dest="wait_for_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(wait_for_gpu=True)
    return parser.parse_args()


def candidate_name(seed: int) -> str:
    return f"v19_yolo11_c0_nooverlap_b1_seed{seed}_v1"


def candidate_checkpoint(seed: int) -> Path:
    return PROJECT / candidate_name(seed) / "weights" / "best.pt"


def baseline_dir(seed: int) -> Path:
    return V15_REPORT_ROOT / f"seed{seed}" / "candidate"


def candidate_dir(seed: int) -> Path:
    return OUTPUT / f"seed{seed}" / "candidate"


def training_command(
    cli: argparse.Namespace,
    seed: int,
    *,
    resume: Path | None = None,
    smoke_test: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", GROUP,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(DATA),
        "--name", f"smoke_{candidate_name(seed)}" if smoke_test else candidate_name(seed),
        "--base-model", "yolo11n-seg.pt",
        "--attention", "ca_eca",
        "--extra-init", "mean",
        "--epochs", str(cli.epochs),
        "--patience", str(cli.patience),
        "--warmup-epochs", "1",
        "--imgsz", str(cli.imgsz),
        "--batch", str(cli.batch),
        "--workers", str(cli.workers),
        "--device", cli.device,
        "--seed", str(seed),
        "--mask-ratio", "4",
        "--no-overlap-mask",
        "--no-cache",
        "--deterministic",
        "--mosaic", "0.5",
        "--mixup", "0.0",
        "--close-mosaic", "15",
    ]
    if smoke_test:
        command.append("--smoke-test")
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def evaluation_command(seed: int, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "sweep_multichannel_confidence.py"),
        "--checkpoint", str(candidate_checkpoint(seed)),
        "--data", str(DATA),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--output", str(candidate_dir(seed)),
        "--imgsz", "1024",
        "--conf-min", "0.001",
        "--step", "0.005",
        "--device", device,
        "--no-overlap-mask",
    ]


def protocol() -> dict[str, object]:
    return {
        "version": "v19b_yolo11_boundary_nooverlap_multiseed_protocol_v1",
        "preregistered_at": now_iso(),
        "prerequisite": "V19A YOLOv8n grouped-CV gate passed",
        "reference": "same-seed V15 no-overlap candidate",
        "architecture": "YOLO11n-seg",
        "attention": "3x CoordAtt + 3x ECA",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "training_overlap_mask": False,
        "validation_overlap_mask": False,
        "augmentation": {"mosaic": 0.5, "mixup": 0.0, "close_mosaic": 15},
        "epochs": 60,
        "patience": 20,
        "imgsz": 1024,
        "batch": 2,
        "seeds": [0, 1, 2],
        "staged_seed0_first": True,
        "fresh_training_required": True,
        "locked_confidence": LOCKED_CONFIDENCE,
        "candidate_threshold_search_allowed": False,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "gate_thresholds": {
            "seed0_map_delta_min": 0.003,
            "seed0_precision_delta_min": -0.01,
            "seed0_recall_delta_min": -0.01,
            "seed0_fixed_f1_delta_min": 0.0,
            "mean_map_delta_min": 0.003,
            "worst_map_delta_min": -0.003,
            "positive_seeds_min": 2,
            "mean_precision_delta_min": -0.01,
            "mean_recall_delta_min": -0.01,
            "mean_fixed_f1_delta_min": 0.0,
            "worst_fixed_f1_delta_min": -0.003,
        },
        "deployment_write_allowed": False,
    }


def validate_sweep(payload: dict, *, label: str) -> None:
    if payload.get("split") != "val" or payload.get("held_out_test_used") is not False:
        raise RuntimeError(f"V19B {label} sweep is not valid-only")
    if payload.get("validation_overlap_mask") is not False:
        raise RuntimeError(f"V19B {label} sweep did not use overlap_mask=False")
    if payload.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError(f"V19B {label} sweep channel contract drifted")


def validate_inputs() -> tuple[str, str]:
    v19a = json.loads(V19A_SUMMARY.read_text(encoding="utf-8"))
    if v19a.get("gate", {}).get("passed") is not True:
        raise RuntimeError("V19A did not authorize V19B")
    if v19a.get("decision") != "promote_combination_to_yolo11_v19b":
        raise RuntimeError("V19A promotion decision drifted")
    v15 = json.loads(V15_SUMMARY.read_text(encoding="utf-8"))
    if v15.get("gate_passed") is not True or v15.get("held_out_test_used") is not False:
        raise RuntimeError("V15 reference contract is not eligible")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("selection_splits") != ["train", "valid"] or manifest.get("held_out_split_used") is not False:
        raise RuntimeError("V19B feature manifest does not preserve the sealed test")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text or "channels: 5" not in data_text:
        raise RuntimeError("V19B requires a train/valid-only five-channel YAML")
    groups = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))["groups"]
    if groups.get(GROUP) != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError("V19B channel selection changed")
    for seed in SEEDS:
        baseline_summary = baseline_dir(seed) / "confidence_sweep.summary.json"
        baseline_curve = baseline_dir(seed) / "confidence_sweep.csv"
        if not baseline_summary.is_file() or not baseline_curve.is_file():
            raise FileNotFoundError(baseline_summary if not baseline_summary.is_file() else baseline_curve)
        validate_sweep(json.loads(baseline_summary.read_text(encoding="utf-8")), label=f"Seed-{seed} baseline")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V19B")
    return sha256(BASE_MODEL), deployment_hash


def free_gpu_mib(device: str) -> int | None:
    if not str(device).isdigit():
        return None
    completed = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def progress_snapshot(planned_seeds: list[int], started: float) -> dict[str, object]:
    progress: dict[str, object] = {}
    recorded = 0
    for seed in planned_seeds:
        item = read_progress(PROJECT / candidate_name(seed) / "results.csv")
        if item:
            progress[str(seed)] = item
            recorded += min(int(item["epochs_recorded"]), 60)
    target = len(planned_seeds) * 60
    elapsed = max(time.monotonic() - started, 0.0)
    eta = elapsed / recorded * max(target - recorded, 0) if recorded else None
    return {
        "runs": progress,
        "epochs_recorded": recorded,
        "target_epochs": target,
        "percent": round(recorded / max(target, 1) * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }


def update_state(state: dict, started: float) -> None:
    state["progress"] = progress_snapshot([int(seed) for seed in state["planned_seeds"]], started)
    state["updated_at"] = now_iso()
    write_json_atomic(STATUS, state)


def wait_for_gpu(cli: argparse.Namespace, state: dict, started: float) -> None:
    while True:
        free = free_gpu_mib(cli.device)
        state["gpu"] = {"device": cli.device, "free_mib": free, "required_free_mib": cli.min_free_gpu_mib}
        if free is None or free >= cli.min_free_gpu_mib:
            return
        state["stage"] = "waiting_for_gpu"
        update_state(state, started)
        if not cli.wait_for_gpu:
            raise RuntimeError(f"Only {free} MiB GPU memory is free; V19B requires {cli.min_free_gpu_mib} MiB")
        time.sleep(max(cli.gpu_poll_seconds, 10.0))


def run_training(cli: argparse.Namespace, seed: int, state: dict, started: float, deployment_hash: str) -> None:
    run_dir = PROJECT / candidate_name(seed)
    best = candidate_checkpoint(seed)
    progress = read_progress(run_dir / "results.csv")
    if progress and int(progress["epochs_recorded"]) >= cli.epochs and best.is_file():
        return
    resume = None
    last = run_dir / "weights" / "last.pt"
    if progress and last.is_file():
        resume = last
    elif run_dir.exists():
        raise RuntimeError(f"Incomplete V19B run has no resumable checkpoint: {run_dir}")
    wait_for_gpu(cli, state, started)
    command = training_command(cli, seed, resume=resume)
    log_path = LOG_DIR / f"{candidate_name(seed)}.log"
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    state.update(stage="training", current_seed=seed, current_log=str(log_path.resolve()))
    update_state(state, started)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        state["current_process_pid"] = process.pid
        update_state(state, started)
        while process.poll() is None:
            time.sleep(max(cli.poll_seconds, 5.0))
            update_state(state, started)
    state["current_process_pid"] = None
    progress = read_progress(run_dir / "results.csv")
    if process.returncode != 0 or not progress or not best.is_file():
        raise RuntimeError(f"V19B Seed-{seed} training failed; see {log_path}")
    if sha256(DEPLOYMENT) != deployment_hash:
        raise RuntimeError("Deployment checkpoint changed during V19B training")


def run_evaluation(cli: argparse.Namespace, seed: int, state: dict, started: float) -> dict:
    result_path = candidate_dir(seed) / "confidence_sweep.summary.json"
    state.update(stage="standardized_validation", current_evaluation={"seed": seed})
    update_state(state, started)
    if not result_path.is_file():
        log_path = LOG_DIR / f"seed{seed}_candidate_evaluation.log"
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            completed = subprocess.run(
                evaluation_command(seed, cli.device),
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"V19B Seed-{seed} evaluation failed; see {log_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_sweep(payload, label=f"Seed-{seed} candidate")
    return payload


def pair_row(seed: int) -> dict[str, object]:
    baseline_summary_path = baseline_dir(seed) / "confidence_sweep.summary.json"
    candidate_summary_path = candidate_dir(seed) / "confidence_sweep.summary.json"
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    validate_sweep(baseline, label=f"Seed-{seed} baseline")
    validate_sweep(candidate, label=f"Seed-{seed} candidate")
    if baseline.get("target_instances") != candidate.get("target_instances"):
        raise RuntimeError(f"V19B Seed-{seed} target instance count changed")
    row: dict[str, object] = {
        "seed": seed,
        "baseline_checkpoint": str((PROJECT / v15_candidate_name(seed) / "weights" / "best.pt").resolve()),
        "candidate_checkpoint": str(candidate_checkpoint(seed).resolve()),
        "candidate_checkpoint_sha256": sha256(candidate_checkpoint(seed)),
    }
    keys = {
        "map": "metrics/mAP50-95(M)",
        "map50": "metrics/mAP50(M)",
        "precision": "metrics/precision(M)",
        "recall": "metrics/recall(M)",
    }
    for short, key in keys.items():
        base = float(baseline["aggregate_metrics"][key])
        trial = float(candidate["aggregate_metrics"][key])
        row[f"baseline_{short}"] = base
        row[f"candidate_{short}"] = trial
        row[f"{short}_delta"] = trial - base
        row[f"{short}_delta_pp"] = (trial - base) * 100.0
    baseline_fixed = fixed_row(baseline_dir(seed) / "confidence_sweep.csv", LOCKED_CONFIDENCE)
    candidate_fixed = fixed_row(candidate_dir(seed) / "confidence_sweep.csv", LOCKED_CONFIDENCE)
    row["locked_confidence"] = LOCKED_CONFIDENCE
    row["baseline_fixed"] = baseline_fixed
    row["candidate_fixed"] = candidate_fixed
    for metric in ("precision", "recall", "f1"):
        delta = float(candidate_fixed[metric]) - float(baseline_fixed[metric])
        row[f"fixed_{metric}_delta"] = delta
        row[f"fixed_{metric}_delta_pp"] = delta * 100.0
    return row


def summarize(rows: list[dict[str, object]], deployment_hash: str) -> dict[str, object]:
    seed0 = next((row for row in rows if int(row["seed"]) == 0), None)
    seed0_checks = {
        "seed0_available": seed0 is not None,
        "seed0_map_delta_at_least_0_003": bool(seed0 and float(seed0["map_delta"]) >= 0.003),
        "seed0_precision_delta_not_below_minus_0_01": bool(seed0 and float(seed0["precision_delta"]) >= -0.01),
        "seed0_recall_delta_not_below_minus_0_01": bool(seed0 and float(seed0["recall_delta"]) >= -0.01),
        "seed0_fixed_f1_nonnegative": bool(seed0 and float(seed0["fixed_f1_delta"]) >= 0.0),
    }
    seed0_passed = all(seed0_checks.values())
    all_complete = len(rows) == 3 and {int(row["seed"]) for row in rows} == {0, 1, 2}
    maps = [float(row["map_delta"]) for row in rows]
    precisions = [float(row["precision_delta"]) for row in rows]
    recalls = [float(row["recall_delta"]) for row in rows]
    fixed_f1s = [float(row["fixed_f1_delta"]) for row in rows]
    final_checks = {
        "seed0_passed": seed0_passed,
        "three_seeds_complete": all_complete,
        "mean_map_delta_at_least_0_003": bool(all_complete and statistics.fmean(maps) >= 0.003),
        "worst_map_delta_not_below_minus_0_003": bool(all_complete and min(maps) >= -0.003),
        "at_least_two_positive_map_seeds": bool(all_complete and sum(value > 0 for value in maps) >= 2),
        "mean_precision_delta_not_below_minus_0_01": bool(all_complete and statistics.fmean(precisions) >= -0.01),
        "mean_recall_delta_not_below_minus_0_01": bool(all_complete and statistics.fmean(recalls) >= -0.01),
        "mean_fixed_f1_nonnegative": bool(all_complete and statistics.fmean(fixed_f1s) >= 0.0),
        "worst_fixed_f1_not_below_minus_0_003": bool(all_complete and min(fixed_f1s) >= -0.003),
    }
    passed = all(final_checks.values())
    best = max(rows, key=lambda row: float(row["candidate_map"])) if passed else None
    return {
        "version": "v19b_yolo11_boundary_nooverlap_multiseed_summary_v1",
        "created_at": now_iso(),
        "protocol": protocol(),
        "rows": rows,
        "seed0_checks": seed0_checks,
        "seed0_passed": seed0_passed,
        "gate_checks": final_checks,
        "gate_metrics": {
            "mean_map_delta": statistics.fmean(maps) if maps else None,
            "mean_map_delta_pp": statistics.fmean(maps) * 100.0 if maps else None,
            "worst_map_delta": min(maps) if maps else None,
            "positive_map_seeds": sum(value > 0 for value in maps),
            "mean_precision_delta": statistics.fmean(precisions) if precisions else None,
            "mean_recall_delta": statistics.fmean(recalls) if recalls else None,
            "mean_fixed_f1_delta": statistics.fmean(fixed_f1s) if fixed_f1s else None,
            "worst_fixed_f1_delta": min(fixed_f1s) if fixed_f1s else None,
        },
        "gate_passed": passed,
        "decision": (
            "freeze_v19b_candidate_for_review_no_auto_deployment" if passed else
            "continue_multiseed" if seed0_passed and not all_complete else
            "reject_v19b_retain_v15_seed2"
        ),
        "candidate_checkpoint": best["candidate_checkpoint"] if best else None,
        "candidate_checkpoint_sha256": best["candidate_checkpoint_sha256"] if best else None,
        "held_out_test_used": False,
        "deployment_eligible": False,
        "deployment_write_allowed": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
    }


def main() -> int:
    cli = parse_args()
    if (cli.epochs, cli.patience, cli.imgsz, cli.batch) != (60, 20, 1024, 2):
        raise ValueError("V19B is locked to epochs=60, patience=20, imgsz=1024, batch=2")
    base_hash, deployment_hash = validate_inputs()
    locked_protocol = protocol()
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if previous.get("protocol", {}).get("version") == locked_protocol["version"]:
        locked_protocol = previous["protocol"]
    if cli.dry_run:
        print(json.dumps(locked_protocol, ensure_ascii=False, indent=2))
        print(subprocess.list2cmdline(training_command(cli, 0)))
        print(subprocess.list2cmdline(evaluation_command(0, cli.device)))
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        "version": "v19b_yolo11_boundary_nooverlap_multiseed_status_v1",
        "state": "running",
        "stage": "seed0_confirmation",
        "runner_pid": os.getpid(),
        "started_at": previous.get("started_at", now_iso()),
        "updated_at": now_iso(),
        "protocol": locked_protocol,
        "planned_seeds": [0],
        "completed_seeds": previous.get("completed_seeds", []),
        "current_seed": None,
        "current_process_pid": None,
        "base_model_sha256": base_hash,
        "v19a_gate_summary": str(V19A_SUMMARY.resolve()),
        "v15_reference_summary": str(V15_SUMMARY.resolve()),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "progress": {},
    }
    update_state(state, started)
    try:
        rows: list[dict[str, object]] = []
        for seed in SEEDS:
            if seed > 0:
                interim = summarize(rows, deployment_hash)
                if not interim["seed0_passed"]:
                    break
                state["planned_seeds"] = [0, 1, 2]
                update_state(state, started)
            run_training(cli, seed, state, started, deployment_hash)
            run_evaluation(cli, seed, state, started)
            rows.append(pair_row(seed))
            if seed not in state["completed_seeds"]:
                state["completed_seeds"].append(seed)
            state.update(current_seed=None, current_process_pid=None, current_evaluation=None)
            update_state(state, started)

        summary = summarize(rows, deployment_hash)
        summary["protocol"] = locked_protocol
        write_json_atomic(SUMMARY, summary)
        current_hash = sha256(DEPLOYMENT)
        if current_hash != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V19B")
        state.update(
            state="complete",
            stage="complete",
            decision=summary["decision"],
            summary=str(SUMMARY.resolve()),
            deployment_sha256_after=current_hash,
            deployment_unchanged=True,
            completed_at=now_iso(),
        )
        update_state(state, started)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        state.update(
            state="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {exc}",
            failed_at=now_iso(),
            deployment_sha256_after=sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None,
        )
        update_state(state, started)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
