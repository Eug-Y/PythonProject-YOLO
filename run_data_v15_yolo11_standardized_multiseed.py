"""Fresh staged YOLO11 confirmation with one independent-overlap validation contract."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256
from run_data_v13_yolo11_overlap_multiseed import (
    BASE_MODEL,
    DATA,
    FEATURE_ROOT,
    GROUP,
    GROUPS_CONFIG,
    MANIFEST,
    PROJECT,
    control_name,
)
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
V14_SUMMARY = ROOT / "logs" / "v14_standardized_overlap_audit.summary.json"
STATUS = ROOT / "logs" / "v15_yolo11_standardized_multiseed.status.json"
SUMMARY = ROOT / "logs" / "v15_yolo11_standardized_multiseed.summary.json"
LOG_DIR = ROOT / "logs" / "v15_yolo11_standardized_multiseed"
EVAL_ROOT = ROOT / "reports" / "Data_v15_yolo11_standardized_multiseed"
SEEDS = (0, 1, 2)


def candidate_name(seed: int) -> str:
    return f"v15_yolo11_c0_nooverlap_standardized_seed{seed}_fresh_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=4300)
    parser.add_argument("--no-wait-for-gpu", dest="wait_for_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(wait_for_gpu=True)
    return parser.parse_args()


def training_command(cli: argparse.Namespace, seed: int, *, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", GROUP,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(DATA),
        "--name", candidate_name(seed),
        "--base-model", "yolo11n-seg.pt",
        "--attention", "ca_eca",
        "--extra-init", "mean",
        "--epochs", str(cli.epochs),
        "--patience", str(cli.patience),
        "--warmup-epochs", "1",
        "--imgsz", str(cli.imgsz),
        "--batch", str(cli.batch),
        "--device", cli.device,
        "--workers", str(cli.workers),
        "--seed", str(seed),
        "--mask-ratio", "4",
        "--no-overlap-mask",
        "--no-cache",
        "--deterministic",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def evaluation_dir(seed: int, variant: str) -> Path:
    return EVAL_ROOT / f"seed{seed}" / variant


def evaluation_checkpoint(seed: int, variant: str) -> Path:
    run = control_name(seed) if variant == "control" else candidate_name(seed)
    return PROJECT / run / "weights" / "best.pt"


def evaluation_command(seed: int, variant: str, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "sweep_multichannel_confidence.py"),
        "--checkpoint", str(evaluation_checkpoint(seed, variant)),
        "--data", str(DATA),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--output", str(evaluation_dir(seed, variant)),
        "--imgsz", "1024",
        "--conf-min", "0.001",
        "--step", "0.005",
        "--device", device,
        "--no-overlap-mask",
    ]


def validate_inputs() -> tuple[str, str]:
    v14 = json.loads(V14_SUMMARY.read_text(encoding="utf-8"))
    if v14.get("gate", {}).get("passed") is not True or v14.get("decision") != "eligible_for_fresh_yolo11_seed0":
        raise RuntimeError("V14 standardized grouped-CV gate did not authorize fresh YOLO11 Seed-0 training")
    if v14.get("protocol", {}).get("held_out_test_used") is not False:
        raise RuntimeError("V14 does not preserve the sealed-test contract")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("held_out_split_used") is not False or manifest.get("selection_splits") != ["train", "valid"]:
        raise RuntimeError("Feature manifest does not preserve the sealed-test contract")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text or "channels: 5" not in data_text:
        raise RuntimeError("V15 requires a train/valid-only five-channel YAML")
    groups = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))["groups"]
    if groups.get(GROUP) != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError("V15 channel contract changed")
    for seed in SEEDS:
        if not evaluation_checkpoint(seed, "control").is_file():
            raise FileNotFoundError(evaluation_checkpoint(seed, "control"))
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V15")
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


def wait_for_gpu(cli: argparse.Namespace, state: dict, started: float) -> None:
    while True:
        free = free_gpu_mib(cli.device)
        state["gpu"] = {"device": cli.device, "free_mib": free, "required_free_mib": cli.min_free_gpu_mib}
        if free is None or free >= cli.min_free_gpu_mib:
            state["stage"] = "training"
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS, state)
            return
        state["stage"] = "waiting_for_gpu"
        state["overall"]["elapsed_seconds"] = round(time.monotonic() - started, 1)
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS, state)
        if not cli.wait_for_gpu:
            raise RuntimeError(f"Only {free} MiB GPU memory is free; V15 requires {cli.min_free_gpu_mib} MiB")
        time.sleep(max(10.0, cli.gpu_poll_seconds))


def update_progress(state: dict, started: float) -> None:
    recorded = 0
    progress = {}
    for seed in state["planned_seeds"]:
        item = read_progress(PROJECT / candidate_name(int(seed)) / "results.csv")
        if item:
            progress[str(seed)] = item
            recorded += min(int(item["epochs_recorded"]), 60)
    target = len(state["planned_seeds"]) * 60
    elapsed = max(time.monotonic() - started, 0.0)
    eta = elapsed / recorded * max(target - recorded, 0) if recorded else None
    state["progress"] = progress
    state["overall"] = {
        "epochs_recorded": recorded,
        "target_epochs": target,
        "percent": round(100.0 * recorded / max(target, 1), 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }
    state["updated_at"] = now_iso()
    write_json_atomic(STATUS, state)


def run_training(cli: argparse.Namespace, seed: int, state: dict, started: float, deployment_hash: str) -> None:
    run_dir = PROJECT / candidate_name(seed)
    best = run_dir / "weights" / "best.pt"
    progress = read_progress(run_dir / "results.csv")
    if progress and int(progress["epochs_recorded"]) >= 60 and best.is_file():
        return
    resume = None
    last = run_dir / "weights" / "last.pt"
    if progress and last.is_file():
        resume = last
    elif run_dir.exists():
        raise RuntimeError(f"Incomplete V15 run has no resumable checkpoint: {run_dir}")
    wait_for_gpu(cli, state, started)
    command = training_command(cli, seed, resume=resume)
    log_path = LOG_DIR / f"{candidate_name(seed)}.log"
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    state.update(stage="training", current_seed=seed, current_log=str(log_path.resolve()))
    write_json_atomic(STATUS, state)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
        state["current_process_pid"] = process.pid
        write_json_atomic(STATUS, state)
        while process.poll() is None:
            update_progress(state, started)
            time.sleep(max(5.0, cli.poll_seconds))
    progress = read_progress(run_dir / "results.csv")
    if process.returncode != 0 or not progress or not best.is_file():
        raise RuntimeError(f"V15 seed {seed} training failed; see {log_path}")
    if sha256(DEPLOYMENT) != deployment_hash:
        raise RuntimeError("Deployment checkpoint changed during V15 training")


def run_evaluation(cli: argparse.Namespace, seed: int, variant: str, state: dict) -> dict:
    result_path = evaluation_dir(seed, variant) / "confidence_sweep.summary.json"
    if not result_path.is_file():
        command = evaluation_command(seed, variant, cli.device)
        log_path = LOG_DIR / f"seed{seed}_{variant}_evaluation.log"
        state.update(stage="standardized_validation", current_evaluation={"seed": seed, "variant": variant})
        write_json_atomic(STATUS, state)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"V15 standardized evaluation failed; see {log_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("held_out_test_used") is not False:
        raise RuntimeError("V15 evaluation left the valid-only contract")
    if payload.get("validation_overlap_mask") is not False:
        raise RuntimeError("V15 validation mask semantics drifted")
    return payload


def pair_row(seed: int) -> dict[str, object]:
    control = json.loads((evaluation_dir(seed, "control") / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((evaluation_dir(seed, "candidate") / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
    keys = {
        "map": "metrics/mAP50-95(M)",
        "map50": "metrics/mAP50(M)",
        "precision": "metrics/precision(M)",
        "recall": "metrics/recall(M)",
    }
    row: dict[str, object] = {
        "seed": seed,
        "control_checkpoint": str(evaluation_checkpoint(seed, "control").resolve()),
        "candidate_checkpoint": str(evaluation_checkpoint(seed, "candidate").resolve()),
        "candidate_checkpoint_sha256": sha256(evaluation_checkpoint(seed, "candidate")),
    }
    for short, key in keys.items():
        base = float(control["aggregate_metrics"][key])
        trial = float(candidate["aggregate_metrics"][key])
        row[f"control_{short}"] = base
        row[f"candidate_{short}"] = trial
        row[f"{short}_delta"] = trial - base
        row[f"{short}_delta_pp"] = (trial - base) * 100.0
    return row


def summarize(rows: list[dict[str, object]], deployment_hash: str) -> dict:
    seed0 = next((row for row in rows if int(row["seed"]) == 0), None)
    seed0_passed = bool(
        seed0
        and float(seed0["map_delta"]) >= 0.002
        and float(seed0["precision_delta"]) >= -0.01
        and float(seed0["recall_delta"]) >= -0.01
    )
    all_complete = len(rows) == 3
    maps = [float(row["map_delta"]) for row in rows]
    precisions = [float(row["precision_delta"]) for row in rows]
    recalls = [float(row["recall_delta"]) for row in rows]
    checks = {
        "seed0_passed": seed0_passed,
        "three_seeds_complete": all_complete,
        "mean_map_delta_at_least_0_003": bool(all_complete and statistics.fmean(maps) >= 0.003),
        "worst_map_delta_not_below_minus_0_003": bool(all_complete and min(maps) >= -0.003),
        "at_least_two_positive_seeds": bool(all_complete and sum(value > 0 for value in maps) >= 2),
        "mean_precision_delta_not_below_minus_0_01": bool(all_complete and statistics.fmean(precisions) >= -0.01),
        "mean_recall_delta_not_below_minus_0_01": bool(all_complete and statistics.fmean(recalls) >= -0.01),
    }
    passed = all(checks.values())
    return {
        "version": "v15_yolo11_standardized_multiseed_summary_v1",
        "created_at": now_iso(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "validation_overlap_mask": False,
        "v13_gate_overridden": False,
        "rows": rows,
        "seed0_passed": seed0_passed,
        "gate_checks": checks,
        "gate_passed": passed,
        "decision": (
            "freeze_candidate_for_review_no_auto_deployment" if passed else
            "continue_multiseed" if seed0_passed and not all_complete else
            "reject_retain_current_deployment"
        ),
        "candidate_checkpoint": max(rows, key=lambda row: float(row["candidate_map"]))["candidate_checkpoint"] if passed else None,
        "deployment_eligible": False,
        "deployment_write_allowed": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
    }


def main() -> int:
    cli = parse_args()
    if (cli.epochs, cli.patience, cli.imgsz, cli.batch) != (60, 20, 1024, 2):
        raise ValueError("V15 is locked to 60 epochs, patience 20, imgsz 1024, batch 2")
    base_hash, deployment_hash = validate_inputs()
    protocol = {
        "version": "v15_yolo11_standardized_multiseed_protocol_v1",
        "preregistered_at": now_iso(),
        "architecture": "YOLO11n-seg",
        "attention": "3x CoordAtt + 3x ECA",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "training_overlap_mask": False,
        "validation_overlap_mask": False,
        "epochs": 60,
        "patience": 20,
        "imgsz": 1024,
        "batch": 2,
        "seeds": [0, 1, 2],
        "staged_seed0_first": True,
        "fresh_training_required": True,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "gate_thresholds": {
            "seed0_map_delta_min": 0.002,
            "mean_map_delta_min": 0.003,
            "worst_map_delta_min": -0.003,
            "positive_seeds_min": 2,
            "precision_delta_min": -0.01,
            "recall_delta_min": -0.01,
        },
        "v13_gate_override_allowed": False,
        "deployment_write_allowed": False,
    }
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if previous.get("protocol", {}).get("version") == protocol["version"]:
        protocol = previous["protocol"]
    if cli.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        print(subprocess.list2cmdline(training_command(cli, 0)))
        print(subprocess.list2cmdline(evaluation_command(0, "candidate", cli.device)))
        return 0
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        "version": "v15_yolo11_standardized_multiseed_status_v1",
        "state": "running",
        "stage": "seed0_confirmation",
        "runner_pid": os.getpid(),
        "started_at": previous.get("started_at", now_iso()),
        "updated_at": now_iso(),
        "protocol": protocol,
        "planned_seeds": [0],
        "completed_seeds": previous.get("completed_seeds", []),
        "current_seed": None,
        "current_process_pid": None,
        "base_model_sha256": base_hash,
        "v14_gate_summary": str(V14_SUMMARY.resolve()),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "overall": {},
    }
    update_progress(state, started)
    try:
        rows: list[dict[str, object]] = []
        for seed in SEEDS:
            if seed > 0:
                interim = summarize(rows, deployment_hash)
                if not interim["seed0_passed"]:
                    break
                state["planned_seeds"] = [0, 1, 2]
            run_training(cli, seed, state, started, deployment_hash)
            run_evaluation(cli, seed, "control", state)
            run_evaluation(cli, seed, "candidate", state)
            rows.append(pair_row(seed))
            if seed not in state["completed_seeds"]:
                state["completed_seeds"].append(seed)
            state.update(current_seed=None, current_process_pid=None, current_evaluation=None)
            write_json_atomic(STATUS, state)
        summary = summarize(rows, deployment_hash)
        write_json_atomic(SUMMARY, summary)
        current_hash = sha256(DEPLOYMENT)
        if current_hash != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V15")
        state.update(
            state="complete",
            stage="complete",
            decision=summary["decision"],
            summary=str(SUMMARY.resolve()),
            deployment_sha256_after=current_hash,
            deployment_unchanged=True,
            completed_at=now_iso(),
        )
        update_progress(state, started)
        return 0
    except BaseException as exc:
        state.update(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            failed_at=now_iso(),
            deployment_sha256_after=sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None,
        )
        update_progress(state, started)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
