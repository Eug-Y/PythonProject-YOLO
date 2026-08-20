"""Evaluate finer mask supervision with staged grouped CV while keeping test sealed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_data_v9_grouped_cv import (
    CONTRACT_PATH as V9_CONTRACT,
    CV_ROOT,
    DEPLOYMENT,
    DEPLOYMENT_SHA256,
    FEATURE_ROOT,
    GROUP,
    GROUPS_CONFIG,
    MANIFEST,
    experiment_name as baseline_experiment_name,
    now_iso,
    sha256,
)
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
STATUS_PATH = ROOT / "logs" / "v11_mask_supervision_cv.status.json"
SUMMARY_PATH = ROOT / "logs" / "v11_mask_supervision_cv.summary.json"
LOG_DIR = ROOT / "logs" / "v11_mask_supervision_cv"
CANDIDATE = "mask_ratio_2"


def experiment_name(fold: int) -> str:
    return f"v11_groupcv_fold{fold}_maskratio2_yolov8n_none_640_seed0_15ep_v1"


def data_yaml(fold: int) -> Path:
    return CV_ROOT / f"fold{fold}" / "data_5ch.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--mean-gate", type=float, default=0.002)
    parser.add_argument("--worst-gate", type=float, default=-0.005)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def training_command(cli: argparse.Namespace, fold: int, *, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", GROUP,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(data_yaml(fold)),
        "--name", experiment_name(fold),
        "--base-model", "yolov8n-seg.pt",
        "--attention", "none",
        "--extra-init", "mean",
        "--epochs", str(cli.epochs),
        "--patience", str(cli.patience),
        "--warmup-epochs", "1",
        "--imgsz", str(cli.imgsz),
        "--batch", str(cli.batch),
        "--device", cli.device,
        "--seed", str(cli.seed),
        "--mask-ratio", "2",
        "--no-cache",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def validate_inputs() -> tuple[dict, str]:
    for path in (V9_CONTRACT, FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint hash differs from the locked baseline")
    contract = json.loads(V9_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("held_out_test_used") or contract.get("test_paths_enumerated"):
        raise RuntimeError("V9 grouped-CV contract indicates sealed-test access")
    for fold in (0, 1, 2):
        text = data_yaml(fold).read_text(encoding="utf-8").lower()
        if "test:" in text or "channels: 5" not in text:
            raise RuntimeError(f"Invalid five-channel grouped-CV YAML: {data_yaml(fold)}")
        baseline = PROJECT / baseline_experiment_name(fold, "h0_refined")
        if read_progress(baseline / "results.csv") is None or not (baseline / "weights" / "best.pt").is_file():
            raise RuntimeError(f"Missing V9 H0 baseline for fold {fold}: {baseline}")
    return contract, deployment_hash


def fold_row(project: Path, fold: int) -> dict[str, object]:
    baseline_name = baseline_experiment_name(fold, "h0_refined")
    candidate_name = experiment_name(fold)
    baseline = read_progress(project / baseline_name / "results.csv")
    candidate = read_progress(project / candidate_name / "results.csv")
    baseline_checkpoint = project / baseline_name / "weights" / "best.pt"
    candidate_checkpoint = project / candidate_name / "weights" / "best.pt"
    row: dict[str, object] = {"fold": fold, "baseline_run": baseline_name, "candidate_run": candidate_name}
    if baseline is None or candidate is None or not baseline_checkpoint.is_file() or not candidate_checkpoint.is_file():
        row["status"] = "incomplete"
        return row
    delta = float(candidate["best_mask_map50_95"]) - float(baseline["best_mask_map50_95"])
    row.update(
        status="complete",
        baseline_mask_map50_95=float(baseline["best_mask_map50_95"]),
        candidate_mask_map50_95=float(candidate["best_mask_map50_95"]),
        delta=delta,
        delta_pp=delta * 100.0,
        precision_delta=float(candidate["best_mask_precision"]) - float(baseline["best_mask_precision"]),
        recall_delta=float(candidate["best_mask_recall"]) - float(baseline["best_mask_recall"]),
        candidate_checkpoint=str(candidate_checkpoint.resolve()),
    )
    return row


def summarize(
    project: Path,
    *,
    mean_gate: float = 0.002,
    worst_gate: float = -0.005,
    deployment: Path | None = DEPLOYMENT,
) -> dict:
    rows = [fold_row(project, fold) for fold in (0, 1, 2)]
    complete = [row for row in rows if row["status"] == "complete"]
    fold0_eligible = rows[0]["status"] == "complete" and float(rows[0]["delta"]) >= worst_gate
    all_complete = len(complete) == 3
    deltas = [float(row["delta"]) for row in complete]
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    worst_delta = min(deltas) if deltas else None
    promoted = bool(
        fold0_eligible
        and all_complete
        and mean_delta is not None
        and worst_delta is not None
        and mean_delta >= mean_gate
        and worst_delta >= worst_gate
    )
    deployment_hash = sha256(deployment) if deployment is not None and deployment.is_file() else None
    stopped_fold0 = rows[0]["status"] == "complete" and not fold0_eligible
    return {
        "version": "data_v11_mask_supervision_cv_summary_v1",
        "created_at": now_iso(),
        "protocol": {
            "model": "yolov8n-seg.pt",
            "channels": ["R", "G", "B", "NDVI", "NDRE"],
            "attention": "none",
            "imgsz": 640,
            "epochs": 15,
            "seed": 0,
            "training_labels": "h0_teacher_refined_mixed",
            "validation_labels": "human_only",
            "control_mask_ratio": 4,
            "candidate_mask_ratio": 2,
            "overlap_mask": True,
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "rows": rows,
        "fold0_eligible": fold0_eligible,
        "stopped_after_fold0": stopped_fold0,
        "all_three_folds_complete": all_complete,
        "mean_delta": mean_delta,
        "mean_delta_pp": mean_delta * 100.0 if mean_delta is not None else None,
        "worst_fold_delta": worst_delta,
        "worst_fold_delta_pp": worst_delta * 100.0 if worst_delta is not None else None,
        "mean_gate": mean_gate,
        "worst_gate": worst_gate,
        "promote_to_yolo11_seed0": promoted,
        "decision": "promote_mask_ratio_2_to_yolo11_seed0" if promoted else ("retain_mask_ratio_4" if stopped_fold0 or all_complete else "incomplete"),
        "deployment_write_allowed": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256 if deployment_hash is not None else None,
    }


def update_overall(state: dict, experiments: list[dict[str, object]], epochs: int, started: float) -> None:
    recorded = 0
    progress_by_fold: dict[str, dict] = {}
    for experiment in experiments:
        progress = read_progress(PROJECT / str(experiment["name"]) / "results.csv")
        if progress:
            progress_by_fold[str(experiment["fold"])] = progress
            recorded += min(int(progress["epochs_recorded"]), epochs)
    target = epochs * len(experiments)
    elapsed = max(time.monotonic() - started, 0.0)
    eta = elapsed / recorded * max(target - recorded, 0) if recorded else None
    state["progress"] = progress_by_fold
    state["overall"] = {
        "epochs_recorded": recorded,
        "target_epochs": target,
        "percent": round(recorded / max(target, 1) * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }


def run_experiments(
    cli: argparse.Namespace,
    experiments: list[dict[str, object]],
    state: dict,
    started: float,
    deployment_hash: str,
    *,
    command_builder=training_command,
    status_path: Path = STATUS_PATH,
    log_dir: Path = LOG_DIR,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    for experiment in experiments:
        fold = int(experiment["fold"])
        if fold not in [int(item["fold"]) for item in state["experiments"]]:
            state["experiments"].append(experiment)
        run_dir = PROJECT / str(experiment["name"])
        progress = read_progress(run_dir / "results.csv")
        best_path = run_dir / "weights" / "best.pt"
        if (fold in state["completed_folds"] or (progress and int(progress["epochs_recorded"]) >= cli.epochs)) and best_path.is_file():
            if fold not in state["completed_folds"]:
                state["completed_folds"].append(fold)
            update_overall(state, state["experiments"], cli.epochs, started)
            write_json_atomic(status_path, state)
            continue
        resume = None
        last_path = run_dir / "weights" / "last.pt"
        if progress and last_path.is_file():
            resume = last_path
        elif run_dir.exists():
            raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
        command = command_builder(cli, fold, resume=resume)
        log_path = log_dir / f"{experiment['name']}.log"
        state["current_fold"] = fold
        state["current_log"] = str(log_path.resolve())
        state["updated_at"] = now_iso()
        write_json_atomic(status_path, state)
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=environment)
            state["current_process_pid"] = process.pid
            write_json_atomic(status_path, state)
            while process.poll() is None:
                state["updated_at"] = now_iso()
                update_overall(state, state["experiments"], cli.epochs, started)
                write_json_atomic(status_path, state)
                time.sleep(max(2.0, cli.poll_seconds))
        progress = read_progress(run_dir / "results.csv")
        if process.returncode != 0:
            raise RuntimeError(f"Run failed with exit code {process.returncode}: {experiment['name']}; see {log_path}")
        if not progress or not best_path.is_file():
            raise RuntimeError(f"Run completed without readable metrics/checkpoint: {experiment['name']}")
        if sha256(DEPLOYMENT) != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V11 screening")
        state["completed_folds"].append(fold)
        state["current_fold"] = None
        state["current_process_pid"] = None
        state["updated_at"] = now_iso()
        update_overall(state, state["experiments"], cli.epochs, started)
        write_json_atomic(status_path, state)


def main() -> int:
    cli = parse_args()
    if (cli.epochs, cli.patience, cli.imgsz, cli.batch, cli.seed) != (15, 6, 640, 4, 0):
        raise ValueError("V11 screening is locked to epochs=15, patience=6, imgsz=640, batch=4, seed=0")
    contract, deployment_hash = validate_inputs()
    first = {"fold": 0, "name": experiment_name(0)}
    if cli.dry_run:
        print(" ".join(training_command(cli, 0)))
        return 0
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    protocol = {"epochs": 15, "patience": 6, "imgsz": 640, "batch": 4, "seed": 0, "mask_ratio": 2}
    completed = previous.get("completed_folds", []) if previous.get("protocol") == protocol else []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        "version": "data_v11_mask_supervision_cv_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "stage": "fold0_screen",
        "protocol": protocol,
        "experiments": [first],
        "completed_folds": list(completed),
        "current_fold": None,
        "current_process_pid": None,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "source_contract": str(V9_CONTRACT.resolve()),
        "validation_label_policy": contract["validation_label_policy"],
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "progress": {},
        "overall": {},
    }
    update_overall(state, state["experiments"], cli.epochs, started)
    write_json_atomic(STATUS_PATH, state)
    try:
        run_experiments(cli, [first], state, started, deployment_hash)
        row0 = fold_row(PROJECT, 0)
        eligible = row0["status"] == "complete" and float(row0["delta"]) >= cli.worst_gate
        state["fold0_result"] = row0
        state["fold0_eligible"] = eligible
        state["stage"] = "three_fold_confirmation" if eligible else "stopped_fold0_regression"
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS_PATH, state)
        if eligible:
            remaining = [{"fold": fold, "name": experiment_name(fold)} for fold in (1, 2)]
            run_experiments(cli, remaining, state, started, deployment_hash)
        summary = summarize(PROJECT, mean_gate=cli.mean_gate, worst_gate=cli.worst_gate)
        write_json_atomic(SUMMARY_PATH, summary)
        if sha256(DEPLOYMENT) != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed while V11 summary was written")
        state["state"] = "completed"
        state["stage"] = "completed"
        state["decision"] = summary["decision"]
        state["summary"] = str(SUMMARY_PATH.resolve())
        state["deployment_sha256_after"] = deployment_hash
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        update_overall(state, state["experiments"], cli.epochs, started)
        write_json_atomic(STATUS_PATH, state)
        return 0
    except BaseException as exc:
        current_hash = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        state["deployment_sha256_after"] = current_hash
        state["deployment_unchanged"] = current_hash == deployment_hash
        update_overall(state, state["experiments"], cli.epochs, started)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
