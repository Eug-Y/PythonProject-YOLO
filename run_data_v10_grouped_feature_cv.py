"""Screen soil/vegetation channels with staged grouped CV while keeping test sealed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_data_v9_grouped_cv import (
    DEPLOYMENT,
    DEPLOYMENT_SHA256,
    experiment_name as baseline_experiment_name,
    now_iso,
    sha256,
)
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
CV_ROOT = ROOT / "configs" / "generated" / "data_v10_grouped_feature_cv"
CONTRACT_PATH = CV_ROOT / "contract.json"
FEATURE_ROOT = ROOT / "datasets" / "Data_v7_soil_vegetation_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "feature_ablation_groups_v7_soil_vegetation.json"
STATUS_PATH = ROOT / "logs" / "v10_grouped_feature_cv.status.json"
SUMMARY_PATH = ROOT / "logs" / "v10_grouped_feature_cv.summary.json"
LOG_DIR = ROOT / "logs" / "v10_grouped_feature_cv"
CANDIDATES = ("add_re_bsi_proxy_6ch", "add_mcari_6ch", "soil_vegetation_7ch")


def experiment_name(fold: int, group: str) -> str:
    return f"v10_groupcv_fold{fold}_{group}_yolov8n_none_640_seed0_15ep_v1"


def data_yaml(fold: int, channel_count: int) -> Path:
    return CV_ROOT / f"fold{fold}" / f"data_{channel_count}ch.yaml"


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


def load_groups() -> dict[str, list[str]]:
    return json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))["groups"]


def training_command(
    cli: argparse.Namespace,
    fold: int,
    group: str,
    *,
    resume: Path | None = None,
) -> list[str]:
    channels = len(load_groups()[group])
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", group,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(data_yaml(fold, channels)),
        "--name", experiment_name(fold, group),
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
        "--no-cache",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def validate_inputs() -> tuple[dict, str]:
    for path in (CONTRACT_PATH, FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint hash differs from the locked baseline")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    pairing = contract.get("feature_pairing", {})
    if not pairing.get("exact_array_match") or pairing.get("checked_arrays") != 270:
        raise RuntimeError("Shared five-channel feature pairing is not verified")
    if contract.get("held_out_test_used") or contract.get("test_paths_enumerated"):
        raise RuntimeError("V10 contract indicates sealed-test access")
    groups = load_groups()
    for group in CANDIDATES:
        channels = len(groups[group])
        for fold in (0, 1, 2):
            path = data_yaml(fold, channels)
            text = path.read_text(encoding="utf-8").lower()
            if "test:" in text or f"channels: {channels}" not in text:
                raise RuntimeError(f"Invalid grouped-CV feature YAML: {path}")
    for fold in (0, 1, 2):
        baseline = PROJECT / baseline_experiment_name(fold, "h0_refined")
        if read_progress(baseline / "results.csv") is None or not (baseline / "weights" / "best.pt").is_file():
            raise RuntimeError(f"Missing V9 H0 baseline for fold {fold}: {baseline}")
    return contract, deployment_hash


def fold_row(project: Path, fold: int, group: str) -> dict[str, object]:
    baseline_name = baseline_experiment_name(fold, "h0_refined")
    candidate_name = experiment_name(fold, group)
    baseline = read_progress(project / baseline_name / "results.csv")
    candidate = read_progress(project / candidate_name / "results.csv")
    baseline_checkpoint = project / baseline_name / "weights" / "best.pt"
    candidate_checkpoint = project / candidate_name / "weights" / "best.pt"
    row: dict[str, object] = {
        "fold": fold,
        "group": group,
        "baseline_run": baseline_name,
        "candidate_run": candidate_name,
    }
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


def select_candidate(project: Path, *, worst_gate: float = -0.005) -> tuple[str | None, list[dict[str, object]]]:
    rows = [fold_row(project, 0, group) for group in CANDIDATES]
    eligible = [row for row in rows if row["status"] == "complete" and float(row["delta"]) >= worst_gate]
    selected = max(eligible, key=lambda row: float(row["delta"]))["group"] if eligible else None
    return str(selected) if selected is not None else None, sorted(
        rows,
        key=lambda row: float(row.get("delta", float("-inf"))),
        reverse=True,
    )


def summarize(
    project: Path,
    *,
    mean_gate: float = 0.002,
    worst_gate: float = -0.005,
    deployment: Path | None = DEPLOYMENT,
) -> dict:
    selected, screening_rows = select_candidate(project, worst_gate=worst_gate)
    fold_rows = [fold_row(project, fold, selected) for fold in (0, 1, 2)] if selected else []
    complete = [row for row in fold_rows if row["status"] == "complete"]
    all_complete = selected is not None and len(complete) == 3
    deltas = [float(row["delta"]) for row in complete]
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    worst_delta = min(deltas) if deltas else None
    promoted = bool(
        all_complete
        and mean_delta is not None
        and worst_delta is not None
        and mean_delta >= mean_gate
        and worst_delta >= worst_gate
    )
    deployment_hash = sha256(deployment) if deployment is not None and deployment.is_file() else None
    return {
        "version": "data_v10_grouped_feature_cv_summary_v1",
        "created_at": now_iso(),
        "protocol": {
            "model": "yolov8n-seg.pt",
            "base_channels": ["R", "G", "B", "NDVI", "NDRE"],
            "attention": "none",
            "imgsz": 640,
            "epochs": 15,
            "seed": 0,
            "training_labels": "h0_teacher_refined_mixed",
            "validation_labels": "human_only",
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "screening_rows": screening_rows,
        "selected_candidate": selected,
        "fold_rows": fold_rows,
        "all_three_folds_complete": all_complete,
        "mean_delta": mean_delta,
        "mean_delta_pp": mean_delta * 100.0 if mean_delta is not None else None,
        "worst_fold_delta": worst_delta,
        "worst_fold_delta_pp": worst_delta * 100.0 if worst_delta is not None else None,
        "mean_gate": mean_gate,
        "worst_gate": worst_gate,
        "promote_to_yolo11_seed0": promoted,
        "decision": f"promote_{selected}_to_yolo11_seed0" if promoted else ("retain_5ch_h0" if selected is None or all_complete else "incomplete"),
        "deployment_write_allowed": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256 if deployment_hash is not None else None,
    }


def update_overall(state: dict, experiments: list[dict[str, object]], epochs: int, started: float) -> None:
    recorded = 0
    progress_by_pair: dict[str, dict] = {}
    for experiment in experiments:
        progress = read_progress(PROJECT / str(experiment["name"]) / "results.csv")
        if progress:
            progress_by_pair[str(experiment["pair"])] = progress
            recorded += min(int(progress["epochs_recorded"]), epochs)
    target = epochs * len(experiments)
    elapsed = max(time.monotonic() - started, 0.0)
    eta = elapsed / recorded * max(target - recorded, 0) if recorded else None
    state["progress"] = progress_by_pair
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
) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    previous_completed = set(state.get("completed_pairs", []))
    for experiment in experiments:
        pair = str(experiment["pair"])
        if pair not in [str(item["pair"]) for item in state["experiments"]]:
            state["experiments"].append(experiment)
        run_dir = PROJECT / str(experiment["name"])
        progress = read_progress(run_dir / "results.csv")
        best_path = run_dir / "weights" / "best.pt"
        if (pair in previous_completed or (progress and int(progress["epochs_recorded"]) >= cli.epochs)) and best_path.is_file():
            if pair not in state["completed_pairs"]:
                state["completed_pairs"].append(pair)
            update_overall(state, state["experiments"], cli.epochs, started)
            write_json_atomic(STATUS_PATH, state)
            continue
        resume = None
        last_path = run_dir / "weights" / "last.pt"
        if progress and last_path.is_file():
            resume = last_path
        elif run_dir.exists():
            raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
        command = training_command(cli, int(experiment["fold"]), str(experiment["group"]), resume=resume)
        log_path = LOG_DIR / f"{experiment['name']}.log"
        state["current_pair"] = pair
        state["current_log"] = str(log_path.resolve())
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS_PATH, state)
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=environment)
            state["current_process_pid"] = process.pid
            write_json_atomic(STATUS_PATH, state)
            while process.poll() is None:
                state["updated_at"] = now_iso()
                update_overall(state, state["experiments"], cli.epochs, started)
                write_json_atomic(STATUS_PATH, state)
                time.sleep(max(2.0, cli.poll_seconds))
        progress = read_progress(run_dir / "results.csv")
        if process.returncode != 0:
            raise RuntimeError(f"Run failed with exit code {process.returncode}: {experiment['name']}; see {log_path}")
        if not progress or not best_path.is_file():
            raise RuntimeError(f"Run completed without readable metrics/checkpoint: {experiment['name']}")
        if sha256(DEPLOYMENT) != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V10 screening")
        state["completed_pairs"].append(pair)
        state["current_pair"] = None
        state["current_process_pid"] = None
        state["updated_at"] = now_iso()
        update_overall(state, state["experiments"], cli.epochs, started)
        write_json_atomic(STATUS_PATH, state)


def main() -> int:
    cli = parse_args()
    if (cli.epochs, cli.patience, cli.imgsz, cli.batch, cli.seed) != (15, 6, 640, 4, 0):
        raise ValueError("V10 screening is locked to epochs=15, patience=6, imgsz=640, batch=4, seed=0")
    contract, deployment_hash = validate_inputs()
    screen = [
        {"fold": 0, "group": group, "pair": f"fold0_{group}", "name": experiment_name(0, group)}
        for group in CANDIDATES
    ]
    if cli.dry_run:
        for experiment in screen:
            print(" ".join(training_command(cli, 0, str(experiment["group"]))))
        return 0
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    protocol = {"epochs": 15, "patience": 6, "imgsz": 640, "batch": 4, "seed": 0}
    completed = previous.get("completed_pairs", []) if previous.get("protocol") == protocol else []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        "version": "data_v10_grouped_feature_cv_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "stage": "fold0_screen",
        "protocol": protocol,
        "experiments": list(screen),
        "completed_pairs": list(completed),
        "current_pair": None,
        "current_process_pid": None,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "feature_pairing": contract["feature_pairing"],
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "progress": {},
        "overall": {},
    }
    update_overall(state, state["experiments"], cli.epochs, started)
    write_json_atomic(STATUS_PATH, state)
    try:
        run_experiments(cli, screen, state, started, deployment_hash)
        selected, ranking = select_candidate(PROJECT, worst_gate=cli.worst_gate)
        state["fold0_ranking"] = ranking
        state["selected_candidate"] = selected
        state["stage"] = "three_fold_confirmation" if selected else "stopped_no_eligible_candidate"
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS_PATH, state)
        if selected:
            confirmation = [
                {"fold": fold, "group": selected, "pair": f"fold{fold}_{selected}", "name": experiment_name(fold, selected)}
                for fold in (1, 2)
            ]
            run_experiments(cli, confirmation, state, started, deployment_hash)
        summary = summarize(PROJECT, mean_gate=cli.mean_gate, worst_gate=cli.worst_gate)
        write_json_atomic(SUMMARY_PATH, summary)
        if sha256(DEPLOYMENT) != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed while V10 summary was written")
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
