"""Confirm separate overlap-mask supervision on YOLO11 with staged paired seeds."""

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
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
FEATURE_ROOT = ROOT / "datasets" / "Data_v4_candidates"
MANIFEST = ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "data_v8_yolo11_early_fusion_groups.json"
DATA = ROOT / "configs" / "data_v8_fusion_selection_5ch.yaml"
BASE_MODEL = ROOT / "yolo11n-seg.pt"
GROUP = "c0_current_indices_early"
STATUS_PATH = ROOT / "logs" / "v13_yolo11_overlap_multiseed.status.json"
SUMMARY_PATH = ROOT / "logs" / "v13_yolo11_overlap_multiseed.summary.json"
LOG_DIR = ROOT / "logs" / "v13_yolo11_overlap_multiseed"
SEEDS = (0, 1, 2)


def control_name(seed: int) -> str:
    return f"v8_yolo11_c0_current_indices_early_ca_eca_1024_seed{seed}_confirm_v1"


def candidate_name(seed: int) -> str:
    return f"v13_yolo11_c0_nooverlap_ca_eca_1024_seed{seed}_confirm_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--seed0-gate", type=float, default=0.002)
    parser.add_argument("--mean-gate", type=float, default=0.003)
    parser.add_argument("--worst-gate", type=float, default=-0.003)
    parser.add_argument("--dry-run", action="store_true")
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


def validate_inputs() -> tuple[dict, str, str]:
    for path in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DATA, BASE_MODEL, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("held_out_split_used") is not False or manifest.get("selection_splits") != ["train", "valid"]:
        raise RuntimeError("Feature manifest does not preserve the sealed-test contract")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text or "channels: 5" not in data_text:
        raise RuntimeError(f"Invalid train/valid-only five-channel YAML: {DATA}")
    groups = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))["groups"]
    if groups.get(GROUP) != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError("YOLO11 control channel contract changed")
    expected_args = (
        "model: yolo11n-seg.pt",
        "epochs: 60",
        "patience: 20",
        "batch: 2",
        "imgsz: 1024",
        "deterministic: true",
        "overlap_mask: true",
        "mask_ratio: 4",
    )
    for seed in SEEDS:
        run_dir = PROJECT / control_name(seed)
        args_path = run_dir / "args.yaml"
        progress = read_progress(run_dir / "results.csv")
        if progress is None or not (run_dir / "weights" / "best.pt").is_file() or not args_path.is_file():
            raise RuntimeError(f"Missing complete YOLO11 paired control for seed {seed}: {run_dir}")
        args_text = args_path.read_text(encoding="utf-8").lower()
        for token in expected_args:
            if token not in args_text:
                raise RuntimeError(f"Seed {seed} control lacks locked argument {token!r}")
        if f"seed: {seed}" not in args_text:
            raise RuntimeError(f"Seed {seed} control seed contract changed")
        if "test:" in args_text:
            raise RuntimeError(f"Seed {seed} control args expose test")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint hash differs from the locked baseline")
    return manifest, sha256(BASE_MODEL), deployment_hash


def pair_row(project: Path, seed: int) -> dict[str, object]:
    control_run = project / control_name(seed)
    candidate_run = project / candidate_name(seed)
    control = read_progress(control_run / "results.csv")
    candidate = read_progress(candidate_run / "results.csv")
    row: dict[str, object] = {
        "seed": seed,
        "control_run": control_name(seed),
        "candidate_run": candidate_name(seed),
    }
    if control is None or candidate is None or not (control_run / "weights" / "best.pt").is_file() or not (candidate_run / "weights" / "best.pt").is_file():
        row["status"] = "incomplete"
        return row
    map_delta = float(candidate["best_mask_map50_95"]) - float(control["best_mask_map50_95"])
    precision_delta = float(candidate["best_mask_precision"]) - float(control["best_mask_precision"])
    recall_delta = float(candidate["best_mask_recall"]) - float(control["best_mask_recall"])
    row.update(
        status="complete",
        control_mask_map50_95=float(control["best_mask_map50_95"]),
        candidate_mask_map50_95=float(candidate["best_mask_map50_95"]),
        delta=map_delta,
        delta_pp=map_delta * 100.0,
        precision_delta=precision_delta,
        recall_delta=recall_delta,
        control_metrics=control,
        candidate_metrics=candidate,
        candidate_checkpoint=str((candidate_run / "weights" / "best.pt").resolve()),
        candidate_checkpoint_sha256=sha256(candidate_run / "weights" / "best.pt"),
    )
    return row


def summarize(
    project: Path,
    *,
    seed0_gate: float = 0.002,
    mean_gate: float = 0.003,
    worst_gate: float = -0.003,
    deployment: Path | None = DEPLOYMENT,
) -> dict:
    rows = [pair_row(project, seed) for seed in SEEDS]
    seed0 = rows[0]
    seed0_passed = bool(
        seed0["status"] == "complete"
        and float(seed0["delta"]) >= seed0_gate
        and float(seed0["precision_delta"]) >= -0.01
        and float(seed0["recall_delta"]) >= -0.01
    )
    complete = [row for row in rows if row["status"] == "complete"]
    all_complete = len(complete) == 3
    map_deltas = [float(row["delta"]) for row in complete]
    precision_deltas = [float(row["precision_delta"]) for row in complete]
    recall_deltas = [float(row["recall_delta"]) for row in complete]
    mean_delta = statistics.fmean(map_deltas) if map_deltas else None
    worst_delta = min(map_deltas) if map_deltas else None
    positive_seeds = sum(delta > 0 for delta in map_deltas)
    mean_precision_delta = statistics.fmean(precision_deltas) if precision_deltas else None
    mean_recall_delta = statistics.fmean(recall_deltas) if recall_deltas else None
    gate_checks = {
        "seed0_passed": seed0_passed,
        "all_three_pairs_complete": all_complete,
        "mean_delta_at_least_0_003": bool(all_complete and mean_delta is not None and mean_delta >= mean_gate),
        "at_least_two_positive_seeds": bool(all_complete and positive_seeds >= 2),
        "worst_seed_not_below_minus_0_003": bool(all_complete and worst_delta is not None and worst_delta >= worst_gate),
        "mean_precision_regression_within_0_01": bool(all_complete and mean_precision_delta is not None and mean_precision_delta >= -0.01),
        "mean_recall_regression_within_0_01": bool(all_complete and mean_recall_delta is not None and mean_recall_delta >= -0.01),
    }
    final_passed = all(gate_checks.values())
    if seed0["status"] != "complete":
        decision = "incomplete_seed0"
    elif not seed0_passed:
        decision = "reject_seed0_retain_current_deployment"
    elif not all_complete:
        decision = "incomplete_multiseed"
    elif final_passed:
        decision = "freeze_nooverlap_candidate_for_review_no_auto_deployment"
    else:
        decision = "reject_multiseed_retain_current_deployment"
    deployment_hash = sha256(deployment) if deployment is not None and deployment.is_file() else None
    best = max(complete, key=lambda row: float(row["candidate_mask_map50_95"])) if complete else None
    return {
        "version": "data_v13_yolo11_overlap_multiseed_summary_v1",
        "created_at": now_iso(),
        "experiment": {
            "architecture": "YOLO11n-seg",
            "attention": "3x CoordAtt + 3x ECA",
            "channels": ["R", "G", "B", "NDVI", "NDRE"],
            "control_overlap_mask": True,
            "candidate_overlap_mask": False,
            "mask_ratio": 4,
            "epochs": 60,
            "patience": 20,
            "imgsz": 1024,
            "batch": 2,
            "paired_initialization": True,
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "paired_results": rows,
        "seed0_gate": seed0_gate,
        "seed0_passed": seed0_passed,
        "all_three_pairs_complete": all_complete,
        "mean_delta": mean_delta,
        "mean_delta_pp": mean_delta * 100.0 if mean_delta is not None else None,
        "worst_seed_delta": worst_delta,
        "worst_seed_delta_pp": worst_delta * 100.0 if worst_delta is not None else None,
        "positive_seeds": positive_seeds,
        "mean_precision_delta": mean_precision_delta,
        "mean_recall_delta": mean_recall_delta,
        "gate_thresholds": {"mean_delta_min": mean_gate, "worst_seed_delta_min": worst_gate, "mean_precision_delta_min": -0.01, "mean_recall_delta_min": -0.01},
        "gate_checks": gate_checks,
        "gate_passed": final_passed,
        "decision": decision,
        "candidate_checkpoint": best["candidate_checkpoint"] if final_passed and best else None,
        "deployment_eligible": False,
        "deployment_write_allowed": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256 if deployment_hash is not None else None,
    }


def update_overall(state: dict, started: float) -> None:
    recorded = 0
    progress: dict[str, dict] = {}
    planned = [int(seed) for seed in state["planned_seeds"]]
    for seed in planned:
        item = read_progress(PROJECT / candidate_name(seed) / "results.csv")
        if item:
            progress[str(seed)] = item
            recorded += min(int(item["epochs_recorded"]), 60)
    target = len(planned) * 60
    elapsed = max(time.monotonic() - started, 0.0)
    eta = elapsed / recorded * max(target - recorded, 0) if recorded else None
    state["progress"] = progress
    state["overall"] = {
        "epochs_recorded": recorded,
        "target_epochs": target,
        "percent": round(recorded / max(target, 1) * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }
    state["updated_at"] = now_iso()
    write_json_atomic(STATUS_PATH, state)


def run_seeds(cli: argparse.Namespace, seeds: tuple[int, ...], state: dict, started: float, deployment_hash: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    for seed in seeds:
        if seed not in state["planned_seeds"]:
            state["planned_seeds"].append(seed)
        run_dir = PROJECT / candidate_name(seed)
        progress = read_progress(run_dir / "results.csv")
        best_path = run_dir / "weights" / "best.pt"
        if (seed in state["completed_seeds"] or (progress and int(progress["epochs_recorded"]) >= 60)) and best_path.is_file():
            if seed not in state["completed_seeds"]:
                state["completed_seeds"].append(seed)
            update_overall(state, started)
            continue
        resume = None
        last_path = run_dir / "weights" / "last.pt"
        if progress and last_path.is_file():
            resume = last_path
        elif run_dir.exists():
            raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
        command = training_command(cli, seed, resume=resume)
        log_path = LOG_DIR / f"{candidate_name(seed)}.log"
        state["current_seed"] = seed
        state["current_log"] = str(log_path.resolve())
        write_json_atomic(STATUS_PATH, state)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
            state["current_process_pid"] = process.pid
            write_json_atomic(STATUS_PATH, state)
            while process.poll() is None:
                update_overall(state, started)
                time.sleep(max(2.0, cli.poll_seconds))
        progress = read_progress(run_dir / "results.csv")
        if process.returncode != 0:
            raise RuntimeError(f"{candidate_name(seed)} exited with code {process.returncode}; see {log_path}")
        if not progress or not best_path.is_file():
            raise RuntimeError(f"Completed run lacks metrics/checkpoint: {run_dir}")
        if sha256(DEPLOYMENT) != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V13 confirmation")
        state["completed_seeds"].append(seed)
        state["current_seed"] = None
        state["current_process_pid"] = None
        update_overall(state, started)


def main() -> int:
    cli = parse_args()
    if (cli.epochs, cli.patience, cli.imgsz, cli.batch) != (60, 20, 1024, 2) or cli.workers < 0:
        raise ValueError("V13 is locked to epochs=60, patience=20, imgsz=1024, batch=2")
    manifest, base_hash, deployment_hash = validate_inputs()
    if cli.dry_run:
        for seed in SEEDS:
            print(subprocess.list2cmdline(training_command(cli, seed)))
        return 0
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    protocol = {"epochs": 60, "patience": 20, "imgsz": 1024, "batch": 2, "attention": "ca_eca", "mask_ratio": 4, "overlap_mask": False}
    completed = previous.get("completed_seeds", []) if previous.get("protocol") == protocol else []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        "version": "data_v13_yolo11_overlap_multiseed_status_v1",
        "state": "running",
        "stage": "seed0_confirmation",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol,
        "planned_seeds": [0],
        "completed_seeds": list(completed),
        "current_seed": None,
        "current_process_pid": None,
        "base_model": str(BASE_MODEL.resolve()),
        "base_model_sha256": base_hash,
        "paired_controls": {str(seed): control_name(seed) for seed in SEEDS},
        "selection_splits": manifest["selection_splits"],
        "held_out_test_used": False,
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "progress": {},
        "overall": {},
    }
    update_overall(state, started)
    try:
        run_seeds(cli, (0,), state, started, deployment_hash)
        interim = summarize(PROJECT, seed0_gate=cli.seed0_gate, mean_gate=cli.mean_gate, worst_gate=cli.worst_gate)
        state["seed0_result"] = interim["paired_results"][0]
        state["seed0_passed"] = interim["seed0_passed"]
        state["stage"] = "multiseed_confirmation" if interim["seed0_passed"] else "stopped_seed0_gate"
        write_json_atomic(STATUS_PATH, state)
        if interim["seed0_passed"]:
            run_seeds(cli, (1, 2), state, started, deployment_hash)
        summary = summarize(PROJECT, seed0_gate=cli.seed0_gate, mean_gate=cli.mean_gate, worst_gate=cli.worst_gate)
        write_json_atomic(SUMMARY_PATH, summary)
        if sha256(DEPLOYMENT) != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed while V13 summary was written")
        state["state"] = "completed"
        state["stage"] = "completed"
        state["decision"] = summary["decision"]
        state["summary"] = str(SUMMARY_PATH.resolve())
        state["current_seed"] = None
        state["current_process_pid"] = None
        state["deployment_sha256_after"] = deployment_hash
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        update_overall(state, started)
        return 0
    except BaseException as exc:
        current_hash = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["deployment_sha256_after"] = current_hash
        state["deployment_unchanged"] = current_hash == deployment_hash
        update_overall(state, started)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
