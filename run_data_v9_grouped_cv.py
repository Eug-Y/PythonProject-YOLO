"""Run paired three-fold H0/H1 YOLOv8 screening without opening the sealed test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
CV_ROOT = ROOT / "configs" / "generated" / "data_v9_grouped_cv"
CONTRACT_PATH = CV_ROOT / "contract.json"
FEATURE_ROOT = ROOT / "datasets" / "Data_v4_candidates"
MANIFEST = ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "data_v8_yolo11_early_fusion_groups.json"
GROUP = "c0_current_indices_early"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
STATUS_PATH = ROOT / "logs" / "v9_grouped_cv.status.json"
SUMMARY_PATH = ROOT / "logs" / "v9_grouped_cv.summary.json"
LOG_DIR = ROOT / "logs" / "v9_grouped_cv"
VARIANTS = ("h0_refined", "h1_human")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_name(fold: int, variant: str) -> str:
    short = {"h0_refined": "h0", "h1_human": "h1"}[variant]
    return f"v9_groupcv_fold{fold}_{short}_yolov8n_none_640_seed0_15ep_v1"


def data_yaml(fold: int, variant: str) -> Path:
    filename = "data_5ch.yaml" if variant == "h0_refined" else "data_human_5ch.yaml"
    return CV_ROOT / f"fold{fold}" / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--mean-gate", type=float, default=0.002, help="Required mean H1-H0 Mask mAP50-95 delta")
    parser.add_argument("--worst-gate", type=float, default=-0.005, help="Allowed worst-fold H1-H0 delta")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def training_command(
    cli: argparse.Namespace,
    fold: int,
    variant: str,
    *,
    resume: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", GROUP,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(data_yaml(fold, variant)),
        "--name", experiment_name(fold, variant),
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


def validate_inputs(folds: tuple[int, ...]) -> tuple[dict, str]:
    required = (CONTRACT_PATH, FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DEPLOYMENT)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(DEPLOYMENT) != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint hash differs from the locked baseline")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("source_splits") != ["train", "valid"]:
        raise RuntimeError("Grouped CV must use train+valid only")
    if contract.get("held_out_test_used") or contract.get("test_paths_enumerated"):
        raise RuntimeError("Grouped CV contract indicates sealed-test access")
    available_folds = {int(row["fold"]) for row in contract.get("fold_contracts", [])}
    if not folds or len(folds) != len(set(folds)) or not set(folds).issubset(available_folds):
        raise ValueError(f"Invalid folds {folds}; available={sorted(available_folds)}")
    for fold in folds:
        h0 = data_yaml(fold, "h0_refined")
        h1 = data_yaml(fold, "h1_human")
        for yaml_path in (h0, h1):
            text = yaml_path.read_text(encoding="utf-8").lower()
            if "test:" in text or "channels: 5" not in text:
                raise RuntimeError(f"Invalid train/valid-only five-channel YAML: {yaml_path}")
        h0_val = next(line for line in h0.read_text(encoding="utf-8").splitlines() if line.startswith("val:"))
        h1_val = next(line for line in h1.read_text(encoding="utf-8").splitlines() if line.startswith("val:"))
        if h0_val != h1_val:
            raise RuntimeError(f"Fold {fold} H0/H1 validation paths differ")
    return contract, sha256(DEPLOYMENT)


def summarize(
    project: Path,
    folds: tuple[int, ...] = (0, 1, 2),
    *,
    mean_gate: float = 0.002,
    worst_gate: float = -0.005,
    deployment: Path | None = DEPLOYMENT,
) -> dict:
    rows: list[dict[str, object]] = []
    for fold in folds:
        metrics: dict[str, dict | None] = {}
        checkpoints: dict[str, str | None] = {}
        for variant in VARIANTS:
            run_dir = project / experiment_name(fold, variant)
            progress = read_progress(run_dir / "results.csv")
            checkpoint = run_dir / "weights" / "best.pt"
            metrics[variant] = progress if progress is not None and checkpoint.is_file() else None
            checkpoints[variant] = str(checkpoint.resolve()) if checkpoint.is_file() else None
        row: dict[str, object] = {
            "fold": fold,
            "h0_run": experiment_name(fold, "h0_refined"),
            "h1_run": experiment_name(fold, "h1_human"),
            "h0_checkpoint": checkpoints["h0_refined"],
            "h1_checkpoint": checkpoints["h1_human"],
        }
        if metrics["h0_refined"] is None or metrics["h1_human"] is None:
            row["status"] = "incomplete"
        else:
            h0 = metrics["h0_refined"]
            h1 = metrics["h1_human"]
            delta = float(h1["best_mask_map50_95"]) - float(h0["best_mask_map50_95"])
            row.update(
                status="complete",
                h0_mask_map50_95=float(h0["best_mask_map50_95"]),
                h1_mask_map50_95=float(h1["best_mask_map50_95"]),
                delta=delta,
                delta_pp=delta * 100.0,
                h0_metrics=h0,
                h1_metrics=h1,
            )
        rows.append(row)
    complete_rows = [row for row in rows if row["status"] == "complete"]
    all_complete = len(complete_rows) == len(folds)
    deltas = [float(row["delta"]) for row in complete_rows]
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
        "version": "data_v9_grouped_cv_summary_v1",
        "created_at": now_iso(),
        "protocol": {
            "model": "yolov8n-seg.pt",
            "channels": ["R", "G", "B", "NDVI", "NDRE"],
            "attention": "none",
            "imgsz": 640,
            "epochs": 15,
            "seed": 0,
            "validation_labels": "human_only",
            "comparison": "H1 human training labels minus H0 teacher-refined/mixed training labels",
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "rows": rows,
        "all_complete": all_complete,
        "mean_delta": mean_delta,
        "mean_delta_pp": mean_delta * 100.0 if mean_delta is not None else None,
        "worst_fold_delta": worst_delta,
        "worst_fold_delta_pp": worst_delta * 100.0 if worst_delta is not None else None,
        "mean_gate": mean_gate,
        "worst_gate": worst_gate,
        "promote_h1_to_yolo11_seed0": promoted,
        "decision": "promote_h1_to_yolo11_seed0" if promoted else ("retain_h0_protocol" if all_complete else "incomplete"),
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


def main() -> int:
    cli = parse_args()
    folds = tuple(cli.folds)
    if cli.epochs < 1 or cli.patience < 1 or cli.imgsz < 32 or cli.batch < 1 or cli.poll_seconds <= 0:
        raise ValueError("epochs, patience, imgsz, batch, and poll-seconds must be positive")
    locked_protocol = (15, 6, 640, 4, 0)
    requested_protocol = (cli.epochs, cli.patience, cli.imgsz, cli.batch, cli.seed)
    if requested_protocol != locked_protocol:
        raise ValueError(
            "Data_v9 paired screening is locked to epochs=15, patience=6, imgsz=640, batch=4, seed=0; "
            f"got {requested_protocol}"
        )
    contract, deployment_hash = validate_inputs(folds)
    experiments = [
        {"fold": fold, "variant": variant, "pair": f"fold{fold}_{variant}", "name": experiment_name(fold, variant)}
        for fold in folds
        for variant in VARIANTS
    ]
    if cli.dry_run:
        for experiment in experiments:
            print(" ".join(training_command(cli, int(experiment["fold"]), str(experiment["variant"]))))
        return 0

    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    expected_protocol = {
        "folds": list(folds),
        "epochs": cli.epochs,
        "patience": cli.patience,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "seed": cli.seed,
    }
    previous_completed = set(previous.get("completed_pairs", [])) if previous.get("protocol") == expected_protocol else set()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        "version": "data_v9_grouped_cv_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": expected_protocol,
        "experiments": experiments,
        "completed_pairs": [],
        "current_pair": None,
        "current_process_pid": None,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "validation_label_policy": contract["validation_label_policy"],
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "progress": {},
        "overall": {},
    }
    update_overall(state, experiments, cli.epochs, started)
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        for experiment in experiments:
            fold = int(experiment["fold"])
            variant = str(experiment["variant"])
            pair = str(experiment["pair"])
            run_dir = PROJECT / str(experiment["name"])
            results_path = run_dir / "results.csv"
            best_path = run_dir / "weights" / "best.pt"
            progress = read_progress(results_path)
            if pair in previous_completed and progress and best_path.is_file():
                state["completed_pairs"].append(pair)
                update_overall(state, experiments, cli.epochs, started)
                write_json_atomic(STATUS_PATH, state)
                continue
            resume = None
            last_path = run_dir / "weights" / "last.pt"
            if progress and last_path.is_file():
                resume = last_path
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
            command = training_command(cli, fold, variant, resume=resume)
            log_path = LOG_DIR / f"{experiment['name']}.log"
            state["current_pair"] = pair
            state["current_log"] = str(log_path.resolve())
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                log_handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                state["current_process_pid"] = process.pid
                write_json_atomic(STATUS_PATH, state)
                while process.poll() is None:
                    state["updated_at"] = now_iso()
                    update_overall(state, experiments, cli.epochs, started)
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
            progress = read_progress(results_path)
            if process.returncode != 0:
                raise RuntimeError(f"Run failed with exit code {process.returncode}: {experiment['name']}; see {log_path}")
            if not progress or not best_path.is_file():
                raise RuntimeError(f"Run completed without readable metrics/checkpoint: {experiment['name']}")
            if sha256(DEPLOYMENT) != deployment_hash:
                raise RuntimeError("Deployment checkpoint changed during grouped CV")
            state["completed_pairs"].append(pair)
            state["current_pair"] = None
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            update_overall(state, experiments, cli.epochs, started)
            write_json_atomic(STATUS_PATH, state)

        summary = summarize(PROJECT, folds, mean_gate=cli.mean_gate, worst_gate=cli.worst_gate)
        write_json_atomic(SUMMARY_PATH, summary)
        if sha256(DEPLOYMENT) != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed while grouped CV summary was written")
        state["state"] = "completed"
        state["summary"] = str(SUMMARY_PATH.resolve())
        state["decision"] = summary["decision"]
        state["deployment_sha256_after"] = deployment_hash
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        update_overall(state, experiments, cli.epochs, started)
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
        update_overall(state, experiments, cli.epochs, started)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
