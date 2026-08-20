"""Run the locked 60-epoch YOLO11 B1 confirmation against existing C0 baselines."""

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
FEATURE_ROOT = ROOT / "datasets" / "Data_v4_candidates"
MANIFEST = ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "data_v8_yolo11_early_fusion_groups.json"
DATA = ROOT / "configs" / "data_v8_fusion_selection_5ch.yaml"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
STATUS_PATH = ROOT / "logs" / "data_v8_boundary_augmentation_long_multiseed.status.json"
LOG_DIR = ROOT / "logs" / "data_v8_boundary_augmentation_long_multiseed"
SEEDS = (0, 1, 2)
EPOCHS = 60
MEAN_GAIN_THRESHOLD = 0.002
WORST_SEED_THRESHOLD = -0.005


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_name(seed: int) -> str:
    return f"v8_yolo11_c0_current_indices_early_ca_eca_1024_seed{seed}_confirm_v1"


def candidate_name(seed: int) -> str:
    return f"v8_b1_boundary_aug_60ep_seed{seed}_confirm_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def validate_inputs() -> tuple[str, dict[int, dict]]:
    for path in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DATA, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    if "test:" in DATA.read_text(encoding="utf-8").lower():
        raise RuntimeError(f"Training YAML must not expose test: {DATA}")
    baselines: dict[int, dict] = {}
    for seed in SEEDS:
        run_dir = PROJECT / baseline_name(seed)
        progress = read_progress(run_dir / "results.csv")
        args_path = run_dir / "args.yaml"
        best_path = run_dir / "weights" / "best.pt"
        if not progress or not args_path.is_file() or not best_path.is_file():
            raise RuntimeError(f"Missing complete 60-epoch baseline for seed {seed}: {run_dir}")
        args_text = args_path.read_text(encoding="utf-8")
        required = ("epochs: 60", "patience: 20", "batch: 2", "imgsz: 1024", "warmup_epochs: 1.0", "cache: false")
        missing = [line for line in required if line not in args_text]
        if missing:
            raise RuntimeError(f"Baseline budget mismatch for seed {seed}: missing {missing}")
        baselines[seed] = progress
    return sha256(DEPLOYMENT), baselines


def training_command(device: str, seed: int, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", "c0_current_indices_early",
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(DATA),
        "--architecture", "yolo11",
        "--model-size", "n",
        "--attention", "ca_eca",
        "--extra-init", "mean",
        "--epochs", str(EPOCHS),
        "--patience", "20",
        "--warmup-epochs", "1",
        "--imgsz", "1024",
        "--batch", "2",
        "--device", device,
        "--workers", "2",
        "--seed", str(seed),
        "--no-cache",
        "--deterministic",
        "--mosaic", "0.5",
        "--mixup", "0.0",
        "--close-mosaic", "15",
        "--name", candidate_name(seed),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def update_pairs(state: dict, baselines: dict[int, dict]) -> None:
    pairs: list[dict[str, object]] = []
    for seed in SEEDS:
        candidate = state["progress"].get(candidate_name(seed))
        if not candidate:
            continue
        baseline_metric = float(baselines[seed]["best_mask_map50_95"])
        candidate_metric = float(candidate["best_mask_map50_95"])
        pairs.append(
            {
                "seed": seed,
                "baseline_best_mask_map50_95": baseline_metric,
                "candidate_best_mask_map50_95": candidate_metric,
                "delta": candidate_metric - baseline_metric,
                "delta_pp": 100.0 * (candidate_metric - baseline_metric),
            }
        )
    state["paired_results"] = pairs


def update_overall(state: dict) -> None:
    recorded = sum(min(int(state["progress"].get(candidate_name(seed), {}).get("epochs_recorded", 0)), EPOCHS) for seed in SEEDS)
    state["overall"] = {
        "epochs_recorded": recorded,
        "target_epochs": EPOCHS * len(SEEDS),
        "percent": round(100.0 * recorded / (EPOCHS * len(SEEDS)), 2),
    }


def main() -> int:
    cli = parse_args()
    deployment_hash, baselines = validate_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    previously_completed = set(previous.get("completed_runs", []))
    started = now_iso()
    state: dict[str, object] = {
        "version": "data_v8_boundary_augmentation_long_multiseed_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": started,
        "updated_at": started,
        "seeds": list(SEEDS),
        "epochs_per_run": EPOCHS,
        "patience": 20,
        "imgsz": 1024,
        "batch": 2,
        "warmup_epochs": 1.0,
        "cache": False,
        "architecture": "YOLO11n-seg",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "attention": "3xCoordAtt+3xECA",
        "candidate": {"mosaic": 0.5, "mixup": 0.0, "close_mosaic": 15},
        "baseline": {"mosaic": 1.0, "mixup": 0.1, "close_mosaic": 10},
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "completed_runs": [],
        "current_run": None,
        "current_process_pid": None,
        "progress": {},
        "paired_results": [],
        "overall": {},
        "report_generated": False,
    }
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    write_json_atomic(STATUS_PATH, state)

    try:
        for seed in SEEDS:
            name = candidate_name(seed)
            run_dir = PROJECT / name
            results_path = run_dir / "results.csv"
            best_path = run_dir / "weights" / "best.pt"
            progress = read_progress(results_path)
            if name in previously_completed and progress and best_path.is_file():
                state["progress"][name] = progress
                state["completed_runs"].append(name)
                update_pairs(state, baselines)
                update_overall(state)
                write_json_atomic(STATUS_PATH, state)
                continue

            resume = None
            last_path = run_dir / "weights" / "last.pt"
            if progress and last_path.is_file():
                resume = last_path
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
            command = training_command(cli.device, seed, resume)
            log_path = LOG_DIR / f"{name}.log"
            state["current_run"] = name
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
                    progress = read_progress(results_path)
                    if progress:
                        state["progress"][name] = progress
                    state["updated_at"] = now_iso()
                    update_pairs(state, baselines)
                    update_overall(state)
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
            progress = read_progress(results_path)
            if process.returncode != 0:
                raise RuntimeError(f"Run failed with exit code {process.returncode}: {name}; see {log_path}")
            if not progress or not best_path.is_file():
                raise RuntimeError(f"Run completed without readable metrics/checkpoint: {name}")
            state["progress"][name] = progress
            state["completed_runs"].append(name)
            state["current_run"] = None
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            update_pairs(state, baselines)
            update_overall(state)
            write_json_atomic(STATUS_PATH, state)

        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during research training")
        deltas = [float(pair["delta"]) for pair in state["paired_results"]]
        if len(deltas) != len(SEEDS):
            raise RuntimeError("Long confirmation ended without all three paired results")
        mean_delta = sum(deltas) / len(deltas)
        state["gate"] = {
            "complete": True,
            "mean_delta": mean_delta,
            "mean_delta_pp": 100.0 * mean_delta,
            "worst_delta": min(deltas),
            "worst_delta_pp": 100.0 * min(deltas),
            "mean_gain_threshold": MEAN_GAIN_THRESHOLD,
            "worst_seed_threshold": WORST_SEED_THRESHOLD,
            "passed": mean_delta >= MEAN_GAIN_THRESHOLD and min(deltas) >= WORST_SEED_THRESHOLD,
            "deployment_action": "none_until_one_time_test",
        }
        state["state"] = "completed"
        state["current_run"] = None
        state["current_process_pid"] = None
        state["deployment_sha256_after"] = deployment_hash_after
        state["deployment_unchanged"] = True
        state["candidate_locked_if_gate_passes"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        update_overall(state)
        write_json_atomic(STATUS_PATH, state)
        return 0
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        state["deployment_sha256_after"] = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["deployment_unchanged"] = state["deployment_sha256_after"] == deployment_hash
        update_pairs(state, baselines)
        update_overall(state)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
