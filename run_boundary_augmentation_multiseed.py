"""Screen boundary-preserving augmentation and confirm it across seeds only if warranted."""

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
STATUS_PATH = ROOT / "logs" / "data_v8_boundary_augmentation_multiseed.status.json"
LOG_DIR = ROOT / "logs" / "data_v8_boundary_augmentation_multiseed"
SCREEN_GAIN = 0.002
WORST_SEED_LIMIT = -0.005


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_name(seed: int) -> str:
    return f"v8_h0_shuffle_30ep_seed{seed}_v1"


def candidate_name(seed: int) -> str:
    return f"v8_b1_boundary_aug_30ep_seed{seed}_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def validate_inputs(seeds: tuple[int, ...]) -> str:
    for path in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DATA, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    if "test:" in DATA.read_text(encoding="utf-8").lower():
        raise RuntimeError(f"Training YAML must not expose test: {DATA}")
    for seed in seeds:
        run_dir = PROJECT / baseline_name(seed)
        progress = read_progress(run_dir / "results.csv")
        if not progress or progress["epochs_recorded"] < 30 or not (run_dir / "weights" / "best.pt").is_file():
            raise RuntimeError(f"Missing complete paired baseline for seed {seed}: {run_dir}")
    return sha256(DEPLOYMENT)


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
        "--epochs", "30",
        "--patience", "12",
        "--imgsz", "1024",
        "--batch", "2",
        "--device", device,
        "--workers", "2",
        "--seed", str(seed),
        "--cache",
        "--deterministic",
        "--mosaic", "0.5",
        "--mixup", "0.0",
        "--close-mosaic", "15",
        "--name", candidate_name(seed),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def update_pairs(state: dict, seeds: tuple[int, ...]) -> None:
    pairs: list[dict[str, object]] = []
    for seed in seeds:
        baseline = read_progress(PROJECT / baseline_name(seed) / "results.csv")
        candidate = state["progress"].get(candidate_name(seed))
        if not baseline or not candidate:
            continue
        baseline_metric = float(baseline["best_mask_map50_95"])
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


def update_overall(state: dict, seeds: tuple[int, ...]) -> None:
    recorded = sum(min(int(state["progress"].get(candidate_name(seed), {}).get("epochs_recorded", 0)), 30) for seed in seeds)
    state["overall"] = {
        "epochs_recorded": recorded,
        "target_epochs": 30 * len(seeds),
        "percent": round(100.0 * recorded / (30 * len(seeds)), 2),
    }


def main() -> int:
    cli = parse_args()
    seeds = tuple(cli.seeds)
    if seeds != (0, 1, 2):
        raise ValueError("Boundary augmentation confirmation is locked to seeds 0, 1, 2")
    deployment_hash = validate_inputs(seeds)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    state: dict[str, object] = {
        "version": "data_v8_boundary_augmentation_multiseed_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": started,
        "updated_at": started,
        "seeds": list(seeds),
        "epochs_per_run": 30,
        "architecture": "YOLO11n-seg",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "attention": "3xCoordAtt+3xECA",
        "candidate": {"mosaic": 0.5, "mixup": 0.0, "close_mosaic": 15},
        "baseline": {"mosaic": 1.0, "mixup": 0.1, "close_mosaic": 10},
        "seed0_gain_threshold": SCREEN_GAIN,
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
        for index, seed in enumerate(seeds):
            if index > 0:
                seed0_delta = float(state["paired_results"][0]["delta"])
                if seed0_delta < SCREEN_GAIN:
                    state["decision"] = "rejected_after_seed0"
                    break
            name = candidate_name(seed)
            run_dir = PROJECT / name
            results_path = run_dir / "results.csv"
            best_path = run_dir / "weights" / "best.pt"
            progress = read_progress(results_path)
            if progress and progress["epochs_recorded"] >= 30 and best_path.is_file():
                state["progress"][name] = progress
                state["completed_runs"].append(name)
                update_pairs(state, seeds)
                update_overall(state, seeds)
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
                    update_pairs(state, seeds)
                    update_overall(state, seeds)
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
            progress = read_progress(results_path)
            if process.returncode != 0:
                raise RuntimeError(f"Run failed with exit code {process.returncode}: {name}; see {log_path}")
            if not progress or progress["epochs_recorded"] < 30 or not best_path.is_file():
                raise RuntimeError(f"Run completed without full metrics/checkpoint: {name}")
            state["progress"][name] = progress
            state["completed_runs"].append(name)
            state["current_run"] = None
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            update_pairs(state, seeds)
            update_overall(state, seeds)
            write_json_atomic(STATUS_PATH, state)

        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during research training")
        pairs = state["paired_results"]
        if len(pairs) == len(seeds):
            deltas = [float(pair["delta"]) for pair in pairs]
            mean_delta = sum(deltas) / len(deltas)
            state["gate"] = {
                "complete": True,
                "mean_delta": mean_delta,
                "mean_delta_pp": 100.0 * mean_delta,
                "worst_delta": min(deltas),
                "worst_delta_pp": 100.0 * min(deltas),
                "mean_gain_threshold": SCREEN_GAIN,
                "worst_seed_threshold": WORST_SEED_LIMIT,
                "passed": mean_delta >= SCREEN_GAIN and min(deltas) >= WORST_SEED_LIMIT,
            }
            state["decision"] = "three_seed_gate_complete"
        else:
            state["gate"] = {
                "complete": False,
                "passed": False,
                "reason": "seed0_gain_below_threshold",
            }
        state["state"] = "completed"
        state["current_run"] = None
        state["current_process_pid"] = None
        state["deployment_sha256_after"] = deployment_hash_after
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        update_overall(state, seeds)
        write_json_atomic(STATUS_PATH, state)
        return 0
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        state["deployment_sha256_after"] = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["deployment_unchanged"] = state["deployment_sha256_after"] == deployment_hash
        update_pairs(state, seeds)
        update_overall(state, seeds)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
