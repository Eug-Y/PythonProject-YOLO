"""Run Data_v5 seed 1/2 validation for RGB, NDVI_STD5 and MCARI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
FEATURE_ROOT = ROOT / "datasets" / "Data_v5_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "feature_ablation_groups_v5.json"
LOG_DIR = ROOT / "logs" / "v5_multiseed"
STATUS_PATH = ROOT / "logs" / "v5_multiseed.status.json"
GROUPS = ("rgb", "add_ndvi_std5", "add_mcari")
SEEDS = (1, 2)
TASKS = tuple((group, seed) for seed in SEEDS for group in GROUPS)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def task_key(group: str, seed: int) -> str:
    return f"{group}_seed{seed}"


def experiment_name(group: str, seed: int) -> str:
    return f"v5_multiseed_{group}_n_seed{seed}_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def update_overall(state: dict, started_monotonic: float) -> None:
    epochs_per_run = int(state["epochs_per_run"])
    total = len(state["tasks"]) * epochs_per_run
    recorded = sum(min(int(item.get("epochs_recorded", 0)), epochs_per_run) for item in state["progress"].values())
    elapsed = max(time.monotonic() - started_monotonic, 0.0)
    eta = elapsed / recorded * (total - recorded) if recorded else None
    state["overall"] = {
        "epochs_recorded": recorded,
        "total_epochs": total,
        "percent": round(recorded / max(total, 1) * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }


def main() -> None:
    cli = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for required in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG):
        if not required.exists():
            raise FileNotFoundError(required)

    started_monotonic = time.monotonic()
    started = now_iso()
    state = {
        "version": "data_v5_multiseed_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": started,
        "updated_at": started,
        "tasks": [{"group": group, "seed": seed} for group, seed in TASKS],
        "completed_tasks": [],
        "current_task": None,
        "current_process_pid": None,
        "epochs_per_run": cli.epochs,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "progress": {},
        "overall": {},
    }
    update_overall(state, started_monotonic)
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        for group, seed in TASKS:
            key = task_key(group, seed)
            run_dir = PROJECT / experiment_name(group, seed)
            results_path = run_dir / "results.csv"
            progress = read_progress(results_path)
            if progress and progress["last_epoch"] >= cli.epochs and (run_dir / "weights" / "best.pt").is_file():
                state["completed_tasks"].append(key)
                state["progress"][key] = progress
                state["updated_at"] = now_iso()
                update_overall(state, started_monotonic)
                write_json_atomic(STATUS_PATH, state)
                continue

            command = [
                sys.executable,
                str(ROOT / "train_feature_ablation.py"),
                "--group", group,
                "--groups-config", str(GROUPS_CONFIG),
                "--manifest", str(MANIFEST),
                "--feature-root", str(FEATURE_ROOT),
                "--name", experiment_name(group, seed),
                "--epochs", str(cli.epochs),
                "--imgsz", str(cli.imgsz),
                "--batch", str(cli.batch),
                "--device", cli.device,
                "--seed", str(seed),
            ]
            last_checkpoint = run_dir / "weights" / "last.pt"
            if progress and progress["last_epoch"] < cli.epochs and last_checkpoint.is_file():
                command.extend(("--resume", str(last_checkpoint)))
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")

            log_path = LOG_DIR / f"{key}.log"
            state["current_task"] = {"group": group, "seed": seed, "key": key}
            state["current_log"] = str(log_path)
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
                        state["progress"][key] = progress
                    state["updated_at"] = now_iso()
                    update_overall(state, started_monotonic)
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
                return_code = process.returncode

            progress = read_progress(results_path)
            if progress:
                state["progress"][key] = progress
            if return_code != 0:
                raise RuntimeError(f"Task {key} failed with exit code {return_code}; see {log_path}")
            if not progress or progress["last_epoch"] < cli.epochs:
                raise RuntimeError(f"Task {key} exited without {cli.epochs} recorded epochs")
            state["completed_tasks"].append(key)
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            update_overall(state, started_monotonic)
            write_json_atomic(STATUS_PATH, state)

        state["state"] = "completed"
        state["current_task"] = None
        state["current_process_pid"] = None
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        update_overall(state, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        subprocess.run([sys.executable, str(ROOT / "summarize_data_v5_multiseed.py")], cwd=ROOT, check=True, env=environment)
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        update_overall(state, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    main()
