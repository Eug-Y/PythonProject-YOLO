"""Wait for the Data_v7 screen, then run seed 1/2 for its promoted candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_feature_ablations import read_progress, write_json_atomic
from run_data_v7_soil_vegetation_screen import (
    DATA_BY_CHANNELS,
    DEPLOYMENT,
    DEPLOYMENT_SHA256,
    FEATURE_ROOT,
    GROUPS,
    GROUPS_CONFIG,
    INITIALIZATION,
    MANIFEST,
    PROJECT,
    experiment_name,
    sha256,
    training_command,
)


ROOT = Path(__file__).resolve().parent
SCREEN_STATUS = ROOT / "logs" / "v7_soil_vegetation_screen.status.json"
SCREEN_SUMMARY = ROOT / "reports" / "Data_v7_soil_vegetation_screen" / "summary.json"
STATUS_PATH = ROOT / "logs" / "v7_soil_vegetation_multiseed.status.json"
LOG_DIR = ROOT / "logs" / "v7_soil_vegetation_multiseed"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_candidate(summary_path: Path = SCREEN_SUMMARY) -> str | None:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Data_v7 screen summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("complete_groups") != summary.get("expected_groups"):
        raise RuntimeError("Data_v7 screen summary is incomplete")
    candidate = summary.get("recommended_next")
    if candidate is None:
        return None
    group = str(candidate.get("group", ""))
    if group not in GROUPS:
        raise RuntimeError(f"Screen selected an unknown candidate group: {group!r}")
    if not candidate.get("promoted"):
        raise RuntimeError("Screen recommendation does not satisfy the promotion threshold")
    return group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-group", choices=GROUPS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--wait-for-screen", action="store_true")
    parser.add_argument("--screen-poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def update_overall(state: dict, started_monotonic: float) -> None:
    target = int(state.get("epochs_per_run", 0)) * len(state.get("seeds", []))
    recorded = sum(
        min(int(progress.get("epochs_recorded", 0)), int(state.get("epochs_per_run", 0)))
        for progress in state.get("progress", {}).values()
    )
    elapsed = max(time.monotonic() - started_monotonic, 0.0)
    eta = elapsed / recorded * max(target - recorded, 0) if recorded else None
    state["overall"] = {
        "epochs_recorded": recorded,
        "target_epochs": target,
        "percent": round(recorded / max(target, 1) * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }


def wait_for_screen(state: dict, poll_seconds: float) -> None:
    while True:
        if SCREEN_STATUS.is_file():
            screen = json.loads(SCREEN_STATUS.read_text(encoding="utf-8"))
            state["screen_state"] = screen.get("state")
            state["screen_current_group"] = screen.get("current_group")
            state["screen_completed_groups"] = screen.get("completed_groups", [])
            state["screen_overall"] = screen.get("overall", {})
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)
            if screen.get("state") == "completed":
                return
            if screen.get("state") == "failed":
                raise RuntimeError(f"Data_v7 screen failed: {screen.get('error')}")
        time.sleep(max(5.0, poll_seconds))


def main() -> int:
    cli = parse_args()
    seeds = tuple(cli.seeds)
    if not seeds or len(seeds) != len(set(seeds)) or 0 in seeds:
        raise ValueError("--seeds must be unique confirmation seeds and must not include reused screen seed 0")
    if cli.epochs < 1 or cli.patience < 1 or cli.batch < 1:
        raise ValueError("--epochs, --patience and --batch must be positive")
    for path in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, INITIALIZATION, DEPLOYMENT, *DATA_BY_CHANNELS.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment SHA256 mismatch before Data_v7 multi-seed confirmation")

    started_monotonic = time.monotonic()
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    previously_completed = set(previous.get("completed_seeds", []))
    state = {
        "version": "data_v7_soil_vegetation_multiseed_status_v1",
        "state": "waiting_for_screen" if cli.wait_for_screen and cli.candidate_group is None else "preparing",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "slope_included": False,
        "teacher_student_pipeline": "YOLO11 teacher-refined labels -> YOLOv8n student",
        "seeds": list(seeds),
        "reused_screen_seed": 0,
        "epochs_per_run": cli.epochs,
        "patience": cli.patience,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "initialization_checkpoint": str(INITIALIZATION.resolve()),
        "initialization_sha256": sha256(INITIALIZATION),
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "candidate_group": cli.candidate_group,
        "completed_seeds": [],
        "current_seed": None,
        "current_process_pid": None,
        "progress": {},
        "overall": {},
    }
    update_overall(state, started_monotonic)
    write_json_atomic(STATUS_PATH, state)

    try:
        if cli.candidate_group is None and cli.wait_for_screen:
            wait_for_screen(state, cli.screen_poll_seconds)
        candidate_group = cli.candidate_group or resolve_candidate()
        if candidate_group is None:
            state["state"] = "completed_no_promoted_candidate"
            state["decision"] = "retain_current_5ch"
            state["completed_at"] = now_iso()
            state["updated_at"] = state["completed_at"]
            write_json_atomic(STATUS_PATH, state)
            return 0

        groups = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))["groups"]
        if candidate_group not in groups:
            raise ValueError(f"Unknown Data_v7 candidate group: {candidate_group}")
        channel_count = len(groups[candidate_group])
        screen_run = PROJECT / experiment_name(candidate_group, 0)
        if read_progress(screen_run / "results.csv") is None or not (screen_run / "weights" / "best.pt").is_file():
            raise RuntimeError(f"Promoted candidate has no reusable seed-0 checkpoint: {screen_run}")

        state["state"] = "running"
        state["candidate_group"] = candidate_group
        state["channels"] = groups[candidate_group]
        state["screen_seed0_run"] = experiment_name(candidate_group, 0)
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS_PATH, state)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"

        for seed in seeds:
            name = experiment_name(candidate_group, seed)
            run_dir = PROJECT / name
            results_path = run_dir / "results.csv"
            best_path = run_dir / "weights" / "best.pt"
            progress = read_progress(results_path)
            if seed in previously_completed and progress and best_path.is_file():
                state["completed_seeds"].append(seed)
                state["progress"][str(seed)] = progress
                update_overall(state, started_monotonic)
                write_json_atomic(STATUS_PATH, state)
                continue
            resume = None
            last_path = run_dir / "weights" / "last.pt"
            if progress and last_path.is_file():
                resume = last_path
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete confirmation run has no resumable checkpoint: {run_dir}")
            task_cli = argparse.Namespace(**vars(cli))
            task_cli.seed = seed
            command = training_command(task_cli, candidate_group, channel_count, resume=resume)
            log_path = LOG_DIR / f"{name}.log"
            state["current_seed"] = seed
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
                        state["progress"][str(seed)] = progress
                    state["updated_at"] = now_iso()
                    update_overall(state, started_monotonic)
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
            progress = read_progress(results_path)
            if process.returncode != 0:
                raise RuntimeError(f"Confirmation seed {seed} failed with exit code {process.returncode}; see {log_path}")
            if not progress or not best_path.is_file():
                raise RuntimeError(f"Confirmation seed {seed} completed without metrics/checkpoint")
            if sha256(DEPLOYMENT) != deployment_hash:
                raise RuntimeError("Deployment checkpoint changed during Data_v7 confirmation")
            state["progress"][str(seed)] = progress
            state["completed_seeds"].append(seed)
            state["current_seed"] = None
            state["current_process_pid"] = None
            state["deployment_sha256_latest"] = deployment_hash
            state["updated_at"] = now_iso()
            update_overall(state, started_monotonic)
            write_json_atomic(STATUS_PATH, state)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "summarize_data_v7_soil_vegetation_multiseed.py"),
                "--candidate-group", candidate_group,
                "--seeds", "0", *[str(seed) for seed in seeds],
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during multi-seed summary")
        state["state"] = "completed"
        state["deployment_sha256_after"] = deployment_hash_after
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        update_overall(state, started_monotonic)
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
        update_overall(state, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
