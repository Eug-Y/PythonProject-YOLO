"""Run paired seed 0/1/2 validation for a promoted Data_v6 terrain candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from multispectral import PROJECT_ROOT
from run_feature_ablations import read_progress, write_json_atomic
from run_terrain_model_screen import FEATURE_ROOT, GROUPS_CONFIG, MANIFEST, PROJECT, training_command
from summarize_terrain_model_screen import experiment_name
from train_ca_5ch import SUPPORTED_SEGMENTATION_MODELS, validate_attention_compatibility


SCREEN_SUMMARY = PROJECT_ROOT / "reports" / "Data_v6_terrain_model_screen" / "summary.json"
STATUS_PATH = PROJECT_ROOT / "logs" / "v6_terrain_multiseed.status.json"
LOG_DIR = PROJECT_ROOT / "logs" / "v6_terrain_multiseed"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_candidate(
    model: str | None,
    group: str | None,
    attention: str | None,
    summary_path: Path = SCREEN_SUMMARY,
) -> tuple[str, str, str]:
    if (model is None) != (group is None):
        raise ValueError("Provide both --model and --candidate-group, or neither")
    if model is not None and group is not None:
        return model, group, attention or "none"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Terrain screen summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate = summary.get("recommended_next")
    if not candidate:
        raise RuntimeError("The terrain screen has no promoted candidate for multi-seed validation")
    return str(candidate["model"]), str(candidate["candidate_group"]), attention or str(summary.get("attention", "none"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_SEGMENTATION_MODELS, default=None)
    parser.add_argument("--candidate-group", default=None)
    parser.add_argument("--attention", choices=("none", "ca_eca"), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    seeds = tuple(cli.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    model, candidate_group, attention = resolve_candidate(cli.model, cli.candidate_group, cli.attention)
    validate_attention_compatibility(model, attention)
    for required in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG):
        if not required.exists():
            raise FileNotFoundError(f"Required Data_v6 artifact is missing: {required}")
    groups = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))["groups"]
    if candidate_group not in groups or candidate_group == "current_5ch":
        raise ValueError(f"Invalid terrain candidate group: {candidate_group}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [(seed, group) for seed in seeds for group in ("current_5ch", candidate_group)]
    state = {
        "version": "data_v6_terrain_multiseed_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "model": model,
        "candidate_group": candidate_group,
        "attention": attention,
        "seeds": list(seeds),
        "completed": [],
        "current": None,
        "progress": {},
    }
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    try:
        for seed, group in pairs:
            name = experiment_name(group, model, seed, attention)
            run_dir = PROJECT / name
            progress = read_progress(run_dir / "results.csv")
            if progress and progress["last_epoch"] >= cli.epochs and (run_dir / "weights" / "best.pt").is_file():
                state["completed"].append(name)
                state["progress"][name] = progress
                continue
            command = training_command(
                group=group,
                model=model,
                attention=attention,
                epochs=cli.epochs,
                imgsz=cli.imgsz,
                device=cli.device,
                seed=seed,
                batch=cli.batch,
            )
            checkpoint = run_dir / "weights" / "last.pt"
            if progress and checkpoint.is_file():
                command.extend(("--resume", str(checkpoint)))
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
            log_path = LOG_DIR / f"{name}.log"
            state["current"] = name
            state["current_log"] = str(log_path)
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                log_handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                state["current_process_pid"] = process.pid
                write_json_atomic(STATUS_PATH, state)
                while process.poll() is None:
                    progress = read_progress(run_dir / "results.csv")
                    if progress:
                        state["progress"][name] = progress
                    state["updated_at"] = now_iso()
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
            progress = read_progress(run_dir / "results.csv")
            if process.returncode != 0:
                raise RuntimeError(f"Run failed with exit code {process.returncode}: {name}; see {log_path}")
            if not progress or progress["last_epoch"] < cli.epochs:
                raise RuntimeError(f"Run exited without {cli.epochs} recorded epochs: {name}")
            state["completed"].append(name)
            state["progress"][name] = progress
            state["current"] = None
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "summarize_terrain_multiseed.py"),
                "--model", model,
                "--candidate-group", candidate_group,
                "--attention", attention,
                "--seeds", *[str(seed) for seed in seeds],
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env=environment,
        )
        state["state"] = "completed"
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        write_json_atomic(STATUS_PATH, state)
        return 0
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

