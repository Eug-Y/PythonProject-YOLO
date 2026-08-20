"""Run paired Data_v6 terrain-factor screens across compatible YOLO segmentation families."""

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
from train_ca_5ch import SUPPORTED_SEGMENTATION_MODELS, validate_attention_compatibility


DEFAULT_MODELS = ("yolo11n-seg.pt", "yolov8n-seg.pt", "yolo12n-seg.pt", "yolo26n-seg.pt")
DEFAULT_GROUPS = ("current_5ch", "slope_6ch")
FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v6_terrain_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = PROJECT_ROOT / "configs" / "feature_ablation_groups_v6_terrain.json"
PROJECT = PROJECT_ROOT / "runs" / "segment" / "land_seg"
STATUS_PATH = PROJECT_ROOT / "logs" / "v6_terrain_model_screen.status.json"
LOG_DIR = PROJECT_ROOT / "logs" / "v6_terrain_model_screen"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def experiment_name(group: str, model: str, seed: int, attention: str = "none") -> str:
    return f"v6_{group}_{Path(model).stem}_{attention}_seed{seed}_v1"


def training_command(
    *,
    group: str,
    model: str,
    attention: str,
    epochs: int,
    imgsz: int,
    device: str,
    seed: int,
    batch: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train_feature_ablation.py"),
        "--group", group,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--base-model", model,
        "--attention", attention,
        "--name", experiment_name(group, model, seed, attention),
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--device", device,
        "--seed", str(seed),
    ]
    if batch is not None:
        command.extend(("--batch", str(batch)))
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_SEGMENTATION_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--attention", choices=("none", "ca_eca"), default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    for required in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG):
        if not required.exists():
            raise FileNotFoundError(
                f"Required Data_v6 artifact is missing: {required}. "
                "Audit and prepare aligned terrain sources before model screening."
            )
    group_config = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))["groups"]
    unknown_groups = sorted(set(cli.groups) - set(group_config))
    if unknown_groups:
        raise ValueError(f"Unknown terrain groups: {unknown_groups}")
    if "current_5ch" not in cli.groups:
        raise ValueError("Paired screening requires the current_5ch control group")
    if len(cli.groups) < 2:
        raise ValueError("Paired screening requires at least one terrain candidate group")
    for model in cli.models:
        validate_attention_compatibility(model, cli.attention)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runs = [(group, model) for model in cli.models for group in cli.groups]
    state = {
        "version": "data_v6_terrain_model_screen_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "models": list(cli.models),
        "groups": list(cli.groups),
        "attention": cli.attention,
        "seed": cli.seed,
        "epochs_per_run": cli.epochs,
        "completed": [],
        "current": None,
        "progress": {},
    }
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        for group, model in runs:
            name = experiment_name(group, model, cli.seed, cli.attention)
            run_dir = PROJECT / name
            progress = read_progress(run_dir / "results.csv")
            if progress and progress["last_epoch"] >= cli.epochs and (run_dir / "weights" / "best.pt").is_file():
                state["completed"].append(name)
                state["progress"][name] = progress
                state["updated_at"] = now_iso()
                write_json_atomic(STATUS_PATH, state)
                continue
            command = training_command(
                group=group,
                model=model,
                attention=cli.attention,
                epochs=cli.epochs,
                imgsz=cli.imgsz,
                device=cli.device,
                seed=cli.seed,
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
            if progress:
                state["progress"][name] = progress
            if process.returncode != 0:
                raise RuntimeError(f"Run failed with exit code {process.returncode}: {name}; see {log_path}")
            if not progress or progress["last_epoch"] < cli.epochs:
                raise RuntimeError(f"Run exited without {cli.epochs} recorded epochs: {name}")
            state["completed"].append(name)
            state["current"] = None
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)
        state["state"] = "completed"
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        write_json_atomic(STATUS_PATH, state)
        candidate_groups = [group for group in cli.groups if group != "current_5ch"]
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "summarize_terrain_model_screen.py"),
                "--models",
                *cli.models,
                "--candidate-groups",
                *candidate_groups,
                "--seed",
                str(cli.seed),
                "--attention",
                cli.attention,
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env=environment,
        )
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
