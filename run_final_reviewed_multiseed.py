"""Run extended seed 0/1/2 fine-tuning on the final-reviewed teacher-refined labels."""

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
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "feature_ablation_groups.json"
DATA = ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed" / "data_5ch_teacher_refined.yaml"
DATA_MANIFEST = DATA.parent / "manifest.json"
INITIALIZATION = PROJECT / "v6_preterrain_current5_yolov8n_none_seed0_v1" / "weights" / "best.pt"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
STATUS_PATH = ROOT / "logs" / "v6_final_reviewed_multiseed.status.json"
LOG_DIR = ROOT / "logs" / "v6_final_reviewed_multiseed"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_name(seed: int) -> str:
    return f"v6_final_reviewed_teacher_refined_yolov8n_long_seed{seed}_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def validate_inputs() -> tuple[str, str]:
    for path in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DATA, DATA_MANIFEST, INITIALIZATION, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text:
        raise RuntimeError(f"Final-reviewed training YAML must not expose test: {DATA}")
    dataset_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    if dataset_manifest.get("held_out_test_used") is not False:
        raise RuntimeError("Final-reviewed dataset manifest does not explicitly keep test sealed")
    return sha256(INITIALIZATION), sha256(DEPLOYMENT)


def update_overall(state: dict, started_monotonic: float) -> None:
    target = int(state["epochs_per_run"]) * len(state["seeds"])
    recorded = sum(
        min(int(progress.get("epochs_recorded", 0)), int(state["epochs_per_run"]))
        for progress in state["progress"].values()
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


def training_command(cli: argparse.Namespace, seed: int, *, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", "current_5ch",
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(DATA),
        "--name", experiment_name(seed),
        "--base-model", "yolov8n-seg.pt",
        "--attention", "none",
        "--epochs", str(cli.epochs),
        "--patience", str(cli.patience),
        "--warmup-epochs", "1",
        "--imgsz", str(cli.imgsz),
        "--batch", str(cli.batch),
        "--device", cli.device,
        "--seed", str(seed),
    ]
    if resume is None:
        command.extend(("--init-checkpoint", str(INITIALIZATION)))
    else:
        command.extend(("--resume", str(resume)))
    return command


def main() -> int:
    cli = parse_args()
    seeds = tuple(cli.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    if cli.epochs <= 15:
        raise ValueError("Extended training must exceed the previous 15-epoch budget")
    if cli.patience < 1 or cli.batch < 1:
        raise ValueError("--patience and --batch must be positive")
    initialization_hash, deployment_hash = validate_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    previously_completed = set(previous.get("completed_runs", []))
    started_monotonic = time.monotonic()
    started = now_iso()
    state = {
        "version": "data_v6_final_reviewed_multiseed_status_v2",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": started,
        "updated_at": started,
        "seeds": list(seeds),
        "epochs_per_run": cli.epochs,
        "patience": cli.patience,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "dataloader_seeded_per_run": True,
        "data": str(DATA.resolve()),
        "initialization_checkpoint": str(INITIALIZATION.resolve()),
        "initialization_sha256": initialization_hash,
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "completed_runs": [],
        "current_run": None,
        "current_process_pid": None,
        "progress": {},
        "overall": {},
    }
    update_overall(state, started_monotonic)
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        for seed in seeds:
            name = experiment_name(seed)
            run_dir = PROJECT / name
            results_path = run_dir / "results.csv"
            best_path = run_dir / "weights" / "best.pt"
            progress = read_progress(results_path)
            if name in previously_completed and progress and best_path.is_file():
                state["completed_runs"].append(name)
                state["progress"][name] = progress
                update_overall(state, started_monotonic)
                write_json_atomic(STATUS_PATH, state)
                continue

            resume = None
            last_path = run_dir / "weights" / "last.pt"
            if progress and last_path.is_file():
                resume = last_path
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
            command = training_command(cli, seed, resume=resume)
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
                    update_overall(state, started_monotonic)
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
            update_overall(state, started_monotonic)
            write_json_atomic(STATUS_PATH, state)

        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during research training")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "summarize_final_reviewed_multiseed.py"),
                "--seeds", *[str(seed) for seed in seeds],
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
        state["state"] = "completed"
        state["current_run"] = None
        state["current_process_pid"] = None
        state["deployment_sha256_after"] = deployment_hash_after
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        update_overall(state, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        return 0
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        state["deployment_sha256_after"] = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["deployment_unchanged"] = state["deployment_sha256_after"] == deployment_hash
        update_overall(state, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
