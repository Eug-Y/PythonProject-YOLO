"""Run the seed-0 Data_v7 soil/vegetation feature screen without opening test."""

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
FEATURE_ROOT = ROOT / "datasets" / "Data_v7_soil_vegetation_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "feature_ablation_groups_v7_soil_vegetation.json"
DATA_BY_CHANNELS = {
    6: ROOT / "configs" / "data_v7_selection_6ch.yaml",
    7: ROOT / "configs" / "data_v7_selection_7ch.yaml",
}
INITIALIZATION = PROJECT / "v6_preterrain_current5_yolov8n_none_seed0_v1" / "weights" / "best.pt"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
STATUS_PATH = ROOT / "logs" / "v7_soil_vegetation_screen.status.json"
LOG_DIR = ROOT / "logs" / "v7_soil_vegetation_screen"
GROUPS = ("add_re_bsi_proxy_6ch", "add_mcari_6ch", "soil_vegetation_7ch")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_name(group: str, seed: int = 0) -> str:
    return f"v7_{group}_yolov8n_none_1024_seed{seed}_screen_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def validate_inputs(groups: tuple[str, ...]) -> tuple[dict[str, list[str]], str, str]:
    required = (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, INITIALIZATION, DEPLOYMENT, *DATA_BY_CHANNELS.values())
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError(
            f"Deployment SHA256 mismatch before screening: expected {DEPLOYMENT_SHA256}, got {deployment_hash}"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if "test" in manifest.get("split_counts", {}):
        raise RuntimeError("Data_v7 feature manifest must not contain the frozen test split")
    if manifest.get("held_out_split") != "test":
        raise RuntimeError("Data_v7 manifest must declare test as held out")
    config = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))
    configured = config.get("groups", {})
    unknown = sorted(set(groups) - set(configured))
    if unknown:
        raise ValueError(f"Unknown Data_v7 groups: {unknown}")
    for group in groups:
        channels = configured[group]
        data = DATA_BY_CHANNELS.get(len(channels))
        if data is None:
            raise ValueError(f"No train/valid YAML for {group} with {len(channels)} channels")
        text = data.read_text(encoding="utf-8").lower()
        if "test:" in text or f"channels: {len(channels)}" not in text:
            raise RuntimeError(f"Invalid train/valid-only channel contract: {data}")
    return configured, sha256(INITIALIZATION), deployment_hash


def training_command(
    cli: argparse.Namespace,
    group: str,
    channel_count: int,
    *,
    resume: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", group,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(DATA_BY_CHANNELS[channel_count]),
        "--name", experiment_name(group, cli.seed),
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
    ]
    if resume is None:
        command.extend(("--init-checkpoint", str(INITIALIZATION)))
    else:
        command.extend(("--resume", str(resume)))
    return command


def update_overall(state: dict, started_monotonic: float) -> None:
    target = int(state["epochs_per_group"]) * len(state["groups"])
    recorded = sum(
        min(int(progress.get("epochs_recorded", 0)), int(state["epochs_per_group"]))
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


def main() -> int:
    cli = parse_args()
    groups = tuple(cli.groups)
    if len(groups) != len(set(groups)):
        raise ValueError("--groups must not contain duplicates")
    if cli.epochs < 1 or cli.patience < 1 or cli.batch < 1:
        raise ValueError("--epochs, --patience and --batch must be positive")
    configured, initialization_hash, deployment_hash = validate_inputs(groups)
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    previously_completed = set(previous.get("completed_groups", []))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    state = {
        "version": "data_v7_soil_vegetation_screen_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "groups": list(groups),
        "seed": cli.seed,
        "epochs_per_group": cli.epochs,
        "patience": cli.patience,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "slope_included": False,
        "attention": "none",
        "initialization_checkpoint": str(INITIALIZATION.resolve()),
        "initialization_sha256": initialization_hash,
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "completed_groups": [],
        "current_group": None,
        "current_process_pid": None,
        "progress": {},
        "overall": {},
    }
    update_overall(state, started_monotonic)
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        for group in groups:
            name = experiment_name(group, cli.seed)
            run_dir = PROJECT / name
            results_path = run_dir / "results.csv"
            best_path = run_dir / "weights" / "best.pt"
            progress = read_progress(results_path)
            if group in previously_completed and progress and best_path.is_file():
                state["completed_groups"].append(group)
                state["progress"][group] = progress
                update_overall(state, started_monotonic)
                write_json_atomic(STATUS_PATH, state)
                continue

            resume = None
            last_path = run_dir / "weights" / "last.pt"
            if progress and last_path.is_file():
                resume = last_path
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
            command = training_command(cli, group, len(configured[group]), resume=resume)
            log_path = LOG_DIR / f"{name}.log"
            state["current_group"] = group
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
                        state["progress"][group] = progress
                    state["updated_at"] = now_iso()
                    update_overall(state, started_monotonic)
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
            progress = read_progress(results_path)
            if process.returncode != 0:
                raise RuntimeError(f"Run failed with exit code {process.returncode}: {name}; see {log_path}")
            if not progress or not best_path.is_file():
                raise RuntimeError(f"Run completed without readable metrics/checkpoint: {name}")
            current_deployment_hash = sha256(DEPLOYMENT)
            if current_deployment_hash != deployment_hash:
                raise RuntimeError("Deployment checkpoint changed during Data_v7 screening")
            state["deployment_sha256_latest"] = current_deployment_hash
            state["progress"][group] = progress
            state["completed_groups"].append(group)
            state["current_group"] = None
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            update_overall(state, started_monotonic)
            write_json_atomic(STATUS_PATH, state)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "summarize_data_v7_soil_vegetation_screen.py"),
                "--groups", *groups,
                "--seed", str(cli.seed),
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during Data_v7 summary")
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
