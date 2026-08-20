"""Run the paired Data_v8 C0/C1 confirmation on YOLO11 for three seeds.

The held-out test split is never exposed.  Every C0/C1 pair is rebuilt from
the same pretrained YOLO11 checkpoint with the same explicit seed.  Research
outputs are written to new run directories and the deployment checkpoint is
hash-guarded before, during, and after training.
"""

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
BASE_MODEL = ROOT / "yolo11n-seg.pt"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
STATUS_PATH = ROOT / "logs" / "v8_yolo11_paired_multiseed.status.json"
LOG_DIR = ROOT / "logs" / "v8_yolo11_paired_multiseed"
SUMMARY_SCRIPT = ROOT / "summarize_data_v8_yolo11_paired_multiseed.py"
GROUPS = ("c0_current_indices_early", "c1_raw_bands_early")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(group: str, seed: int) -> str:
    return f"v8_yolo11_{group}_ca_eca_1024_seed{seed}_confirm_v1"


def run_key(group: str, seed: int) -> str:
    return f"seed{seed}:{group}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen contract and print the six commands without starting training.",
    )
    return parser.parse_args()


def validate_inputs(cli: argparse.Namespace) -> tuple[dict, dict, str, str]:
    required = (
        FEATURE_ROOT,
        MANIFEST,
        GROUPS_CONFIG,
        DATA,
        BASE_MODEL,
        DEPLOYMENT,
        ROOT / "train_feature_ablation.py",
        SUMMARY_SCRIPT,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    seeds = tuple(cli.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    if seeds != (0, 1, 2):
        raise ValueError("The locked paired confirmation requires --seeds 0 1 2 in that order")
    if cli.epochs < 1 or cli.patience < 1 or cli.batch < 1 or cli.workers < 0:
        raise ValueError("--epochs, --patience and --batch must be positive; --workers cannot be negative")
    if cli.epochs != 60 or cli.patience != 20:
        raise ValueError("The locked confirmation contract requires --epochs 60 and --patience 20")
    if cli.imgsz != 1024 or cli.batch != 2:
        raise ValueError("The locked confirmation contract requires --imgsz 1024 and --batch 2")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    groups = json.loads(GROUPS_CONFIG.read_text(encoding="utf-8"))
    if manifest.get("held_out_split_used") is not False:
        raise RuntimeError("Data_v8 manifest does not explicitly keep the held-out split sealed")
    if manifest.get("selection_splits") != ["train", "valid"]:
        raise RuntimeError("Data_v8 manifest selection_splits must be exactly train/valid")
    configured = groups.get("groups", {})
    for group in GROUPS:
        channels = configured.get(group)
        if not isinstance(channels, list) or len(channels) != 5:
            raise RuntimeError(f"{group} must define exactly five channels")
        if channels[:3] != ["R", "G", "B"]:
            raise RuntimeError(f"{group} must preserve RGB as the first three channels")
    if configured[GROUPS[0]] != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError("C0 channel contract changed")
    if configured[GROUPS[1]] != ["R", "G", "B", "NIR", "RedEdge"]:
        raise RuntimeError("C1 channel contract changed")

    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text:
        raise RuntimeError(f"Training YAML exposes test: {DATA}")
    if "channels: 5" not in data_text:
        raise RuntimeError(f"Training YAML must declare five channels: {DATA}")

    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError(f"Deployment SHA256 mismatch before training: {deployment_hash}")
    return manifest, groups, sha256(BASE_MODEL), deployment_hash


def training_command(
    cli: argparse.Namespace,
    group: str,
    seed: int,
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
        "--data", str(DATA),
        "--name", run_name(group, seed),
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
        "--no-cache",
        "--deterministic",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def collect_progress(state: dict, started_monotonic: float) -> None:
    completed = set(state.get("completed_runs", []))
    actual_epochs = 0
    progress_units = 0
    budget = int(state["epochs_per_run"])
    total_runs = len(state["seeds"]) * len(GROUPS)
    for seed in state["seeds"]:
        for group in GROUPS:
            key = run_key(group, int(seed))
            progress = read_progress(PROJECT / run_name(group, int(seed)) / "results.csv")
            if progress:
                state["progress"][key] = progress
                actual_epochs += int(progress.get("epochs_recorded", 0))
            if key in completed:
                progress_units += budget
            elif progress:
                progress_units += min(int(progress.get("epochs_recorded", 0)), budget)
    target_units = total_runs * budget
    elapsed = max(time.monotonic() - started_monotonic, 0.0)
    eta = elapsed / progress_units * max(target_units - progress_units, 0) if progress_units else None
    state["overall"] = {
        "completed_runs": len(completed),
        "total_runs": total_runs,
        "actual_epochs_recorded": actual_epochs,
        "max_epoch_budget": target_units,
        "progress_units": progress_units,
        "percent": round(progress_units / max(target_units, 1) * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
        "early_stopping_note": "A completed early-stopped run counts as its full scheduled unit.",
    }
    state["updated_at"] = now_iso()
    write_json_atomic(STATUS_PATH, state)


def print_validation(cli: argparse.Namespace, base_hash: str, deployment_hash: str) -> None:
    print("Data_v8 YOLO11 paired confirmation contract: VALID")
    print(f"base_model_sha256={base_hash}")
    print(f"deployment_sha256={deployment_hash}")
    print("held_out_test_used=False")
    for seed in cli.seeds:
        for group in GROUPS:
            print("COMMAND:", subprocess.list2cmdline(training_command(cli, group, seed)))


def main() -> int:
    cli = parse_args()
    manifest, groups_config, base_hash, deployment_hash = validate_inputs(cli)
    if cli.validate_only:
        print_validation(cli, base_hash, deployment_hash)
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else {}
    previously_completed = set(previous.get("completed_runs", []))
    started_monotonic = time.monotonic()
    state = {
        "version": "data_v8_yolo11_paired_multiseed_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "groups": list(GROUPS),
        "channels": {group: groups_config["groups"][group] for group in GROUPS},
        "seeds": list(cli.seeds),
        "epochs_per_run": cli.epochs,
        "patience": cli.patience,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "device": cli.device,
        "workers": cli.workers,
        "base_model": str(BASE_MODEL.resolve()),
        "base_model_sha256": base_hash,
        "attention": "3x CoordAtt + 3x ECA",
        "extra_channel_initialization": "mean of pretrained RGB kernels",
        "paired_initialization": "fresh identical YOLO11 checkpoint and explicit RNG seed for each C0/C1 pair",
        "deterministic": True,
        "selection_splits": manifest["selection_splits"],
        "held_out_test_used": False,
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "completed_runs": [],
        "current_seed": None,
        "current_group": None,
        "current_run": None,
        "current_process_pid": None,
        "progress": {},
        "overall": {},
    }
    collect_progress(state, started_monotonic)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        # Pair C0 and C1 within each seed so both use the same deterministic
        # initialization contract before moving to the next seed.
        for seed in cli.seeds:
            for group in GROUPS:
                key = run_key(group, seed)
                name = run_name(group, seed)
                run_dir = PROJECT / name
                results_path = run_dir / "results.csv"
                best_path = run_dir / "weights" / "best.pt"
                progress = read_progress(results_path)
                if key in previously_completed and progress and best_path.is_file():
                    state["completed_runs"].append(key)
                    collect_progress(state, started_monotonic)
                    continue

                resume = None
                last_path = run_dir / "weights" / "last.pt"
                if progress and last_path.is_file():
                    resume = last_path
                elif run_dir.exists():
                    raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")

                command = training_command(cli, group, seed, resume=resume)
                log_path = LOG_DIR / f"{name}.log"
                state["current_seed"] = seed
                state["current_group"] = group
                state["current_run"] = name
                state["current_log"] = str(log_path.resolve())
                state["updated_at"] = now_iso()
                write_json_atomic(STATUS_PATH, state)
                with log_path.open("a", encoding="utf-8", buffering=1) as log:
                    log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    state["current_process_pid"] = process.pid
                    write_json_atomic(STATUS_PATH, state)
                    while process.poll() is None:
                        collect_progress(state, started_monotonic)
                        time.sleep(max(2.0, cli.poll_seconds))
                collect_progress(state, started_monotonic)
                if process.returncode != 0:
                    raise RuntimeError(f"{name} exited with code {process.returncode}; see {log_path}")
                if not read_progress(results_path) or not best_path.is_file():
                    raise RuntimeError(f"Completed process did not create readable metrics/checkpoint: {run_dir}")
                if sha256(DEPLOYMENT) != deployment_hash:
                    raise RuntimeError("Deployment checkpoint changed during paired confirmation")
                state["completed_runs"].append(key)
                state["current_seed"] = None
                state["current_group"] = None
                state["current_run"] = None
                state["current_process_pid"] = None
                state["deployment_sha256_latest"] = deployment_hash
                collect_progress(state, started_monotonic)

        summary_log = LOG_DIR / "summary.log"
        summary_command = [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--seeds",
            *[str(seed) for seed in cli.seeds],
        ]
        with summary_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                summary_command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Summary exited with code {result.returncode}; see {summary_log}")
        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during final summary")
        state["state"] = "completed"
        state["current_seed"] = None
        state["current_group"] = None
        state["current_run"] = None
        state["current_process_pid"] = None
        state["deployment_sha256_after"] = deployment_hash_after
        state["deployment_unchanged"] = True
        state["summary"] = str(
            (ROOT / "reports" / "Data_v8_yolo11_paired_multiseed" / "summary.json").resolve()
        )
        state["completed_at"] = now_iso()
        collect_progress(state, started_monotonic)
        return 0
    except BaseException as exc:
        current_hash = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        state["deployment_sha256_after"] = current_hash
        state["deployment_unchanged"] = current_hash == deployment_hash
        collect_progress(state, started_monotonic)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
