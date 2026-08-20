"""Run and monitor paired H0/H1 train-only difficulty-sampling confirmation."""

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
SAMPLING_WEIGHTS = ROOT / "configs" / "generated" / "data_v8_train_difficulty_weights_h1.json"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
STATUS_PATH = ROOT / "logs" / "data_v8_difficulty_sampling_multiseed.status.json"
LOG_DIR = ROOT / "logs" / "data_v8_difficulty_sampling_multiseed"
VARIANTS = ("h0", "h1")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_name(variant: str, seed: int) -> str:
    suffix = "shuffle" if variant == "h0" else "difficulty"
    return f"v8_{variant}_{suffix}_30ep_seed{seed}_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def validate_inputs() -> tuple[str, str]:
    for path in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DATA, SAMPLING_WEIGHTS, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    if "test:" in DATA.read_text(encoding="utf-8").lower():
        raise RuntimeError(f"Training YAML must not expose test: {DATA}")
    weights = json.loads(SAMPLING_WEIGHTS.read_text(encoding="utf-8"))
    if (
        weights.get("source_split") != "train"
        or weights.get("validation_split_used") is not False
        or weights.get("held_out_split_used") is not False
    ):
        raise RuntimeError("Difficulty weights are not explicitly train-only")
    return sha256(SAMPLING_WEIGHTS), sha256(DEPLOYMENT)


def training_command(cli: argparse.Namespace, variant: str, seed: int, resume: Path | None = None) -> list[str]:
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
        "--epochs", str(cli.epochs),
        "--patience", str(cli.patience),
        "--imgsz", str(cli.imgsz),
        "--batch", str(cli.batch),
        "--device", cli.device,
        "--workers", str(cli.workers),
        "--seed", str(seed),
        "--cache",
        "--deterministic",
        "--name", experiment_name(variant, seed),
    ]
    if variant == "h1":
        command.extend(("--sampling-weights", str(SAMPLING_WEIGHTS)))
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def aggregate_pairs(state: dict, seeds: tuple[int, ...]) -> None:
    pairs: list[dict[str, object]] = []
    for seed in seeds:
        h0_name = experiment_name("h0", seed)
        h1_name = experiment_name("h1", seed)
        h0 = state["progress"].get(h0_name)
        h1 = state["progress"].get(h1_name)
        if not h0 or not h1:
            continue
        h0_metric = float(h0["best_mask_map50_95"])
        h1_metric = float(h1["best_mask_map50_95"])
        pairs.append(
            {
                "seed": seed,
                "h0_best_mask_map50_95": h0_metric,
                "h1_best_mask_map50_95": h1_metric,
                "delta": h1_metric - h0_metric,
                "delta_pp": 100.0 * (h1_metric - h0_metric),
            }
        )
    state["paired_results"] = pairs
    if len(pairs) == len(seeds):
        deltas = [float(pair["delta"]) for pair in pairs]
        mean_delta = sum(deltas) / len(deltas)
        state["gate"] = {
            "complete": True,
            "mean_delta": mean_delta,
            "mean_delta_pp": 100.0 * mean_delta,
            "worst_delta": min(deltas),
            "worst_delta_pp": 100.0 * min(deltas),
            "mean_gain_threshold": 0.002,
            "worst_seed_threshold": -0.005,
            "passed": mean_delta >= 0.002 and min(deltas) >= -0.005,
            "deployment_action": "none_research_only",
        }


def update_overall(state: dict, seeds: tuple[int, ...], epochs: int, started_monotonic: float) -> None:
    names = [experiment_name(variant, seed) for seed in seeds for variant in VARIANTS]
    recorded = sum(min(int(state["progress"].get(name, {}).get("epochs_recorded", 0)), epochs) for name in names)
    target = len(names) * epochs
    elapsed = max(time.monotonic() - started_monotonic, 0.0)
    eta = elapsed / recorded * max(target - recorded, 0) if recorded else None
    state["overall"] = {
        "epochs_recorded": recorded,
        "target_epochs": target,
        "percent": round(100.0 * recorded / max(target, 1), 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }


def main() -> int:
    cli = parse_args()
    seeds = tuple(cli.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    if cli.epochs != 30:
        raise ValueError("This paired confirmation is locked to 30 epochs")
    if cli.patience < 1 or cli.batch < 1 or cli.workers < 0:
        raise ValueError("Invalid training budget")
    weights_hash, deployment_hash = validate_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    started = now_iso()
    state: dict[str, object] = {
        "version": "data_v8_difficulty_sampling_multiseed_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": started,
        "updated_at": started,
        "seeds": list(seeds),
        "variants": list(VARIANTS),
        "epochs_per_run": cli.epochs,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "architecture": "YOLO11n-seg",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "attention": "3xCoordAtt+3xECA",
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "sampling_weights": str(SAMPLING_WEIGHTS.resolve()),
        "sampling_weights_sha256": weights_hash,
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

    try:
        for seed in seeds:
            for variant in VARIANTS:
                name = experiment_name(variant, seed)
                run_dir = PROJECT / name
                results_path = run_dir / "results.csv"
                best_path = run_dir / "weights" / "best.pt"
                progress = read_progress(results_path)
                if progress and int(progress["epochs_recorded"]) >= cli.epochs and best_path.is_file():
                    state["progress"][name] = progress
                    state["completed_runs"].append(name)
                    aggregate_pairs(state, seeds)
                    update_overall(state, seeds, cli.epochs, started_monotonic)
                    write_json_atomic(STATUS_PATH, state)
                    continue

                resume = None
                last_path = run_dir / "weights" / "last.pt"
                if progress and last_path.is_file():
                    resume = last_path
                elif run_dir.exists():
                    raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
                command = training_command(cli, variant, seed, resume)
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
                        aggregate_pairs(state, seeds)
                        update_overall(state, seeds, cli.epochs, started_monotonic)
                        write_json_atomic(STATUS_PATH, state)
                        time.sleep(max(2.0, cli.poll_seconds))
                progress = read_progress(results_path)
                if process.returncode != 0:
                    raise RuntimeError(f"Run failed with exit code {process.returncode}: {name}; see {log_path}")
                if not progress or int(progress["epochs_recorded"]) < cli.epochs or not best_path.is_file():
                    raise RuntimeError(f"Run completed without full metrics/checkpoint: {name}")
                state["progress"][name] = progress
                state["completed_runs"].append(name)
                state["current_run"] = None
                state["current_process_pid"] = None
                state["updated_at"] = now_iso()
                aggregate_pairs(state, seeds)
                update_overall(state, seeds, cli.epochs, started_monotonic)
                write_json_atomic(STATUS_PATH, state)

        deployment_hash_after = sha256(DEPLOYMENT)
        if deployment_hash_after != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during research training")
        state["state"] = "completed"
        state["deployment_sha256_after"] = deployment_hash_after
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        aggregate_pairs(state, seeds)
        update_overall(state, seeds, cli.epochs, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        return 0
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        state["deployment_sha256_after"] = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["deployment_unchanged"] = state["deployment_sha256_after"] == deployment_hash
        aggregate_pairs(state, seeds)
        update_overall(state, seeds, cli.epochs, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
