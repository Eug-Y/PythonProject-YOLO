"""Run and monitor the locked C0/C1/F1/F2 Data_v8 Seed-0 screen."""

from __future__ import annotations

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
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
STATUS = ROOT / "logs" / "v8_fusion_screen.status.json"
LOG_DIR = ROOT / "logs" / "v8_fusion_screen"
SUMMARY = ROOT / "reports" / "Data_v8_multispectral_fusion_screen" / "summary.json"
GROUPS = (
    "c0_current_indices_early",
    "c1_raw_bands_early",
    "f1_p4_concat",
    "f2_multiscale_gated",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(group: str) -> str:
    return f"v8_{group}_yolov8n_none_1024_seed0_screen_v1"


def verify_inputs() -> str:
    required = (
        ROOT / "train_multispectral_fusion.py",
        ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json",
        ROOT / "configs" / "data_v8_fusion_groups.json",
        DEPLOYMENT,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError(f"Deployment SHA256 mismatch before screen: {deployment_hash}")
    for group in GROUPS:
        smoke = PROJECT / f"smoke_{group}_seed0_v1" / "weights" / "best.pt"
        if not smoke.is_file():
            raise FileNotFoundError(f"Required smoke checkpoint missing: {smoke}")
    return deployment_hash


def update_progress(state: dict, started: float) -> None:
    completed_epochs = 0
    for group in GROUPS:
        results = PROJECT / run_name(group) / "results.csv"
        progress = read_progress(results)
        if progress:
            state["progress"][group] = progress
            completed_epochs += min(int(progress.get("epochs_recorded", 0)), 30)
    elapsed = max(time.monotonic() - started, 0.0)
    target = len(GROUPS) * 30
    remaining = max(target - completed_epochs, 0)
    eta = elapsed / completed_epochs * remaining if completed_epochs else None
    state["overall"] = {
        "epochs_recorded": completed_epochs,
        "target_epochs": target,
        "percent": round(completed_epochs / target * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1) if eta is not None else None,
    }
    state["updated_at"] = now_iso()
    write_json_atomic(STATUS, state)


def command_for(group: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "train_multispectral_fusion.py"),
        "--group", group,
        "--name", run_name(group),
        "--epochs", "30",
        "--patience", "12",
        "--imgsz", "1024",
        "--batch", "2",
        "--device", "0",
        "--workers", "2",
        "--seed", "0",
    ]


def main() -> int:
    deployment_hash = verify_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {
        "version": "data_v8_multispectral_fusion_screen_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "groups": list(GROUPS),
        "current_group": None,
        "current_process_pid": None,
        "completed_groups": [],
        "progress": {},
        "overall": {},
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "f3_included": False,
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
    }
    update_progress(state, started)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        for group in GROUPS:
            run_dir = PROJECT / run_name(group)
            results = run_dir / "results.csv"
            best = run_dir / "weights" / "best.pt"
            progress = read_progress(results)
            if progress and int(progress.get("epochs_recorded", 0)) >= 30 and best.is_file():
                state["completed_groups"].append(group)
                update_progress(state, started)
                continue
            if run_dir.exists():
                raise RuntimeError(
                    f"Incomplete pre-existing run requires manual audit before resume: {run_dir}"
                )
            state["current_group"] = group
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS, state)
            log_path = LOG_DIR / f"{group}.log"
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                process = subprocess.Popen(
                    command_for(group),
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                state["current_process_pid"] = process.pid
                write_json_atomic(STATUS, state)
                while process.poll() is None:
                    update_progress(state, started)
                    time.sleep(5)
                update_progress(state, started)
                if process.returncode != 0:
                    raise RuntimeError(f"{group} exited with code {process.returncode}; see {log_path}")
            if not best.is_file() or not read_progress(results):
                raise RuntimeError(f"Completed process did not create valid outputs: {run_dir}")
            if sha256(DEPLOYMENT) != DEPLOYMENT_SHA256:
                raise RuntimeError("Deployment checkpoint changed during Data_v8 screening")
            state["completed_groups"].append(group)
            state["current_group"] = None
            state["current_process_pid"] = None
            update_progress(state, started)

        summary_log = LOG_DIR / "summary.log"
        with summary_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(ROOT / "summarize_data_v8_fusion.py")],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0 or not SUMMARY.is_file():
            raise RuntimeError(f"Summary failed with code {result.returncode}; see {summary_log}")
        state["state"] = "completed"
        state["summary"] = str(SUMMARY.resolve())
        state["completed_at"] = now_iso()
        state["deployment_sha256_after"] = sha256(DEPLOYMENT)
        state["deployment_unchanged"] = state["deployment_sha256_after"] == DEPLOYMENT_SHA256
        update_progress(state, started)
        return 0
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["deployment_sha256_after"] = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state["deployment_unchanged"] = state["deployment_sha256_after"] == DEPLOYMENT_SHA256
        update_progress(state, started)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
