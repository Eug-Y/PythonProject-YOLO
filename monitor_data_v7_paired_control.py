"""Monitor the exact Data_v7 paired control and summarize it when training exits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

from run_feature_ablations import read_progress, write_json_atomic
from run_data_v7_soil_vegetation_screen import GROUPS, experiment_name
from summarize_data_v7_paired_control import CONTROL_NAME, PROJECT


ROOT = Path(__file__).resolve().parent
RUN_DIR = PROJECT / CONTROL_NAME
STATUS_PATH = ROOT / "logs" / "v7_paired_control.status.json"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
FEATURE_ROOT = ROOT / "datasets" / "Data_v7_soil_vegetation_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
ANALYSIS_ROOT = ROOT / "reports" / "Data_v7_failure_analysis"
DATA_5CH = ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed" / "data_5ch_teacher_refined.yaml"
DATA_6CH = ROOT / "configs" / "data_v7_selection_6ch.yaml"
DATA_7CH = ROOT / "configs" / "data_v7_selection_7ch.yaml"
CHANNELS = {
    "current_5ch": ("R", "G", "B", "NDVI", "NDRE"),
    "add_re_bsi_proxy_6ch": ("R", "G", "B", "NDVI", "NDRE", "RE_BSI_PROXY"),
    "add_mcari_6ch": ("R", "G", "B", "NDVI", "NDRE", "MCARI"),
    "soil_vegetation_7ch": ("R", "G", "B", "NDVI", "NDRE", "RE_BSI_PROXY", "MCARI"),
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_alive(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--target-epochs", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment SHA256 mismatch before paired-control monitoring")
    state = {
        "version": "data_v7_paired_control_status_v1",
        "state": "running",
        "monitor_pid": psutil.Process().pid,
        "training_pid": args.training_pid,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run": CONTROL_NAME,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "target_epochs": args.target_epochs,
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "progress": None,
    }
    write_json_atomic(STATUS_PATH, state)
    try:
        while process_alive(args.training_pid):
            progress = read_progress(RUN_DIR / "results.csv")
            if progress:
                recorded = int(progress["epochs_recorded"])
                progress["percent"] = round(recorded / max(args.target_epochs, 1) * 100.0, 2)
                state["progress"] = progress
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)
            time.sleep(max(2.0, args.poll_seconds))
        time.sleep(1.0)
        progress = read_progress(RUN_DIR / "results.csv")
        best = RUN_DIR / "weights" / "best.pt"
        if progress is None or not best.is_file():
            raise RuntimeError("Paired-control process exited without readable results/best.pt")
        state["progress"] = progress
        current_hash = sha256(DEPLOYMENT)
        if current_hash != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during paired-control training")
        subprocess.run(
            [sys.executable, str(ROOT / "summarize_data_v7_paired_control.py")],
            cwd=ROOT,
            check=True,
        )
        analysis_specs = [("current_5ch", RUN_DIR / "weights" / "best.pt", DATA_5CH)]
        for group in GROUPS:
            checkpoint = PROJECT / experiment_name(group, 0) / "weights" / "best.pt"
            data = DATA_7CH if len(CHANNELS[group]) == 7 else DATA_6CH
            analysis_specs.append((group, checkpoint, data))
        state["state"] = "analyzing_failure_locations"
        state["analysis_completed"] = []
        write_json_atomic(STATUS_PATH, state)
        for group, checkpoint, data in analysis_specs:
            state["analysis_current"] = group
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "error_analysis_multichannel.py"),
                    "--checkpoint", str(checkpoint),
                    "--data", str(data),
                    "--manifest", str(MANIFEST),
                    "--feature-root", str(FEATURE_ROOT),
                    "--channels", *CHANNELS[group],
                    "--output", str(ANALYSIS_ROOT / group),
                    "--imgsz", "1024",
                    "--conf", "0.25",
                    "--device", "0",
                ],
                cwd=ROOT,
                check=True,
            )
            state["analysis_completed"].append(group)
            write_json_atomic(STATUS_PATH, state)
        state["analysis_current"] = "building_web_report"
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS_PATH, state)
        subprocess.run(
            [sys.executable, str(ROOT / "build_data_v7_feature_failure_report.py")],
            cwd=ROOT,
            check=True,
        )
        state["state"] = "completed"
        state["analysis_current"] = None
        state["failure_report"] = str(
            (ROOT / "reports" / "Data_v7_feature_failure_report" / "index.html").resolve()
        )
        state["deployment_sha256_after"] = current_hash
        state["deployment_unchanged"] = True
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
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
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
