"""Run the four Data_v4 channel-group ablations sequentially with live status."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
LOG_DIR = ROOT / "logs" / "v4_feature_ablations"
STATUS_PATH = ROOT / "logs" / "v4_feature_ablation.status.json"
GROUP_ORDER = ("rgb", "current_5ch", "raw_spectral_5ch", "topk_6ch")
SINGLE_GROUP_ORDER = (
    "add_ndvi",
    "add_ndre",
    "add_nir",
    "add_rededge",
    "add_evi2",
    "add_gndvi",
    "add_re_bsi_proxy",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def read_progress(results_path: Path) -> dict | None:
    if not results_path.is_file():
        return None
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    normalized = [{key.strip(): value for key, value in row.items()} for row in rows]
    last = normalized[-1]
    best = max(normalized, key=lambda row: float(row.get("metrics/mAP50-95(M)", 0.0) or 0.0))
    return {
        "epochs_recorded": len(normalized),
        "last_epoch": int(float(last["epoch"])),
        "last_mask_map50_95": float(last.get("metrics/mAP50-95(M)", 0.0) or 0.0),
        "best_epoch": int(float(best["epoch"])),
        "best_mask_map50_95": float(best.get("metrics/mAP50-95(M)", 0.0) or 0.0),
        "best_mask_map50": float(best.get("metrics/mAP50(M)", 0.0) or 0.0),
        "best_mask_precision": float(best.get("metrics/precision(M)", 0.0) or 0.0),
        "best_mask_recall": float(best.get("metrics/recall(M)", 0.0) or 0.0),
    }


def experiment_name(group: str) -> str:
    return f"v4_ablate_{group}_n_seed0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("group", "single"), default="group")
    parser.add_argument("--groups", nargs="+", choices=GROUP_ORDER + SINGLE_GROUP_ORDER, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    default_groups = GROUP_ORDER if cli.phase == "group" else SINGLE_GROUP_ORDER
    groups = tuple(cli.groups) if cli.groups else default_groups
    log_dir = LOG_DIR if cli.phase == "group" else ROOT / "logs" / "v4_single_feature_ablations"
    status_path = STATUS_PATH if cli.phase == "group" else ROOT / "logs" / "v4_single_feature_ablation.status.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    state = {
        "version": "data_v4_feature_ablation_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": started,
        "updated_at": started,
        "phase": cli.phase,
        "groups": list(groups),
        "completed_groups": [],
        "current_group": None,
        "current_process_pid": None,
        "epochs_per_group": cli.epochs,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "progress": {},
    }
    write_json_atomic(status_path, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        for group in groups:
            name = experiment_name(group)
            run_dir = PROJECT / name
            results_path = run_dir / "results.csv"
            progress = read_progress(results_path)
            if progress and progress["last_epoch"] >= cli.epochs and (run_dir / "weights" / "best.pt").is_file():
                state["completed_groups"].append(group)
                state["progress"][group] = progress
                state["updated_at"] = now_iso()
                write_json_atomic(status_path, state)
                continue

            command = [
                sys.executable,
                str(ROOT / "train_feature_ablation.py"),
                "--group",
                group,
                "--name",
                name,
                "--epochs",
                str(cli.epochs),
                "--imgsz",
                str(cli.imgsz),
                "--batch",
                str(cli.batch),
                "--device",
                cli.device,
            ]
            last_checkpoint = run_dir / "weights" / "last.pt"
            if progress and progress["last_epoch"] < cli.epochs and last_checkpoint.is_file():
                command.extend(("--resume", str(last_checkpoint)))
            elif run_dir.exists():
                raise RuntimeError(
                    f"Incomplete run directory has no resumable checkpoint: {run_dir}. "
                    "Move it aside before restarting."
                )

            log_path = log_dir / f"{group}.log"
            state["current_group"] = group
            state["current_log"] = str(log_path)
            state["updated_at"] = now_iso()
            write_json_atomic(status_path, state)
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
                write_json_atomic(status_path, state)
                while process.poll() is None:
                    progress = read_progress(results_path)
                    if progress:
                        state["progress"][group] = progress
                    state["updated_at"] = now_iso()
                    write_json_atomic(status_path, state)
                    time.sleep(max(2.0, cli.poll_seconds))
                return_code = process.returncode
            progress = read_progress(results_path)
            if progress:
                state["progress"][group] = progress
            if return_code != 0:
                raise RuntimeError(f"Group {group} failed with exit code {return_code}; see {log_path}")
            if not progress or progress["last_epoch"] < cli.epochs:
                raise RuntimeError(f"Group {group} exited without {cli.epochs} recorded epochs")
            state["completed_groups"].append(group)
            state["current_process_pid"] = None
            state["updated_at"] = now_iso()
            write_json_atomic(status_path, state)

        state["state"] = "completed"
        state["current_group"] = None
        state["current_process_pid"] = None
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        write_json_atomic(status_path, state)
        summary_script = "summarize_feature_ablations.py" if cli.phase == "group" else "summarize_single_feature_ablations.py"
        subprocess.run(
            [sys.executable, str(ROOT / summary_script)],
            cwd=ROOT,
            check=True,
            env=environment,
        )
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        write_json_atomic(status_path, state)
        raise


if __name__ == "__main__":
    main()
