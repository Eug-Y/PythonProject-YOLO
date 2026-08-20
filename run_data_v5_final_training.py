"""Train and freeze the formal Data_v5 RGB+MCARI candidate without reading test."""

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


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
FEATURE_ROOT = ROOT / "datasets" / "Data_v5_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "feature_ablation_groups_v5.json"
GROUP = "add_mcari"
RUN_NAME = "v5_final_rgb_mcari_1024_seed0_v1"
RUN_DIR = PROJECT / RUN_NAME
STATUS_PATH = ROOT / "logs" / "v5_final_mcari.status.json"
TRAIN_LOG = ROOT / "logs" / "v5_final_mcari.train.log"
COMPLETION_MARKER = RUN_DIR / "training_complete.json"
REPORT_DIR = ROOT / "reports" / "Data_v5_final_training"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def update_timing(state: dict, started_monotonic: float) -> None:
    elapsed = max(time.monotonic() - started_monotonic, 0.0)
    progress = state.get("progress") or {}
    recorded = int(progress.get("epochs_recorded", 0))
    maximum = int(state["max_epochs"])
    eta = elapsed / recorded * (maximum - recorded) if recorded else None
    state["overall"] = {
        "epochs_recorded": recorded,
        "max_epochs": maximum,
        "percent_of_max": round(recorded / maximum * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds_to_max": round(eta, 1) if eta is not None else None,
    }


def write_completion_report(marker: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    progress = marker["progress"]
    stopped = "是" if marker["stopped_early"] else "否"
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v5 正式训练</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:900px;margin:auto;background:#fff;padding:28px;border-radius:14px}}dt{{font-weight:700;margin-top:12px}}dd{{margin:4px 0}}</style></head><body><main><h1>Data_v5 RGB + MCARI 正式训练</h1><dl>
<dt>输入通道</dt><dd>R, G, B, MCARI</dd><dt>实际轮次</dt><dd>{progress['epochs_recorded']}</dd><dt>最佳轮次</dt><dd>{progress['best_epoch']}</dd><dt>Mask mAP50-95</dt><dd>{progress['best_mask_map50_95']:.5f}</dd><dt>提前停止</dt><dd>{stopped}</dd><dt>独立 test</dt><dd>未读取</dd>
</dl></main></body></html>"""
    (REPORT_DIR / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    cli = parse_args()
    for required in (FEATURE_ROOT, MANIFEST, GROUPS_CONFIG):
        if not required.exists():
            raise FileNotFoundError(required)
    started_monotonic = time.monotonic()
    state = {
        "version": "data_v5_final_mcari_training_status_v1",
        "state": "training",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run_name": RUN_NAME,
        "group": GROUP,
        "channels": ["R", "G", "B", "MCARI"],
        "max_epochs": cli.epochs,
        "patience": cli.patience,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "training_splits": ["train", "valid"],
        "held_out_test_used": False,
        "candidate_test_evaluation_count": 0,
        "progress": None,
        "overall": {},
    }
    update_timing(state, started_monotonic)
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    try:
        if COMPLETION_MARKER.is_file() and (RUN_DIR / "weights" / "best.pt").is_file():
            marker = json.loads(COMPLETION_MARKER.read_text(encoding="utf-8"))
            state.update(marker)
            state["state"] = "completed"
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS_PATH, state)
            write_completion_report(marker)
            return

        progress = read_progress(RUN_DIR / "results.csv")
        command = [
            sys.executable, str(ROOT / "train_feature_ablation.py"),
            "--group", GROUP,
            "--groups-config", str(GROUPS_CONFIG),
            "--manifest", str(MANIFEST),
            "--feature-root", str(FEATURE_ROOT),
            "--name", RUN_NAME,
            "--epochs", str(cli.epochs),
            "--patience", str(cli.patience),
            "--warmup-epochs", "10",
            "--imgsz", str(cli.imgsz),
            "--batch", str(cli.batch),
            "--device", cli.device,
            "--seed", "0",
            "--no-deterministic",
        ]
        checkpoint = RUN_DIR / "weights" / "last.pt"
        if progress and checkpoint.is_file():
            command.extend(("--resume", str(checkpoint)))
        elif RUN_DIR.exists():
            raise RuntimeError(f"Incomplete formal run has no resumable checkpoint: {RUN_DIR}")

        TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TRAIN_LOG.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=environment)
            state["training_process_pid"] = process.pid
            write_json_atomic(STATUS_PATH, state)
            while process.poll() is None:
                progress = read_progress(RUN_DIR / "results.csv")
                if progress:
                    state["progress"] = progress
                state["updated_at"] = now_iso()
                update_timing(state, started_monotonic)
                write_json_atomic(STATUS_PATH, state)
                time.sleep(max(2.0, cli.poll_seconds))
            return_code = process.returncode

        progress = read_progress(RUN_DIR / "results.csv")
        best = RUN_DIR / "weights" / "best.pt"
        last = RUN_DIR / "weights" / "last.pt"
        if return_code != 0:
            raise RuntimeError(f"Formal training failed with exit code {return_code}; see {TRAIN_LOG}")
        if not progress or not best.is_file() or not last.is_file():
            raise RuntimeError("Formal training exited without complete results and checkpoints")

        marker = {
            "version": "data_v5_final_mcari_training_complete_v1",
            "completed_at": now_iso(),
            "run_name": RUN_NAME,
            "group": GROUP,
            "channels": ["R", "G", "B", "MCARI"],
            "max_epochs": cli.epochs,
            "patience": cli.patience,
            "imgsz": cli.imgsz,
            "batch": cli.batch,
            "stopped_early": progress["last_epoch"] < cli.epochs,
            "progress": progress,
            "best_checkpoint": str(best.resolve()),
            "last_checkpoint": str(last.resolve()),
            "held_out_test_used": False,
            "candidate_test_evaluation_count": 0,
        }
        COMPLETION_MARKER.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        write_completion_report(marker)
        state["state"] = "completed"
        state["training_process_pid"] = None
        state["progress"] = progress
        state["stopped_early"] = marker["stopped_early"]
        state["frozen_checkpoint"] = marker["best_checkpoint"]
        state["completed_at"] = marker["completed_at"]
        state["updated_at"] = state["completed_at"]
        update_timing(state, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        update_timing(state, started_monotonic)
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    main()
