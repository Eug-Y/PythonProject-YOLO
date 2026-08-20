"""Train, freeze, test once, and compare the final Data_v4 high-resolution candidate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
RUN_NAME = "v4_final_rgb_re_bsi_proxy_1024_seed0_v1"
RUN_DIR = ROOT / "runs" / "segment" / "land_seg" / RUN_NAME
STATUS_PATH = ROOT / "logs" / "v4_final_highres.status.json"
TRAIN_LOG = ROOT / "logs" / "v4_final_highres.train.log"
EVAL_LOG = ROOT / "logs" / "v4_final_highres.test.log"
TEST_OUTPUT = ROOT / "reports" / "Data_v4_final_candidate_test"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
        return subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment).returncode


def main() -> None:
    cli = parse_args()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    state = {
        "version": "data_v4_final_highres_status_v1",
        "state": "training",
        "runner_pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run_name": RUN_NAME,
        "epochs": cli.epochs,
        "imgsz": cli.imgsz,
        "batch": cli.batch,
        "training_splits": ["train", "valid"],
        "test_visible_during_training": False,
        "candidate_test_evaluation_count": 0,
    }
    write_json_atomic(STATUS_PATH, state)
    try:
        results_path = RUN_DIR / "results.csv"
        progress = read_progress(results_path)
        if not (progress and progress["last_epoch"] >= cli.epochs and (RUN_DIR / "weights" / "best.pt").is_file()):
            command = [
                sys.executable,
                str(ROOT / "train_feature_ablation.py"),
                "--group",
                "add_re_bsi_proxy",
                "--name",
                RUN_NAME,
                "--epochs",
                str(cli.epochs),
                "--patience",
                "50",
                "--warmup-epochs",
                "10",
                "--imgsz",
                str(cli.imgsz),
                "--batch",
                str(cli.batch),
                "--device",
                cli.device,
                "--seed",
                "0",
                "--no-deterministic",
            ]
            checkpoint = RUN_DIR / "weights" / "last.pt"
            if progress and checkpoint.is_file():
                command.extend(("--resume", str(checkpoint)))
            elif RUN_DIR.exists():
                raise RuntimeError(f"Incomplete final run is not resumable: {RUN_DIR}")
            TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
            with TRAIN_LOG.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment
                )
                state["training_process_pid"] = process.pid
                write_json_atomic(STATUS_PATH, state)
                while process.poll() is None:
                    progress = read_progress(results_path)
                    if progress:
                        state["progress"] = progress
                    state["updated_at"] = now_iso()
                    write_json_atomic(STATUS_PATH, state)
                    time.sleep(max(2.0, cli.poll_seconds))
                if process.returncode != 0:
                    raise RuntimeError(f"Final training failed with exit code {process.returncode}")
        best = RUN_DIR / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError(f"Frozen best checkpoint is missing: {best}")

        state["state"] = "evaluating_test_once"
        state["training_process_pid"] = None
        state["frozen_checkpoint"] = str(best)
        state["test_started_at"] = now_iso()
        state["candidate_test_evaluation_count"] = 1
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS_PATH, state)
        validation_command = [
            sys.executable,
            str(ROOT / "validate_multichannel.py"),
            "--checkpoint",
            str(best),
            "--group",
            "add_re_bsi_proxy",
            "--data",
            str(ROOT / "datasets" / "Data_v4_candidates" / "data_v4_final_4ch.yaml"),
            "--output",
            str(TEST_OUTPUT),
            "--split",
            "test",
            "--imgsz",
            str(cli.imgsz),
            "--batch",
            "1",
            "--workers",
            "0",
            "--device",
            cli.device,
        ]
        if run_logged(validation_command, EVAL_LOG, environment) != 0:
            raise RuntimeError(f"Frozen candidate test evaluation failed; see {EVAL_LOG}")
        if run_logged([sys.executable, str(ROOT / "compare_final_candidate.py")], EVAL_LOG, environment) != 0:
            raise RuntimeError(f"Final comparison failed; see {EVAL_LOG}")

        state["state"] = "completed"
        state["completed_at"] = now_iso()
        state["updated_at"] = state["completed_at"]
        state["comparison"] = str(ROOT / "reports" / "Data_v4_final_model_comparison" / "summary.json")
        write_json_atomic(STATUS_PATH, state)
    except BaseException as exc:
        state["state"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = now_iso()
        state["updated_at"] = state["failed_at"]
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    main()

