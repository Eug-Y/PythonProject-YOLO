"""Evaluate the frozen Data_v5 candidate on held-out test exactly once."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "segment" / "land_seg" / "v5_final_rgb_mcari_1024_seed0_v1"
CHECKPOINT = RUN_DIR / "weights" / "best.pt"
TRAINING_MARKER = RUN_DIR / "training_complete.json"
FEATURE_ROOT = ROOT / "datasets" / "Data_v5_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "feature_ablation_groups_v5.json"
DATA_YAML = ROOT / "configs" / "data_v5_final_4ch.yaml"
TEST_OUTPUT = ROOT / "reports" / "Data_v5_final_candidate_test"
COMPARISON_OUTPUT = ROOT / "reports" / "Data_v5_final_model_comparison"
GUARD_PATH = TEST_OUTPUT / "evaluation_guard.json"
STATUS_PATH = ROOT / "logs" / "v5_final_evaluation.status.json"
EVAL_LOG = ROOT / "logs" / "v5_final_evaluation.log"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command: list[str], environment: dict[str, str]) -> None:
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_LOG.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}; see {EVAL_LOG}")


def main() -> None:
    required = (CHECKPOINT, TRAINING_MARKER, MANIFEST, GROUPS_CONFIG, DATA_YAML)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen evaluation inputs: {missing}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if int(manifest.get("split_counts", {}).get("test", 0)) != 30:
        raise RuntimeError("Data_v5 manifest must contain exactly 30 test feature arrays before evaluation")
    feature_count = len(list((FEATURE_ROOT / "test" / "features").glob("*.npy")))
    if feature_count != 30:
        raise RuntimeError(f"Expected 30 Data_v5 test feature files, found {feature_count}")

    if GUARD_PATH.is_file():
        guard = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
        if guard.get("state") == "completed" and (COMPARISON_OUTPUT / "summary.json").is_file():
            print("Frozen candidate was already evaluated once; returning existing result.")
            return
        raise RuntimeError(f"Test evaluation was already attempted and cannot be repeated: {guard}")

    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(CHECKPOINT)
    guard = {
        "version": "data_v5_test_evaluation_guard_v1",
        "state": "evaluation_started",
        "evaluation_count": 1,
        "started_at": now_iso(),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "channels": ["R", "G", "B", "MCARI"],
        "test_images": 30,
    }
    write_json_atomic(GUARD_PATH, guard)
    state = {
        "version": "data_v5_final_evaluation_status_v1",
        "state": "evaluating_test_once",
        "started_at": guard["started_at"],
        "updated_at": guard["started_at"],
        "candidate_test_evaluation_count": 1,
        "checkpoint_sha256": checkpoint_hash,
    }
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    try:
        run_logged([
            sys.executable, str(ROOT / "validate_multichannel.py"),
            "--checkpoint", str(CHECKPOINT),
            "--group", "add_mcari",
            "--data", str(DATA_YAML),
            "--output", str(TEST_OUTPUT),
            "--feature-root", str(FEATURE_ROOT),
            "--manifest", str(MANIFEST),
            "--groups-config", str(GROUPS_CONFIG),
            "--split", "test",
            "--imgsz", "1024",
            "--batch", "1",
            "--workers", "0",
            "--device", "0",
        ], environment)
        run_logged([sys.executable, str(ROOT / "compare_data_v5_final.py")], environment)
        guard["state"] = "completed"
        guard["completed_at"] = now_iso()
        guard["metrics"] = str((TEST_OUTPUT / "metrics.json").resolve())
        guard["comparison"] = str((COMPARISON_OUTPUT / "summary.json").resolve())
        write_json_atomic(GUARD_PATH, guard)
        state["state"] = "completed"
        state["completed_at"] = guard["completed_at"]
        state["updated_at"] = state["completed_at"]
        state["comparison"] = guard["comparison"]
        write_json_atomic(STATUS_PATH, state)
    except BaseException as exc:
        guard["state"] = "failed_after_evaluation_started"
        guard["failed_at"] = now_iso()
        guard["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(GUARD_PATH, guard)
        state["state"] = "failed"
        state["failed_at"] = guard["failed_at"]
        state["updated_at"] = state["failed_at"]
        state["error"] = guard["error"]
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    main()
