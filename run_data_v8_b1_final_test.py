"""Evaluate the locked B1 candidate and fixed incumbent on test exactly once."""

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
CANDIDATE = ROOT / "runs" / "segment" / "land_seg" / "v8_b1_boundary_aug_60ep_seed0_confirm_v1" / "weights" / "best.pt"
INCUMBENT = ROOT / "runs" / "segment" / "land_seg" / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
EXPECTED_CANDIDATE_SHA256 = "a730aaeac46abb5af2832f737434652dcefcc89ebce2ef4b0376275dce5c8c1d"
EXPECTED_INCUMBENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
FEATURE_ROOT = ROOT / "datasets" / "Data_v4_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "data_v8_yolo11_early_fusion_groups.json"
DATA = ROOT / "datasets" / "Data_v3" / "data_5ch_v3.yaml"
TEST_ROOT = ROOT / "datasets" / "Data_v3" / "test"
OUTPUT_ROOT = ROOT / "runs" / "final_evaluation" / "data_v8_b1_locked_test"
INCUMBENT_OUTPUT = OUTPUT_ROOT / "incumbent"
CANDIDATE_OUTPUT = OUTPUT_ROOT / "candidate"
DECISION_PATH = OUTPUT_ROOT / "decision.json"
GUARD_PATH = ROOT / "logs" / "data_v8_b1_final_test_guard.json"
STATUS_PATH = ROOT / "logs" / "data_v8_b1_final_test.status.json"
EVAL_LOG = ROOT / "logs" / "data_v8_b1_final_test.log"
PRIMARY_GAIN_THRESHOLD = 0.002
MAP50_FLOOR = -0.002
RECALL_FLOOR = -0.01


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_contract() -> dict[str, object]:
    images = sorted((TEST_ROOT / "images").glob("*.JPG"))
    labels = sorted((TEST_ROOT / "labels").glob("*.txt"))
    features = sorted((FEATURE_ROOT / "test" / "features").glob("*.npy"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    feature_stems = {path.stem for path in features}
    if len(images) != 30 or len(labels) != 30 or len(features) != 30:
        raise RuntimeError(
            f"Expected 30 test images/labels/features, got {len(images)}/{len(labels)}/{len(features)}"
        )
    if image_stems != label_stems or image_stems != feature_stems:
        raise RuntimeError("Test image, label, and feature stems do not match exactly")
    signature = hashlib.sha256("\n".join(sorted(image_stems)).encode("utf-8")).hexdigest()
    return {
        "test_images": len(images),
        "test_labels": len(labels),
        "test_features": len(features),
        "stem_signature": signature,
        "labels_sha256": labels_sha256(labels),
    }


def run_logged(command: list[str], environment: dict[str, str]) -> None:
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_LOG.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with exit code {result.returncode}; see {EVAL_LOG}")


def evaluation_command(checkpoint: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "validate_multichannel.py"),
        "--checkpoint", str(checkpoint),
        "--group", "c0_current_indices_early",
        "--data", str(DATA),
        "--output", str(output),
        "--feature-root", str(FEATURE_ROOT),
        "--manifest", str(MANIFEST),
        "--groups-config", str(GROUPS_CONFIG),
        "--split", "test",
        "--imgsz", "1024",
        "--batch", "1",
        "--workers", "0",
        "--device", "0",
    ]


def main() -> int:
    for path in (CANDIDATE, INCUMBENT, FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DATA, TEST_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    candidate_hash = sha256_file(CANDIDATE)
    incumbent_hash = sha256_file(INCUMBENT)
    if candidate_hash != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"Locked candidate hash mismatch: {candidate_hash}")
    if incumbent_hash != EXPECTED_INCUMBENT_SHA256:
        raise RuntimeError(f"Fixed incumbent hash mismatch: {incumbent_hash}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("split_counts", {}).get("test") != 30:
        raise RuntimeError("Feature manifest does not declare exactly 30 test arrays")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" not in data_text or "channels: 5" not in data_text:
        raise RuntimeError("Final test YAML must expose test and declare five channels")
    contract = test_contract()

    if GUARD_PATH.is_file():
        guard = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
        if guard.get("state") == "completed" and DECISION_PATH.is_file():
            print("Locked B1 candidate was already evaluated once; returning the immutable result.")
            return 0
        raise RuntimeError(f"Final test was already attempted and cannot be repeated automatically: {GUARD_PATH}")

    started = now_iso()
    guard = {
        "version": "data_v8_b1_final_test_guard_v1",
        "state": "evaluation_started",
        "started_at": started,
        "candidate_test_evaluation_count": 1,
        "incumbent_reference_evaluation_count": 1,
        "candidate": str(CANDIDATE.resolve()),
        "candidate_sha256": candidate_hash,
        "incumbent": str(INCUMBENT.resolve()),
        "incumbent_sha256": incumbent_hash,
        "candidate_selected_before_test": True,
        "candidate_selection_basis": "best valid Mask mAP50-95 among the three B1 60-epoch seeds",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "test_contract": contract,
        "upgrade_rule": {
            "mask_map50_95_delta_gt": PRIMARY_GAIN_THRESHOLD,
            "mask_map50_delta_gte": MAP50_FLOOR,
            "mask_recall_delta_gte": RECALL_FLOOR,
        },
        "test_results_may_be_used_for_further_tuning": False,
        "deployment_write_allowed": False,
        "report_generated": False,
    }
    write_json_atomic(GUARD_PATH, guard)
    state = {
        "version": "data_v8_b1_final_test_status_v1",
        "state": "evaluating_locked_pair_once",
        "started_at": started,
        "updated_at": started,
        "guard": str(GUARD_PATH.resolve()),
        "deployment_write_allowed": False,
        "report_generated": False,
    }
    write_json_atomic(STATUS_PATH, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    try:
        run_logged(evaluation_command(INCUMBENT, INCUMBENT_OUTPUT), environment)
        state["state"] = "evaluating_locked_candidate"
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS_PATH, state)
        run_logged(evaluation_command(CANDIDATE, CANDIDATE_OUTPUT), environment)

        incumbent_metrics = json.loads((INCUMBENT_OUTPUT / "metrics.json").read_text(encoding="utf-8"))
        candidate_metrics = json.loads((CANDIDATE_OUTPUT / "metrics.json").read_text(encoding="utf-8"))
        keys = (
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)", "fitness",
        )
        missing = [key for key in keys if key not in incumbent_metrics or key not in candidate_metrics]
        if missing:
            raise KeyError(f"Missing final test metrics: {missing}")
        deltas = {key: float(candidate_metrics[key]) - float(incumbent_metrics[key]) for key in keys}
        eligible = (
            deltas["metrics/mAP50-95(M)"] > PRIMARY_GAIN_THRESHOLD
            and deltas["metrics/mAP50(M)"] >= MAP50_FLOOR
            and deltas["metrics/recall(M)"] >= RECALL_FLOOR
        )
        completed = now_iso()
        decision = {
            "version": "data_v8_b1_final_test_decision_v1",
            "completed_at": completed,
            "test_evaluation_count_for_candidate": 1,
            "candidate_selected_before_test": True,
            "candidate_sha256": candidate_hash,
            "incumbent_sha256": incumbent_hash,
            "test_contract": contract,
            "incumbent": incumbent_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_incumbent": deltas,
            "upgrade_rule": guard["upgrade_rule"],
            "upgrade_eligible": eligible,
            "decision": "eligible_pending_explicit_deployment_approval" if eligible else "keep_existing_deployment",
            "further_tuning_on_this_test_forbidden": True,
            "deployment_weight_modified": False,
            "report_generated": False,
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        write_json_atomic(DECISION_PATH, decision)
        guard["state"] = "completed"
        guard["completed_at"] = completed
        guard["decision"] = str(DECISION_PATH.resolve())
        write_json_atomic(GUARD_PATH, guard)
        state["state"] = "completed"
        state["completed_at"] = completed
        state["updated_at"] = completed
        state["decision"] = str(DECISION_PATH.resolve())
        state["upgrade_eligible"] = eligible
        write_json_atomic(STATUS_PATH, state)
        print(json.dumps({"upgrade_eligible": eligible, "decision": decision["decision"]}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        failed = now_iso()
        guard["state"] = "failed_after_evaluation_started"
        guard["failed_at"] = failed
        guard["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(GUARD_PATH, guard)
        state["state"] = "failed"
        state["failed_at"] = failed
        state["updated_at"] = failed
        state["error"] = guard["error"]
        write_json_atomic(STATUS_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
