"""Evaluate a locked sparse-context candidate against production exactly once."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
V20_SUMMARY = ROOT / "logs" / "v20_sparse_context_multiseed.summary.json"
INCUMBENT = ROOT / "runs" / "segment" / "land_seg" / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
EXPECTED_INCUMBENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
FEATURE_ROOT = ROOT / "datasets" / "Data_v4_candidates"
MANIFEST = FEATURE_ROOT / "manifest.json"
GROUPS_CONFIG = ROOT / "configs" / "data_v8_yolo11_early_fusion_groups.json"
DATA = ROOT / "datasets" / "Data_v3" / "data_5ch_v3.yaml"
TEST_ROOT = ROOT / "datasets" / "Data_v3" / "test"
OUTPUT = ROOT / "runs" / "final_evaluation" / "data_v20_locked_test"
GUARD = ROOT / "logs" / "data_v20_final_test_guard.json"
STATUS = ROOT / "logs" / "data_v20_final_test.status.json"
LOG = ROOT / "logs" / "data_v20_final_test.log"
LOCKED_CONFIDENCE = 0.371
EXPERIMENT_LABEL = "V20"
ARTIFACT_PREFIX = "data_v20"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
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
    if len(images) != 30 or len(labels) != 30 or len(features) != 30:
        raise RuntimeError(f"Expected 30 test images/labels/features, got {len(images)}/{len(labels)}/{len(features)}")
    if image_stems != {path.stem for path in labels} or image_stems != {path.stem for path in features}:
        raise RuntimeError("Test image, label, and feature stems do not match")
    return {
        "test_images": 30,
        "test_labels": 30,
        "test_features": 30,
        "stem_signature": hashlib.sha256("\n".join(sorted(image_stems)).encode("utf-8")).hexdigest(),
        "labels_sha256": labels_sha256(labels),
    }


def validation_command(checkpoint: Path, output: Path) -> list[str]:
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


def fixed_command(checkpoint: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "error_analysis_multichannel.py"),
        "--checkpoint", str(checkpoint),
        "--data", str(DATA),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--output", str(output),
        "--imgsz", "1024",
        "--conf", str(LOCKED_CONFIDENCE),
        "--device", "0",
        "--no-overlap-mask",
        "--split", "test",
        "--allow-test",
    ]


def run_logged(command: list[str]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    with LOG.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"Final-test command failed ({result.returncode}); see {LOG}")


def fixed_totals(path: Path) -> dict[str, float | int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 per-image fixed-threshold rows, got {len(rows)}")
    tp = sum(int(float(row["mask_tp"])) for row in rows)
    fp = sum(int(float(row["mask_fp"])) for row in rows)
    fn = sum(int(float(row["mask_fn"])) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"images": 30, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def upgrade_decision(metric_deltas: dict[str, float], fixed_deltas: dict[str, float]) -> bool:
    return bool(
        metric_deltas["metrics/mAP50-95(M)"] > 0.002
        and metric_deltas["metrics/mAP50(M)"] >= -0.002
        and metric_deltas["metrics/recall(M)"] >= -0.01
        and fixed_deltas["f1"] >= 0.0
        and fixed_deltas["precision"] >= -0.01
    )


def extract_frozen_candidate(summary: dict[str, object]) -> dict[str, object] | None:
    frozen = summary.get("frozen_candidate")
    if isinstance(frozen, dict):
        return frozen
    checkpoint = summary.get("candidate_checkpoint")
    checkpoint_hash = summary.get("candidate_checkpoint_sha256")
    if not checkpoint or not checkpoint_hash:
        return None
    seed = None
    for row in summary.get("rows", []):
        if isinstance(row, dict) and row.get("candidate_checkpoint") == checkpoint:
            seed = row.get("seed")
            break
    if seed is None:
        return None
    return {
        "seed": seed,
        "candidate_checkpoint": checkpoint,
        "candidate_checkpoint_sha256": checkpoint_hash,
    }


def main() -> int:
    for path in (V20_SUMMARY, INCUMBENT, FEATURE_ROOT, MANIFEST, GROUPS_CONFIG, DATA, TEST_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = json.loads(V20_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("gate_passed") is not True or summary.get("held_out_test_used") is not False:
        raise RuntimeError(f"{EXPERIMENT_LABEL} validation gate did not authorize final test")
    frozen = extract_frozen_candidate(summary)
    if not frozen:
        raise RuntimeError(f"{EXPERIMENT_LABEL} summary has no frozen candidate")
    candidate = Path(frozen["candidate_checkpoint"])
    candidate_hash = sha256(candidate)
    if candidate_hash != frozen["candidate_checkpoint_sha256"]:
        raise RuntimeError(f"Frozen {EXPERIMENT_LABEL} candidate hash mismatch")
    incumbent_hash = sha256(INCUMBENT)
    if incumbent_hash != EXPECTED_INCUMBENT_SHA256:
        raise RuntimeError("Production incumbent hash changed before final test")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("split_counts", {}).get("test") != 30:
        raise RuntimeError("Test feature manifest does not declare 30 arrays")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" not in data_text or "channels: 5" not in data_text:
        raise RuntimeError("Final-test YAML contract drifted")
    contract = test_contract()
    if GUARD.is_file():
        guard = json.loads(GUARD.read_text(encoding="utf-8"))
        if guard.get("state") == "completed" and (OUTPUT / "decision.json").is_file():
            print(f"{EXPERIMENT_LABEL} final test already completed; returning immutable result")
            return 0
        raise RuntimeError(f"{EXPERIMENT_LABEL} final test was already attempted and cannot be repeated automatically")

    started = now_iso()
    guard = {
        "version": f"{ARTIFACT_PREFIX}_final_test_guard_v1",
        "state": "evaluation_started",
        "started_at": started,
        "candidate_test_evaluation_count": 1,
        "candidate_selected_before_test": True,
        "candidate": str(candidate.resolve()),
        "candidate_sha256": candidate_hash,
        "incumbent": str(INCUMBENT.resolve()),
        "incumbent_sha256": incumbent_hash,
        "locked_confidence": LOCKED_CONFIDENCE,
        "test_contract": contract,
        "test_results_may_be_used_for_further_tuning": False,
        "deployment_write_allowed": False,
    }
    write_json_atomic(GUARD, guard)
    state = {
        "version": f"{ARTIFACT_PREFIX}_final_test_status_v1",
        "state": "evaluating_locked_pair_once",
        "started_at": started,
        "updated_at": started,
        "deployment_write_allowed": False,
    }
    write_json_atomic(STATUS, state)
    incumbent_output = OUTPUT / "incumbent"
    candidate_output = OUTPUT / "candidate"
    incumbent_fixed_output = OUTPUT / "incumbent_conf0371"
    candidate_fixed_output = OUTPUT / "candidate_conf0371"
    try:
        run_logged(validation_command(INCUMBENT, incumbent_output))
        run_logged(fixed_command(INCUMBENT, incumbent_fixed_output))
        state.update(state="evaluating_locked_candidate", updated_at=now_iso())
        write_json_atomic(STATUS, state)
        run_logged(validation_command(candidate, candidate_output))
        run_logged(fixed_command(candidate, candidate_fixed_output))
        incumbent_metrics = json.loads((incumbent_output / "metrics.json").read_text(encoding="utf-8"))
        candidate_metrics = json.loads((candidate_output / "metrics.json").read_text(encoding="utf-8"))
        keys = (
            "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)",
            "metrics/mAP50-95(M)", "fitness",
        )
        metric_deltas = {key: float(candidate_metrics[key]) - float(incumbent_metrics[key]) for key in keys}
        incumbent_fixed = fixed_totals(incumbent_fixed_output / "per_image_metrics.csv")
        candidate_fixed = fixed_totals(candidate_fixed_output / "per_image_metrics.csv")
        fixed_deltas = {
            key: float(candidate_fixed[key]) - float(incumbent_fixed[key])
            for key in ("precision", "recall", "f1")
        }
        for key in ("tp", "fp", "fn"):
            fixed_deltas[key] = int(candidate_fixed[key]) - int(incumbent_fixed[key])
        eligible = upgrade_decision(metric_deltas, fixed_deltas)
        completed = now_iso()
        decision = {
            "version": f"{ARTIFACT_PREFIX}_final_test_decision_v1",
            "completed_at": completed,
            "candidate_selected_before_test": True,
            "candidate_seed": frozen["seed"],
            "candidate": str(candidate.resolve()),
            "candidate_sha256": candidate_hash,
            "incumbent": str(INCUMBENT.resolve()),
            "incumbent_sha256": incumbent_hash,
            "test_contract": contract,
            "aggregate_incumbent": incumbent_metrics,
            "aggregate_candidate": candidate_metrics,
            "candidate_minus_incumbent": metric_deltas,
            "fixed_confidence": LOCKED_CONFIDENCE,
            "fixed_incumbent": incumbent_fixed,
            "fixed_candidate": candidate_fixed,
            "fixed_candidate_minus_incumbent": fixed_deltas,
            "upgrade_rule": {
                "mask_map50_95_delta_gt": 0.002,
                "mask_map50_delta_gte": -0.002,
                "mask_recall_delta_gte": -0.01,
                "fixed_f1_delta_gte": 0.0,
                "fixed_precision_delta_gte": -0.01,
            },
            "upgrade_eligible": eligible,
            "decision": "authorized_for_deployment" if eligible else "keep_existing_deployment",
            "further_tuning_on_this_test_forbidden": True,
            "deployment_weight_modified": False,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json_atomic(OUTPUT / "decision.json", decision)
        guard.update(state="completed", completed_at=completed, decision=str((OUTPUT / "decision.json").resolve()))
        write_json_atomic(GUARD, guard)
        state.update(
            state="completed",
            completed_at=completed,
            updated_at=completed,
            upgrade_eligible=eligible,
            decision=str((OUTPUT / "decision.json").resolve()),
        )
        write_json_atomic(STATUS, state)
        print(json.dumps({"upgrade_eligible": eligible, "decision": decision["decision"]}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        failed = now_iso()
        guard.update(state="failed_after_evaluation_started", failed_at=failed, error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(GUARD, guard)
        state.update(state="failed", failed_at=failed, updated_at=failed, error=guard["error"])
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
