"""Run V22 train-only high-overlap replay confirmation against V19."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import run_data_v20_sparse_context_multiseed as base
from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "datasets" / "Data_v22_sparse_context_dense_replay_train"
DATASET_MANIFEST = DATASET_ROOT / "manifest.json"
DATA = ROOT / "configs" / "generated" / "data_v22_sparse_context_dense_replay_5ch.yaml"
V21_SUMMARY = ROOT / "logs" / "v21_sparse_context_half_multiseed.summary.json"
STATUS = ROOT / "logs" / "v22_dense_replay_multiseed.status.json"
SUMMARY = ROOT / "logs" / "v22_dense_replay_multiseed.summary.json"
LOG_DIR = ROOT / "logs" / "v22_dense_replay_multiseed"
OUTPUT = ROOT / "reports" / "Data_v22_dense_replay_multiseed"


def candidate_name(seed: int) -> str:
    return f"v22_yolo11_5ch_sparsectx_dense_replay_seed{seed}_v1"


def protocol() -> dict[str, object]:
    return {
        "version": "data_v22_dense_replay_multiseed_protocol_v1",
        "preregistered_at": now_iso(),
        "reference": "same-seed V19B checkpoint",
        "prerequisite": "V21 retained sparse FP suppression but failed aggregate/fixed-F1 gates and degraded high-overlap scenes",
        "single_training_change": "add 10 train-only top-quartile-overlap original-image replays from 10 flight groups to V21",
        "architecture": "YOLO11n-seg + 3x CoordAtt + 3x ECA",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "training_overlap_mask": False,
        "validation_overlap_mask": False,
        "augmentation": {"mosaic": 0.5, "mixup": 0.0, "close_mosaic": 15},
        "epochs": 60,
        "imgsz": 1024,
        "batch": 2,
        "seeds": list(base.SEEDS),
        "locked_confidence": base.LOCKED_CONFIDENCE,
        "candidate_threshold_search_allowed": False,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "seed0_gate": {
            "map_delta_min": -0.002,
            "precision_delta_min": -0.01,
            "recall_delta_min": -0.01,
            "fixed_f1_delta_min": -0.002,
            "sparse_f1_delta_min": 0.01,
            "sparse_fp_delta_max": -8,
            "high_overlap_f1_delta_min": -0.01,
        },
        "three_seed_gate": {
            "mean_map_delta_min": -0.001,
            "worst_map_delta_min": -0.005,
            "mean_fixed_f1_delta_min": -0.002,
            "worst_fixed_f1_delta_min": -0.005,
            "mean_sparse_f1_delta_min": 0.008,
            "worst_sparse_f1_delta_min": -0.01,
            "positive_sparse_seeds_min": 2,
            "mean_sparse_fp_delta_max": -5,
            "mean_high_overlap_f1_delta_min": -0.005,
            "worst_high_overlap_f1_delta_min": -0.015,
        },
        "deployment_write_allowed": False,
    }


def validate_inputs() -> str:
    for path in (
        DATASET_ROOT, DATASET_MANIFEST, DATA, V21_SUMMARY, base.V19_SUMMARY,
        base.GROUPS_CONFIG, base.BASE_MODEL, DEPLOYMENT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    v21 = json.loads(V21_SUMMARY.read_text(encoding="utf-8"))
    if v21.get("decision") != "reject_v21_retain_v19_frozen_candidate" or v21.get("held_out_test_used") is not False:
        raise RuntimeError("V21 prerequisite result drifted")
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    replayed = dataset.get("replayed", [])
    sparse_context = dataset.get("sparse_context_generated", [])
    if (
        dataset.get("version") != "data_v22_sparse_context_dense_replay_train_v1"
        or dataset.get("selection_basis_splits") != ["train"]
        or dataset.get("selection_splits") != ["train", "valid"]
        or dataset.get("validation_labels_used_for_replay_selection") is not False
        or dataset.get("held_out_split_used") is not False
        or dataset.get("source_train_images") != 220
        or dataset.get("total_train_images") != 230
        or len(replayed) != 10
        or len(sparse_context) != 10
        or dataset.get("replay_selection", {}).get("selected_groups") != 10
        or any(float(row["overlap_foreground_fraction"]) < float(dataset["replay_selection"]["threshold"]) for row in replayed)
    ):
        raise RuntimeError("V22 train-only dense replay dataset contract drifted")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text or "channels: 5" not in data_text:
        raise RuntimeError("V22 YAML exposed test or changed channels")
    for relative, expected in (("train/images", 230), ("train/labels", 230), ("train/features", 230), ("valid/features", 60)):
        if len(list((DATASET_ROOT / relative).iterdir())) != expected:
            raise RuntimeError(f"V22 artifact count drifted: {relative}")
    v19 = json.loads(base.V19_SUMMARY.read_text(encoding="utf-8"))
    if v19.get("gate_passed") is not True or v19.get("held_out_test_used") is not False:
        raise RuntimeError("V19 reference is not eligible")
    for seed in base.SEEDS:
        payload = json.loads((base.v19_eval_dir(seed) / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
        base.validate_sweep(payload, f"V19 Seed-{seed}")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V22")
    return deployment_hash


def high_overlap_delta(row: dict[str, object]) -> float:
    return float(row["slice_summary"]["high_instance_overlap"]["delta"]["f1"])


def summarize(rows: list[dict[str, object]], deployment_hash: str) -> dict[str, object]:
    seed0 = next((row for row in rows if row["seed"] == 0), None)
    seed0_passed = bool(
        seed0
        and float(seed0["map_delta"]) >= -0.002
        and float(seed0["precision_delta"]) >= -0.01
        and float(seed0["recall_delta"]) >= -0.01
        and float(seed0["fixed_f1_delta"]) >= -0.002
        and float(seed0["sparse_f1_delta"]) >= 0.01
        and int(seed0["sparse_fp_delta"]) <= -8
        and high_overlap_delta(seed0) >= -0.01
    )
    metrics: dict[str, object] = {}
    gate_passed = False
    if len(rows) == len(base.SEEDS):
        map_deltas = [float(row["map_delta"]) for row in rows]
        fixed_deltas = [float(row["fixed_f1_delta"]) for row in rows]
        sparse_f1 = [float(row["sparse_f1_delta"]) for row in rows]
        sparse_fp = [int(row["sparse_fp_delta"]) for row in rows]
        overlap_f1 = [high_overlap_delta(row) for row in rows]
        metrics = {
            "mean_map_delta": statistics.fmean(map_deltas),
            "worst_map_delta": min(map_deltas),
            "mean_fixed_f1_delta": statistics.fmean(fixed_deltas),
            "worst_fixed_f1_delta": min(fixed_deltas),
            "mean_sparse_f1_delta": statistics.fmean(sparse_f1),
            "worst_sparse_f1_delta": min(sparse_f1),
            "positive_sparse_seeds": sum(value > 0.0 for value in sparse_f1),
            "mean_sparse_fp_delta": statistics.fmean(sparse_fp),
            "mean_high_overlap_f1_delta": statistics.fmean(overlap_f1),
            "worst_high_overlap_f1_delta": min(overlap_f1),
        }
        gate_passed = bool(
            seed0_passed
            and metrics["mean_map_delta"] >= -0.001
            and metrics["worst_map_delta"] >= -0.005
            and metrics["mean_fixed_f1_delta"] >= -0.002
            and metrics["worst_fixed_f1_delta"] >= -0.005
            and metrics["mean_sparse_f1_delta"] >= 0.008
            and metrics["worst_sparse_f1_delta"] >= -0.01
            and metrics["positive_sparse_seeds"] >= 2
            and metrics["mean_sparse_fp_delta"] <= -5
            and metrics["mean_high_overlap_f1_delta"] >= -0.005
            and metrics["worst_high_overlap_f1_delta"] >= -0.015
        )
    frozen = max(rows, key=lambda row: float(row["candidate_map"])) if gate_passed else None
    return {
        "version": "data_v22_dense_replay_multiseed_summary_v1",
        "created_at": now_iso(),
        "protocol": protocol(),
        "rows": rows,
        "seed0_passed": seed0_passed,
        "gate_metrics": metrics,
        "gate_passed": gate_passed,
        "decision": (
            "freeze_v22_candidate_for_one_time_test" if gate_passed else
            "continue_v22_multiseed" if seed0_passed and len(rows) < len(base.SEEDS) else
            "reject_v22_retain_v19_frozen_candidate"
        ),
        "frozen_candidate": frozen,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_eligible": False,
        "deployment_write_allowed": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
    }


def configure_base() -> None:
    base.DATASET_ROOT = DATASET_ROOT
    base.DATASET_MANIFEST = DATASET_MANIFEST
    base.DATA = DATA
    base.STATUS = STATUS
    base.SUMMARY = SUMMARY
    base.LOG_DIR = LOG_DIR
    base.OUTPUT = OUTPUT
    base.candidate_name = candidate_name
    base.protocol = protocol
    base.validate_inputs = validate_inputs
    base.summarize = summarize


def training_command(cli, seed: int, resume=None):
    names = (
        "DATASET_ROOT", "DATASET_MANIFEST", "DATA", "STATUS", "SUMMARY", "LOG_DIR", "OUTPUT",
        "candidate_name", "protocol", "validate_inputs", "summarize",
    )
    previous = {name: getattr(base, name) for name in names}
    try:
        configure_base()
        return base.training_command(cli, seed, resume)
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def main() -> int:
    configure_base()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
