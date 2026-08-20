"""Run the one-step V21 half-exposure sparse-context confirmation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import run_data_v20_sparse_context_multiseed as base
from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "datasets" / "Data_v21_sparse_context_half_train"
DATASET_MANIFEST = DATASET_ROOT / "manifest.json"
DATA = ROOT / "configs" / "generated" / "data_v21_sparse_context_half_5ch.yaml"
STATUS = ROOT / "logs" / "v21_sparse_context_half_multiseed.status.json"
SUMMARY = ROOT / "logs" / "v21_sparse_context_half_multiseed.summary.json"
LOG_DIR = ROOT / "logs" / "v21_sparse_context_half_multiseed"
OUTPUT = ROOT / "reports" / "Data_v21_sparse_context_half_multiseed"


def candidate_name(seed: int) -> str:
    return f"v21_yolo11_5ch_sparsectx_half_b1_seed{seed}_v1"


def protocol() -> dict[str, object]:
    return {
        "version": "data_v21_sparse_context_half_multiseed_protocol_v1",
        "preregistered_at": now_iso(),
        "reference": "same-seed V19B checkpoint",
        "prerequisite": "V20 full sparse-context exposure reduced FP but failed aggregate gates",
        "single_training_change": "reduce train-only synthetic exposure from 14+6 crops to 7+3 crops",
        "architecture": "YOLO11n-seg + 3x CoordAtt + 3x ECA",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "training_overlap_mask": False,
        "validation_overlap_mask": False,
        "augmentation": {"mosaic": 0.5, "mixup": 0.0, "close_mosaic": 15},
        "epochs": 60,
        "imgsz": 1024,
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
        },
        "deployment_write_allowed": False,
    }


def validate_inputs() -> str:
    for path in (
        DATASET_ROOT,
        DATASET_MANIFEST,
        DATA,
        base.V19_SUMMARY,
        base.GROUPS_CONFIG,
        base.BASE_MODEL,
        DEPLOYMENT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    generated = dataset.get("generated", [])
    kinds = [row.get("kind") for row in generated]
    groups = {row.get("source_group") for row in generated if row.get("source_group")}
    if (
        dataset.get("version") != "data_v21_sparse_context_half_train_v1"
        or dataset.get("selection_basis_splits") != ["train"]
        or dataset.get("selection_splits") != ["train", "valid"]
        or dataset.get("validation_labels_used_for_generation") is not False
        or dataset.get("held_out_split_used") is not False
        or dataset.get("total_train_images") != 220
        or kinds.count("sparse_positive") != 7
        or kinds.count("empty_background") != 3
        or len(groups) < 5
    ):
        raise RuntimeError("V21 half-exposure train-only dataset contract drifted")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text or "channels: 5" not in data_text:
        raise RuntimeError("V21 YAML exposed test or changed channels")
    if len(list((DATASET_ROOT / "train" / "images").glob("*.JPG"))) != 220:
        raise RuntimeError("V21 image count drifted")
    if len(list((DATASET_ROOT / "train" / "features").glob("*.npy"))) != 220:
        raise RuntimeError("V21 feature count drifted")
    v19 = json.loads(base.V19_SUMMARY.read_text(encoding="utf-8"))
    if v19.get("gate_passed") is not True or v19.get("held_out_test_used") is not False:
        raise RuntimeError("V19 reference is not eligible")
    for seed in base.SEEDS:
        payload = json.loads((base.v19_eval_dir(seed) / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
        base.validate_sweep(payload, f"V19 Seed-{seed}")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V21")
    return deployment_hash


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
    )
    metrics: dict[str, object] = {}
    gate_passed = False
    if len(rows) == len(base.SEEDS):
        map_deltas = [float(row["map_delta"]) for row in rows]
        fixed_deltas = [float(row["fixed_f1_delta"]) for row in rows]
        sparse_f1 = [float(row["sparse_f1_delta"]) for row in rows]
        sparse_fp = [int(row["sparse_fp_delta"]) for row in rows]
        metrics = {
            "mean_map_delta": statistics.fmean(map_deltas),
            "worst_map_delta": min(map_deltas),
            "mean_fixed_f1_delta": statistics.fmean(fixed_deltas),
            "worst_fixed_f1_delta": min(fixed_deltas),
            "mean_sparse_f1_delta": statistics.fmean(sparse_f1),
            "worst_sparse_f1_delta": min(sparse_f1),
            "positive_sparse_seeds": sum(value > 0.0 for value in sparse_f1),
            "mean_sparse_fp_delta": statistics.fmean(sparse_fp),
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
        )
    frozen = max(rows, key=lambda row: float(row["candidate_map"])) if gate_passed else None
    return {
        "version": "data_v21_sparse_context_half_multiseed_summary_v1",
        "created_at": now_iso(),
        "protocol": protocol(),
        "rows": rows,
        "seed0_passed": seed0_passed,
        "gate_metrics": metrics,
        "gate_passed": gate_passed,
        "decision": (
            "freeze_v21_candidate_for_one_time_test" if gate_passed else
            "continue_v21_multiseed" if seed0_passed and len(rows) < len(base.SEEDS) else
            "reject_v21_retain_v19_frozen_candidate"
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
    """Point the audited V20 orchestration at V21 inside the runner process."""
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
    names = ("DATASET_ROOT", "DATASET_MANIFEST", "DATA", "STATUS", "SUMMARY", "LOG_DIR", "OUTPUT", "candidate_name", "protocol", "validate_inputs", "summarize")
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
