"""Summarize the locked Data_v8 Seed-0 fusion screen and apply numeric gates."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
OUTPUT_DIR = ROOT / "reports" / "Data_v8_multispectral_fusion_screen"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
GROUPS = (
    "c0_current_indices_early",
    "c1_raw_bands_early",
    "f1_p4_concat",
    "f2_multiscale_gated",
)
RUNS = {group: f"v8_{group}_yolov8n_none_1024_seed0_screen_v1" for group in GROUPS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_best(group: str) -> dict:
    run = RUNS[group]
    run_dir = PROJECT / run
    results = run_dir / "results.csv"
    contract_path = run_dir / "fusion_contract.json"
    best_path = run_dir / "weights" / "best.pt"
    for required in (results, contract_path, best_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    with results.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No result rows: {results}")
    best = max(rows, key=lambda row: as_float(row, "metrics/mAP50-95(M)"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return {
        "group": group,
        "run": run,
        "epochs_recorded": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "mask_map50_95": as_float(best, "metrics/mAP50-95(M)"),
        "mask_map50": as_float(best, "metrics/mAP50(M)"),
        "mask_precision": as_float(best, "metrics/precision(M)"),
        "mask_recall": as_float(best, "metrics/recall(M)"),
        "box_map50_95": as_float(best, "metrics/mAP50-95(B)"),
        "total_parameters": int(contract["total_parameters"]),
        "fusion_parameters": int(contract["fusion_parameters"]),
        "checkpoint": str(best_path.resolve()),
    }


def main() -> None:
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError(f"Deployment SHA256 mismatch: {deployment_hash}")
    rows = [load_best(group) for group in GROUPS]
    by_group = {row["group"]: row for row in rows}
    c0 = by_group["c0_current_indices_early"]
    c1 = by_group["c1_raw_bands_early"]
    candidates = []
    for group in ("f1_p4_concat", "f2_multiscale_gated"):
        row = dict(by_group[group])
        row["delta_vs_c0"] = row["mask_map50_95"] - c0["mask_map50_95"]
        row["delta_vs_c1"] = row["mask_map50_95"] - c1["mask_map50_95"]
        row["precision_delta_vs_c0"] = row["mask_precision"] - c0["mask_precision"]
        row["recall_delta_vs_c0"] = row["mask_recall"] - c0["mask_recall"]
        row["parameter_growth_vs_c0"] = row["total_parameters"] / c0["total_parameters"] - 1.0
        row["numeric_gate"] = bool(
            row["delta_vs_c0"] >= 0.003
            and row["delta_vs_c1"] >= 0.001
            and row["precision_delta_vs_c0"] >= -0.01
            and row["recall_delta_vs_c0"] >= -0.01
            and row["parameter_growth_vs_c0"] <= 0.25
        )
        candidates.append(row)
    best_candidate = max(candidates, key=lambda row: row["mask_map50_95"])
    payload = {
        "version": "data_v8_multispectral_fusion_seed0_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "screen_contract": {
            "base_model": "yolov8n-seg.pt",
            "attention": "none",
            "seed": 0,
            "epochs": 30,
            "imgsz": 1024,
            "batch": 2,
        },
        "controls": [c0, c1],
        "candidates": candidates,
        "best_candidate": best_candidate["group"],
        "numeric_gate_passed": [row["group"] for row in candidates if row["numeric_gate"]],
        "difficult_slice_review_required": [row["group"] for row in candidates if row["numeric_gate"]],
        "f3_allowed": False,
        "f3_note": "F3 remains blocked until F2 passes the numeric gate and difficult-slice review.",
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
