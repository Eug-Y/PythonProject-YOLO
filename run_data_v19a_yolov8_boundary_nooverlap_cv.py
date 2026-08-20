"""Paired YOLOv8n grouped-CV screen for boundary augmentation with separate masks."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_data_v9_grouped_cv import (
    CONTRACT_PATH,
    CV_ROOT,
    DEPLOYMENT,
    DEPLOYMENT_SHA256,
    FEATURE_ROOT,
    GROUP,
    GROUPS_CONFIG,
    MANIFEST,
    PROJECT,
    now_iso,
    sha256,
)
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "logs" / "v19a_yolov8_boundary_nooverlap_cv.status.json"
SUMMARY = ROOT / "logs" / "v19a_yolov8_boundary_nooverlap_cv.summary.json"
LOG_DIR = ROOT / "logs" / "v19a_yolov8_boundary_nooverlap_cv"
OUTPUT = ROOT / "reports" / "Data_v19a_yolov8_boundary_nooverlap_cv"
FOLDS = (0, 1, 2)
VARIANTS = ("control", "candidate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def data_yaml(fold: int) -> Path:
    return CV_ROOT / f"fold{fold}" / "data_5ch.yaml"


def experiment_name(fold: int, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown V19A variant: {variant}")
    suffix = "baselineaug" if variant == "control" else "b1aug"
    return f"v19a_groupcv_fold{fold}_nooverlap_{suffix}_yolov8n_none_640_seed0_30ep_v1"


def checkpoint(fold: int, variant: str) -> Path:
    return PROJECT / experiment_name(fold, variant) / "weights" / "best.pt"


def output_dir(fold: int, variant: str) -> Path:
    return OUTPUT / f"fold{fold}" / variant


def training_command(
    cli: argparse.Namespace,
    fold: int,
    variant: str,
    *,
    resume: Path | None = None,
    smoke_test: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", GROUP,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--data", str(data_yaml(fold)),
        "--name", (
            f"smoke_{experiment_name(fold, variant)}"
            if smoke_test else experiment_name(fold, variant)
        ),
        "--base-model", "yolov8n-seg.pt",
        "--attention", "none",
        "--extra-init", "mean",
        "--epochs", str(cli.epochs),
        "--patience", str(cli.patience),
        "--warmup-epochs", "1",
        "--imgsz", str(cli.imgsz),
        "--batch", str(cli.batch),
        "--workers", str(cli.workers),
        "--device", cli.device,
        "--seed", str(cli.seed),
        "--mask-ratio", "4",
        "--no-overlap-mask",
        "--no-cache",
        "--deterministic",
    ]
    if variant == "control":
        command.extend(("--mosaic", "1.0", "--mixup", "0.1", "--close-mosaic", "10"))
    else:
        command.extend(("--mosaic", "0.5", "--mixup", "0.0", "--close-mosaic", "15"))
    if smoke_test:
        command.append("--smoke-test")
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def sweep_command(fold: int, variant: str, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "sweep_multichannel_confidence.py"),
        "--checkpoint", str(checkpoint(fold, variant)),
        "--data", str(data_yaml(fold)),
        "--manifest", str(MANIFEST),
        "--feature-root", str(FEATURE_ROOT),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--output", str(output_dir(fold, variant)),
        "--imgsz", "640",
        "--conf-min", "0.001",
        "--step", "0.005",
        "--device", device,
        "--no-overlap-mask",
    ]


def protocol() -> dict[str, object]:
    return {
        "version": "v19a_yolov8_boundary_nooverlap_cv_protocol_v1",
        "purpose": "isolate B1 boundary augmentation on top of overlap_mask=False",
        "model": "YOLOv8n-seg",
        "folds": 3,
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "attention": "none",
        "epochs": 30,
        "patience": 12,
        "imgsz": 640,
        "batch": 4,
        "seed": 0,
        "mask_ratio": 4,
        "training_overlap_mask": False,
        "validation_overlap_mask": False,
        "control_augmentation": {"mosaic": 1.0, "mixup": 0.1, "close_mosaic": 10},
        "candidate_augmentation": {"mosaic": 0.5, "mixup": 0.0, "close_mosaic": 15},
        "gate_thresholds": {
            "mean_map_delta_min": 0.002,
            "worst_map_delta_min": -0.005,
            "positive_folds_min": 2,
            "mean_precision_delta_min": -0.01,
            "mean_recall_delta_min": -0.01,
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_write_allowed": False,
    }


def validate_inputs() -> tuple[dict, str]:
    for path in (
        CONTRACT_PATH,
        FEATURE_ROOT,
        MANIFEST,
        GROUPS_CONFIG,
        ROOT / "yolov8n-seg.pt",
        DEPLOYMENT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if contract.get("source_splits") != ["train", "valid"]:
        raise RuntimeError("V19A source split contract drifted")
    if contract.get("held_out_test_used") is not False or contract.get("test_paths_enumerated") is not False:
        raise RuntimeError("V19A grouped-CV contract does not preserve the sealed test")
    if manifest.get("selection_splits") != ["train", "valid"] or manifest.get("held_out_split_used") is not False:
        raise RuntimeError("V19A feature manifest does not preserve the sealed test")
    for fold in FOLDS:
        text = data_yaml(fold).read_text(encoding="utf-8").lower()
        if "test:" in text or "channels: 5" not in text:
            raise RuntimeError(f"Invalid V19A grouped-CV YAML: {data_yaml(fold)}")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V19A")
    return contract, deployment_hash


def read_result(fold: int, variant: str) -> dict:
    path = output_dir(fold, variant) / "confidence_sweep.summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("held_out_test_used") is not False:
        raise RuntimeError(f"Non-validation result in V19A: {path}")
    if payload.get("validation_overlap_mask") is not False:
        raise RuntimeError(f"V19A validation overlap contract drifted: {path}")
    if payload.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError(f"V19A channel contract drifted: {path}")
    return payload


def paired_row(fold: int, contract: dict) -> dict[str, object]:
    control = read_result(fold, "control")
    candidate = read_result(fold, "candidate")
    control_metrics = control["aggregate_metrics"]
    candidate_metrics = candidate["aggregate_metrics"]
    keys = {
        "map": "metrics/mAP50-95(M)",
        "map50": "metrics/mAP50(M)",
        "precision": "metrics/precision(M)",
        "recall": "metrics/recall(M)",
    }
    row: dict[str, object] = {
        "fold": fold,
        "validation_images": int(contract["fold_contracts"][fold]["validation_images"]),
        "control_checkpoint": str(checkpoint(fold, "control").resolve()),
        "candidate_checkpoint": str(checkpoint(fold, "candidate").resolve()),
        "control_checkpoint_sha256": sha256(checkpoint(fold, "control")),
        "candidate_checkpoint_sha256": sha256(checkpoint(fold, "candidate")),
    }
    for short, key in keys.items():
        baseline = float(control_metrics[key])
        trial = float(candidate_metrics[key])
        row[f"control_{short}"] = baseline
        row[f"candidate_{short}"] = trial
        row[f"{short}_delta"] = trial - baseline
        row[f"{short}_delta_pp"] = (trial - baseline) * 100.0
    return row


def gate_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 3:
        return {"passed": False, "checks": {"three_folds_complete": False}}
    map_deltas = [float(row["map_delta"]) for row in rows]
    precision_deltas = [float(row["precision_delta"]) for row in rows]
    recall_deltas = [float(row["recall_delta"]) for row in rows]
    checks = {
        "three_folds_complete": True,
        "mean_map_delta_at_least_0_002": statistics.fmean(map_deltas) >= 0.002,
        "worst_map_delta_not_below_minus_0_005": min(map_deltas) >= -0.005,
        "at_least_two_positive_map_folds": sum(delta > 0 for delta in map_deltas) >= 2,
        "mean_precision_delta_not_below_minus_0_01": statistics.fmean(precision_deltas) >= -0.01,
        "mean_recall_delta_not_below_minus_0_01": statistics.fmean(recall_deltas) >= -0.01,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "mean_map_delta": statistics.fmean(map_deltas),
        "mean_map_delta_pp": statistics.fmean(map_deltas) * 100.0,
        "worst_map_delta": min(map_deltas),
        "worst_map_delta_pp": min(map_deltas) * 100.0,
        "positive_map_folds": sum(delta > 0 for delta in map_deltas),
        "mean_precision_delta": statistics.fmean(precision_deltas),
        "mean_recall_delta": statistics.fmean(recall_deltas),
    }


def progress_snapshot() -> dict[str, object]:
    items: dict[str, object] = {}
    recorded = 0
    for fold in FOLDS:
        for variant in VARIANTS:
            name = experiment_name(fold, variant)
            item = read_progress(PROJECT / name / "results.csv")
            if item:
                items[f"fold{fold}_{variant}"] = item
                recorded += min(int(item["epochs_recorded"]), 30)
    return {
        "runs": items,
        "epochs_recorded": recorded,
        "target_epochs": len(FOLDS) * len(VARIANTS) * 30,
        "percent": round(recorded / (len(FOLDS) * len(VARIANTS) * 30) * 100.0, 2),
    }


def run_logged(command: list[str], log_path: Path, state: dict, cli: argparse.Namespace) -> None:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        state["current_process_pid"] = process.pid
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS, state)
        while process.poll() is None:
            time.sleep(max(cli.poll_seconds, 2.0))
            state["progress"] = progress_snapshot()
            state["updated_at"] = now_iso()
            write_json_atomic(STATUS, state)
    state["current_process_pid"] = None
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}; see {log_path}")


def run_training(cli: argparse.Namespace, fold: int, variant: str, state: dict, deployment_hash: str) -> None:
    name = experiment_name(fold, variant)
    run_dir = PROJECT / name
    result_csv = run_dir / "results.csv"
    best = checkpoint(fold, variant)
    progress = read_progress(result_csv)
    if progress and int(progress["epochs_recorded"]) >= cli.epochs and best.is_file():
        return
    resume = None
    last = run_dir / "weights" / "last.pt"
    if progress and last.is_file():
        resume = last
    elif run_dir.exists():
        raise RuntimeError(f"Incomplete V19A run has no resumable checkpoint: {run_dir}")
    state.update(stage="training", current={"fold": fold, "variant": variant, "name": name})
    run_logged(training_command(cli, fold, variant, resume=resume), LOG_DIR / f"{name}.log", state, cli)
    progress = read_progress(result_csv)
    if not progress or not best.is_file():
        raise RuntimeError(f"V19A training completed without metrics/checkpoint: {name}")
    if sha256(DEPLOYMENT) != deployment_hash:
        raise RuntimeError("Deployment checkpoint changed during V19A training")


def run_evaluation(cli: argparse.Namespace, fold: int, variant: str, state: dict) -> None:
    result = output_dir(fold, variant) / "confidence_sweep.summary.json"
    state.update(stage="standardized_validation", current={"fold": fold, "variant": variant})
    state["updated_at"] = now_iso()
    write_json_atomic(STATUS, state)
    if not result.is_file():
        run_logged(
            sweep_command(fold, variant, cli.device),
            LOG_DIR / f"fold{fold}_{variant}_evaluation.log",
            state,
            cli,
        )
    read_result(fold, variant)


def main() -> int:
    cli = parse_args()
    if (cli.epochs, cli.patience, cli.imgsz, cli.batch, cli.seed) != (30, 12, 640, 4, 0):
        raise ValueError("V19A is locked to epochs=30, patience=12, imgsz=640, batch=4, seed=0")
    contract, deployment_hash = validate_inputs()
    locked_protocol = protocol()
    if cli.dry_run:
        print(json.dumps(locked_protocol, ensure_ascii=False, indent=2))
        for variant in VARIANTS:
            print(subprocess.list2cmdline(training_command(cli, 0, variant)))
            print(subprocess.list2cmdline(sweep_command(0, variant, cli.device)))
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    if previous.get("protocol") not in (None, locked_protocol):
        raise RuntimeError("Existing V19A status uses a different protocol")
    state = {
        "version": "v19a_yolov8_boundary_nooverlap_cv_status_v1",
        "state": "running",
        "runner_pid": os.getpid(),
        "started_at": previous.get("started_at", now_iso()),
        "updated_at": now_iso(),
        "protocol": locked_protocol,
        "stage": "paired_training",
        "current": None,
        "current_process_pid": None,
        "completed": previous.get("completed", []),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
        "progress": progress_snapshot(),
    }
    write_json_atomic(STATUS, state)
    try:
        for fold in FOLDS:
            for variant in VARIANTS:
                run_training(cli, fold, variant, state, deployment_hash)
                run_evaluation(cli, fold, variant, state)
                completed = {f"{item['fold']}:{item['variant']}" for item in state["completed"]}
                if f"{fold}:{variant}" not in completed:
                    state["completed"].append({"fold": fold, "variant": variant})
                state["progress"] = progress_snapshot()
                state["updated_at"] = now_iso()
                write_json_atomic(STATUS, state)

        rows = [paired_row(fold, contract) for fold in FOLDS]
        gate = gate_decision(rows)
        current_hash = sha256(DEPLOYMENT)
        if current_hash != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed before V19A summary")
        summary = {
            "version": "v19a_yolov8_boundary_nooverlap_cv_summary_v1",
            "created_at": now_iso(),
            "protocol": locked_protocol,
            "rows": rows,
            "gate": gate,
            "decision": "promote_combination_to_yolo11_v19b" if gate["passed"] else "retain_v15_nooverlap_candidate",
            "held_out_test_used": False,
            "deployment_write_allowed": False,
            "deployment_sha256_expected": DEPLOYMENT_SHA256,
            "deployment_sha256_current": current_hash,
            "deployment_unchanged": current_hash == DEPLOYMENT_SHA256,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(
            state="complete",
            stage="complete",
            current=None,
            updated_at=now_iso(),
            completed_at=now_iso(),
            gate=gate,
            decision=summary["decision"],
            summary=str(SUMMARY.resolve()),
            deployment_sha256_after=current_hash,
            deployment_unchanged=True,
            progress=progress_snapshot(),
        )
        write_json_atomic(STATUS, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        current_hash = sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None
        state.update(
            state="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {exc}",
            failed_at=now_iso(),
            updated_at=now_iso(),
            deployment_sha256_after=current_hash,
            deployment_unchanged=current_hash == deployment_hash,
            progress=progress_snapshot(),
        )
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
