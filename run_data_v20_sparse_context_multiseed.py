"""Run staged V20 sparse-context risk mitigation and validation gates."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from run_data_v13_yolo11_overlap_multiseed import BASE_MODEL, GROUP, GROUPS_CONFIG, PROJECT
from run_data_v17_seed2_threshold_calibration import fixed_row
from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256
from run_feature_ablations import read_progress, write_json_atomic


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "datasets" / "Data_v20_sparse_context_train"
DATASET_MANIFEST = DATASET_ROOT / "manifest.json"
DATA = ROOT / "configs" / "generated" / "data_v20_sparse_context_5ch.yaml"
V19_SUMMARY = ROOT / "logs" / "v19b_yolo11_boundary_nooverlap_multiseed.summary.json"
V19_REPORT = ROOT / "reports" / "Data_v19b_yolo11_boundary_nooverlap_multiseed"
STATUS = ROOT / "logs" / "v20_sparse_context_multiseed.status.json"
SUMMARY = ROOT / "logs" / "v20_sparse_context_multiseed.summary.json"
LOG_DIR = ROOT / "logs" / "v20_sparse_context_multiseed"
OUTPUT = ROOT / "reports" / "Data_v20_sparse_context_multiseed"
SEEDS = (0, 1, 2)
LOCKED_CONFIDENCE = 0.371


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=4300)
    parser.add_argument("--no-wait-for-gpu", dest="wait_for_gpu", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(wait_for_gpu=True)
    return parser.parse_args()


def candidate_name(seed: int) -> str:
    return f"v20_yolo11_5ch_sparsectx_b1_seed{seed}_v1"


def candidate_checkpoint(seed: int) -> Path:
    return PROJECT / candidate_name(seed) / "weights" / "best.pt"


def candidate_eval_dir(seed: int) -> Path:
    return OUTPUT / f"seed{seed}" / "standardized_validation"


def v19_eval_dir(seed: int) -> Path:
    return V19_REPORT / f"seed{seed}" / "candidate"


def fixed_dir(seed: int, variant: str) -> Path:
    return OUTPUT / f"seed{seed}" / f"{variant}_conf0371"


def audit_dir(seed: int) -> Path:
    return OUTPUT / f"seed{seed}" / "difficult_slice_audit"


def training_command(cli: argparse.Namespace, seed: int, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train_feature_ablation.py"),
        "--group", GROUP,
        "--groups-config", str(GROUPS_CONFIG),
        "--manifest", str(DATASET_MANIFEST),
        "--feature-root", str(DATASET_ROOT),
        "--data", str(DATA),
        "--name", candidate_name(seed),
        "--base-model", "yolo11n-seg.pt",
        "--attention", "ca_eca",
        "--extra-init", "mean",
        "--epochs", str(cli.epochs),
        "--patience", str(cli.patience),
        "--warmup-epochs", "1",
        "--imgsz", str(cli.imgsz),
        "--batch", str(cli.batch),
        "--workers", str(cli.workers),
        "--device", cli.device,
        "--seed", str(seed),
        "--mask-ratio", "4",
        "--no-overlap-mask",
        "--no-cache",
        "--deterministic",
        "--mosaic", "0.5",
        "--mixup", "0.0",
        "--close-mosaic", "15",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def sweep_command(seed: int, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "sweep_multichannel_confidence.py"),
        "--checkpoint", str(candidate_checkpoint(seed)),
        "--data", str(DATA),
        "--manifest", str(DATASET_MANIFEST),
        "--feature-root", str(DATASET_ROOT),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--output", str(candidate_eval_dir(seed)),
        "--imgsz", "1024",
        "--conf-min", "0.001",
        "--step", "0.005",
        "--device", device,
        "--no-overlap-mask",
    ]


def fixed_error_command(checkpoint: Path, output: Path, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "error_analysis_multichannel.py"),
        "--checkpoint", str(checkpoint),
        "--data", str(DATA),
        "--manifest", str(DATASET_MANIFEST),
        "--feature-root", str(DATASET_ROOT),
        "--channels", "R", "G", "B", "NDVI", "NDRE",
        "--output", str(output),
        "--imgsz", "1024",
        "--conf", str(LOCKED_CONFIDENCE),
        "--device", device,
        "--no-overlap-mask",
    ]


def paired_audit_command(seed: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "analyze_v13_precision_regression.py"),
        "--control", str(fixed_dir(seed, "v19_control")),
        "--candidate", str(fixed_dir(seed, "v20_candidate")),
        "--output", str(audit_dir(seed)),
    ]


def protocol() -> dict[str, object]:
    return {
        "version": "data_v20_sparse_context_multiseed_protocol_v1",
        "preregistered_at": now_iso(),
        "reference": "same-seed V19B checkpoint",
        "single_training_change": "add 14 train-only sparse-positive crops and 6 train-only empty-background crops",
        "architecture": "YOLO11n-seg + 3x CoordAtt + 3x ECA",
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "training_overlap_mask": False,
        "validation_overlap_mask": False,
        "augmentation": {"mosaic": 0.5, "mixup": 0.0, "close_mosaic": 15},
        "epochs": 60,
        "imgsz": 1024,
        "seeds": list(SEEDS),
        "locked_confidence": LOCKED_CONFIDENCE,
        "candidate_threshold_search_allowed": False,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "seed0_gate": {
            "map_delta_min": -0.002,
            "precision_delta_min": -0.01,
            "recall_delta_min": -0.01,
            "fixed_f1_delta_min": -0.002,
            "sparse_f1_delta_min": 0.02,
            "sparse_fp_delta_max": -5,
        },
        "three_seed_gate": {
            "mean_map_delta_min": -0.001,
            "worst_map_delta_min": -0.005,
            "mean_fixed_f1_delta_min": -0.002,
            "worst_fixed_f1_delta_min": -0.005,
            "mean_sparse_f1_delta_min": 0.015,
            "worst_sparse_f1_delta_min": -0.02,
            "positive_sparse_seeds_min": 2,
            "mean_sparse_fp_delta_max": -5,
        },
        "deployment_write_allowed": False,
    }


def validate_sweep(payload: dict, label: str) -> None:
    if payload.get("split") != "val" or payload.get("held_out_test_used") is not False:
        raise RuntimeError(f"{label} is not valid-only")
    if payload.get("validation_overlap_mask") is not False:
        raise RuntimeError(f"{label} overlap-mask contract drifted")
    if payload.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError(f"{label} channel contract drifted")


def validate_inputs() -> str:
    for path in (DATASET_ROOT, DATASET_MANIFEST, DATA, V19_SUMMARY, GROUPS_CONFIG, BASE_MODEL, DEPLOYMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    generated = dataset.get("generated", [])
    kinds = [row.get("kind") for row in generated]
    if (
        dataset.get("selection_basis_splits") != ["train"]
        or dataset.get("validation_labels_used_for_generation") is not False
        or dataset.get("held_out_split_used") is not False
        or dataset.get("total_train_images") != 230
        or kinds.count("sparse_positive") != 14
        or kinds.count("empty_background") != 6
    ):
        raise RuntimeError("V20 train-only dataset contract drifted")
    data_text = DATA.read_text(encoding="utf-8").lower()
    if "test:" in data_text or "channels: 5" not in data_text:
        raise RuntimeError("V20 training YAML exposed test or changed channel count")
    if len(list((DATASET_ROOT / "train" / "images").glob("*.JPG"))) != 230:
        raise RuntimeError("V20 image count drifted")
    if len(list((DATASET_ROOT / "train" / "features").glob("*.npy"))) != 230:
        raise RuntimeError("V20 train feature count drifted")
    if len(list((DATASET_ROOT / "valid" / "features").glob("*.npy"))) != 60:
        raise RuntimeError("V20 valid feature count drifted")
    v19 = json.loads(V19_SUMMARY.read_text(encoding="utf-8"))
    if v19.get("gate_passed") is not True or v19.get("held_out_test_used") is not False:
        raise RuntimeError("V19 reference is not eligible")
    for seed in SEEDS:
        payload = json.loads((v19_eval_dir(seed) / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
        validate_sweep(payload, f"V19 Seed-{seed}")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V20")
    return deployment_hash


def free_gpu_mib(device: str) -> int | None:
    if not str(device).isdigit():
        return None
    result = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        return int(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None
    except (ValueError, IndexError):
        return None


def run_logged(command: list[str], log_path: Path, state: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{now_iso()}] COMMAND: {' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        state["current_process_pid"] = process.pid
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS, state)
        return_code = process.wait()
    state["current_process_pid"] = None
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}); see {log_path}")


def wait_for_gpu(cli: argparse.Namespace, state: dict[str, object]) -> None:
    if not cli.wait_for_gpu:
        return
    while True:
        free = free_gpu_mib(cli.device)
        state["gpu_free_mib"] = free
        state["updated_at"] = now_iso()
        write_json_atomic(STATUS, state)
        if free is None or free >= cli.min_free_gpu_mib:
            return
        time.sleep(max(2.0, cli.gpu_poll_seconds))


def run_training(cli: argparse.Namespace, seed: int, state: dict[str, object], deployment_hash: str) -> None:
    name = candidate_name(seed)
    run_dir = PROJECT / name
    results = run_dir / "results.csv"
    best = candidate_checkpoint(seed)
    progress = read_progress(results)
    if progress and int(progress["epochs_recorded"]) >= cli.epochs and best.is_file():
        state["progress"][name] = progress
        return
    last = run_dir / "weights" / "last.pt"
    resume = last if progress and last.is_file() else None
    if run_dir.exists() and resume is None:
        raise RuntimeError(f"Incomplete non-resumable run exists: {run_dir}")
    wait_for_gpu(cli, state)
    state["stage"] = "training"
    state["current_seed"] = seed
    state["current_run"] = name
    write_json_atomic(STATUS, state)
    run_logged(training_command(cli, seed, resume), LOG_DIR / f"{name}.log", state)
    progress = read_progress(results)
    if not progress or int(progress["epochs_recorded"]) < cli.epochs or not best.is_file():
        raise RuntimeError(f"Training did not produce a complete checkpoint: {name}")
    if sha256(DEPLOYMENT) != deployment_hash:
        raise RuntimeError("Deployment checkpoint changed during V20 training")
    state["progress"][name] = progress
    state["completed_runs"].append(name)


def ensure_sweep(cli: argparse.Namespace, seed: int, state: dict[str, object]) -> dict:
    path = candidate_eval_dir(seed) / "confidence_sweep.summary.json"
    if not path.is_file():
        state["stage"] = "standardized_validation"
        state["current_seed"] = seed
        write_json_atomic(STATUS, state)
        run_logged(sweep_command(seed, cli.device), LOG_DIR / f"{candidate_name(seed)}.sweep.log", state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_sweep(payload, f"V20 Seed-{seed}")
    return payload


def ensure_audit(cli: argparse.Namespace, seed: int, state: dict[str, object]) -> dict:
    paired = audit_dir(seed) / "paired_summary.json"
    if paired.is_file():
        return json.loads(paired.read_text(encoding="utf-8"))
    state["stage"] = "fixed_threshold_difficult_slice_audit"
    state["current_seed"] = seed
    write_json_atomic(STATUS, state)
    control_output = fixed_dir(seed, "v19_control")
    candidate_output = fixed_dir(seed, "v20_candidate")
    if not (control_output / "per_image_metrics.csv").is_file():
        run_logged(
            fixed_error_command(Path(json.loads(V19_SUMMARY.read_text(encoding="utf-8"))["rows"][seed]["candidate_checkpoint"]), control_output, cli.device),
            LOG_DIR / f"seed{seed}.v19_fixed.log",
            state,
        )
    if not (candidate_output / "per_image_metrics.csv").is_file():
        run_logged(
            fixed_error_command(candidate_checkpoint(seed), candidate_output, cli.device),
            LOG_DIR / f"seed{seed}.v20_fixed.log",
            state,
        )
    run_logged(paired_audit_command(seed), LOG_DIR / f"seed{seed}.audit.log", state)
    return json.loads(paired.read_text(encoding="utf-8"))


def pair_row(seed: int) -> dict[str, object]:
    v19 = json.loads((v19_eval_dir(seed) / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
    v20 = json.loads((candidate_eval_dir(seed) / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
    validate_sweep(v19, f"V19 Seed-{seed}")
    validate_sweep(v20, f"V20 Seed-{seed}")
    base = v19["aggregate_metrics"]
    trial = v20["aggregate_metrics"]
    base_fixed = fixed_row(v19_eval_dir(seed) / "confidence_sweep.csv", LOCKED_CONFIDENCE)
    trial_fixed = fixed_row(candidate_eval_dir(seed) / "confidence_sweep.csv", LOCKED_CONFIDENCE)
    audit = json.loads((audit_dir(seed) / "paired_summary.json").read_text(encoding="utf-8"))
    sparse = audit["slice_summary"]["sparse_low_coverage"]
    return {
        "seed": seed,
        "reference_checkpoint": str(Path(v19["checkpoint"]).resolve()),
        "candidate_checkpoint": str(candidate_checkpoint(seed).resolve()),
        "candidate_checkpoint_sha256": sha256(candidate_checkpoint(seed)),
        "reference_map": float(base["metrics/mAP50-95(M)"]),
        "candidate_map": float(trial["metrics/mAP50-95(M)"]),
        "map_delta": float(trial["metrics/mAP50-95(M)"]) - float(base["metrics/mAP50-95(M)"]),
        "precision_delta": float(trial["metrics/precision(M)"]) - float(base["metrics/precision(M)"]),
        "recall_delta": float(trial["metrics/recall(M)"]) - float(base["metrics/recall(M)"]),
        "reference_fixed": base_fixed,
        "candidate_fixed": trial_fixed,
        "fixed_f1_delta": float(trial_fixed["f1"]) - float(base_fixed["f1"]),
        "sparse_images": int(sparse["images"]),
        "sparse_f1_delta": float(sparse["delta"]["f1"]),
        "sparse_precision_delta": float(sparse["delta"]["precision"]),
        "sparse_recall_delta": float(sparse["delta"]["recall"]),
        "sparse_tp_delta": int(sparse["delta"]["tp"]),
        "sparse_fp_delta": int(sparse["delta"]["fp"]),
        "sparse_fn_delta": int(sparse["delta"]["fn"]),
        "slice_summary": audit["slice_summary"],
    }


def summarize(rows: list[dict[str, object]], deployment_hash: str) -> dict[str, object]:
    seed0 = next((row for row in rows if row["seed"] == 0), None)
    seed0_passed = bool(
        seed0
        and float(seed0["map_delta"]) >= -0.002
        and float(seed0["precision_delta"]) >= -0.01
        and float(seed0["recall_delta"]) >= -0.01
        and float(seed0["fixed_f1_delta"]) >= -0.002
        and float(seed0["sparse_f1_delta"]) >= 0.02
        and int(seed0["sparse_fp_delta"]) <= -5
    )
    metrics: dict[str, object] = {}
    gate_passed = False
    if len(rows) == len(SEEDS):
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
            and metrics["mean_sparse_f1_delta"] >= 0.015
            and metrics["worst_sparse_f1_delta"] >= -0.02
            and metrics["positive_sparse_seeds"] >= 2
            and metrics["mean_sparse_fp_delta"] <= -5
        )
    frozen = max(rows, key=lambda row: float(row["candidate_map"])) if gate_passed else None
    return {
        "version": "data_v20_sparse_context_multiseed_summary_v1",
        "created_at": now_iso(),
        "protocol": protocol(),
        "rows": rows,
        "seed0_passed": seed0_passed,
        "gate_metrics": metrics,
        "gate_passed": gate_passed,
        "decision": (
            "freeze_v20_candidate_for_one_time_test" if gate_passed else
            "continue_v20_multiseed" if seed0_passed and len(rows) < len(SEEDS) else
            "reject_v20_retain_v19_frozen_candidate"
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


def main() -> int:
    cli = parse_args()
    if cli.epochs != 60 or cli.imgsz != 1024 or cli.batch != 2:
        raise ValueError("V20 formal protocol is locked to 60 epochs, imgsz=1024, batch=2")
    deployment_hash = validate_inputs()
    if cli.dry_run:
        print(json.dumps({"protocol": protocol(), "commands": [training_command(cli, seed) for seed in SEEDS]}, ensure_ascii=False, indent=2))
        return 0
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "version": "data_v20_sparse_context_multiseed_status_v1",
        "state": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol(),
        "completed_runs": [],
        "progress": {},
        "rows": [],
        "current_seed": None,
        "current_run": None,
        "current_process_pid": None,
        "deployment_sha256_before": deployment_hash,
        "deployment_write_allowed": False,
    }
    write_json_atomic(STATUS, state)
    rows: list[dict[str, object]] = []
    try:
        for seed in SEEDS:
            run_training(cli, seed, state, deployment_hash)
            ensure_sweep(cli, seed, state)
            ensure_audit(cli, seed, state)
            row = pair_row(seed)
            rows.append(row)
            state["rows"] = rows
            current = summarize(rows, deployment_hash)
            write_json_atomic(SUMMARY, current)
            write_json_atomic(STATUS, state)
            if seed == 0 and not current["seed0_passed"]:
                break
        final = summarize(rows, deployment_hash)
        write_json_atomic(SUMMARY, final)
        current_deployment = sha256(DEPLOYMENT)
        if current_deployment != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V20")
        state.update(
            state="completed",
            stage="completed",
            completed_at=now_iso(),
            updated_at=now_iso(),
            rows=rows,
            decision=final["decision"],
            gate_passed=final["gate_passed"],
            deployment_sha256_after=current_deployment,
            deployment_unchanged=True,
            current_seed=None,
            current_run=None,
            current_process_pid=None,
        )
        write_json_atomic(STATUS, state)
        print(json.dumps({"decision": final["decision"], "gate_passed": final["gate_passed"]}, ensure_ascii=False))
        return 0
    except BaseException as exc:
        state.update(
            state="failed",
            failed_at=now_iso(),
            updated_at=now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            deployment_sha256_after=sha256(DEPLOYMENT) if DEPLOYMENT.is_file() else None,
        )
        state["deployment_unchanged"] = state["deployment_sha256_after"] == deployment_hash
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
