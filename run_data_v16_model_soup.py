"""Preregister, build, and evaluate the single V15 uniform model-soup candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "logs" / "v16_model_soup.status.json"
SUMMARY = ROOT / "logs" / "v16_model_soup.summary.json"
V15_SUMMARY = ROOT / "logs" / "v15_yolo11_standardized_multiseed.summary.json"
SOUP = ROOT / "runs" / "segment" / "land_seg" / "v16_yolo11_v15_seed012_uniform_soup_v1" / "weights" / "best.pt"
SOUP_MANIFEST = SOUP.parent.parent / "soup_manifest.json"
EVALUATION = ROOT / "reports" / "Data_v16_yolo11_uniform_soup"
SEED2_EVALUATION = ROOT / "reports" / "Data_v15_yolo11_standardized_multiseed" / "seed2" / "candidate" / "confidence_sweep.summary.json"
LOG = ROOT / "logs" / "v16_model_soup.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def gate(control: dict, candidate: dict) -> dict[str, object]:
    keys = {
        "map": "metrics/mAP50-95(M)",
        "map50": "metrics/mAP50(M)",
        "precision": "metrics/precision(M)",
        "recall": "metrics/recall(M)",
    }
    values = {}
    for short, key in keys.items():
        baseline = float(control["aggregate_metrics"][key])
        trial = float(candidate["aggregate_metrics"][key])
        values[f"seed2_{short}"] = baseline
        values[f"soup_{short}"] = trial
        values[f"{short}_delta"] = trial - baseline
        values[f"{short}_delta_pp"] = (trial - baseline) * 100.0
    checks = {
        "map_delta_at_least_0_002": values["map_delta"] >= 0.002,
        "precision_delta_not_below_minus_0_01": values["precision_delta"] >= -0.01,
        "recall_delta_not_below_minus_0_01": values["recall_delta"] >= -0.01,
    }
    return {"metrics": values, "checks": checks, "passed": all(checks.values())}


def main() -> int:
    args = parse_args()
    v15 = json.loads(V15_SUMMARY.read_text(encoding="utf-8"))
    if v15.get("gate_passed") is not True or v15.get("decision") != "freeze_candidate_for_review_no_auto_deployment":
        raise RuntimeError("V15 did not authorize V16 model-soup review")
    if v15.get("held_out_test_used") is not False or v15.get("validation_overlap_mask") is not False:
        raise RuntimeError("V15 validation contract drifted")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V16")
    protocol = {
        "version": "v16_uniform_model_soup_protocol_v1",
        "preregistered_at": now_iso(),
        "candidate": "uniform parameter average of V15 Seed 0/1/2 best checkpoints",
        "reference": "V15 strongest individual candidate Seed-2",
        "single_candidate_only": True,
        "validation_overlap_mask": False,
        "imgsz": 1024,
        "inference_confidence_min": 0.001,
        "gate_thresholds": {
            "map_delta_min": 0.002,
            "precision_delta_min": -0.01,
            "recall_delta_min": -0.01,
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_write_allowed": False,
    }
    if STATUS.is_file():
        previous = json.loads(STATUS.read_text(encoding="utf-8"))
        if previous.get("protocol", {}).get("version") == protocol["version"]:
            protocol = previous["protocol"]
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0
    state = {
        "version": "v16_model_soup_status_v1",
        "state": "running",
        "stage": "build_soup",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "protocol": protocol,
        "deployment_sha256_before": deployment_hash,
    }
    write_json_atomic(STATUS, state)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    try:
        commands = [
            [sys.executable, str(ROOT / "build_data_v16_model_soup.py")],
            [
                sys.executable,
                str(ROOT / "sweep_multichannel_confidence.py"),
                "--checkpoint", str(SOUP),
                "--data", str(ROOT / "configs" / "data_v8_fusion_selection_5ch.yaml"),
                "--manifest", str(ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json"),
                "--feature-root", str(ROOT / "datasets" / "Data_v4_candidates"),
                "--channels", "R", "G", "B", "NDVI", "NDRE",
                "--output", str(EVALUATION),
                "--imgsz", "1024",
                "--conf-min", "0.001",
                "--step", "0.005",
                "--device", args.device,
                "--no-overlap-mask",
            ],
        ]
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            for index, command in enumerate(commands):
                state["stage"] = "build_soup" if index == 0 else "standardized_validation"
                state["updated_at"] = now_iso()
                write_json_atomic(STATUS, state)
                log.write(f"\n[{now_iso()}] COMMAND: {subprocess.list2cmdline(command)}\n")
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(f"V16 command failed with exit code {completed.returncode}; see {LOG}")
        soup_manifest = json.loads(SOUP_MANIFEST.read_text(encoding="utf-8"))
        reference = json.loads(SEED2_EVALUATION.read_text(encoding="utf-8"))
        candidate = json.loads((EVALUATION / "confidence_sweep.summary.json").read_text(encoding="utf-8"))
        for payload in (reference, candidate):
            if payload.get("held_out_test_used") is not False or payload.get("validation_overlap_mask") is not False:
                raise RuntimeError("V16 comparison left the standardized valid-only contract")
        result = gate(reference, candidate)
        current_hash = sha256(DEPLOYMENT)
        if current_hash != deployment_hash:
            raise RuntimeError("Deployment checkpoint changed during V16")
        summary = {
            "version": "v16_uniform_model_soup_summary_v1",
            "created_at": now_iso(),
            "protocol": protocol,
            "soup_manifest": soup_manifest,
            "result": result,
            "decision": "freeze_soup_for_review_no_auto_deployment" if result["passed"] else "retain_v15_seed2_for_review",
            "candidate_checkpoint": str(SOUP.resolve()) if result["passed"] else v15["candidate_checkpoint"],
            "candidate_checkpoint_sha256": sha256(SOUP) if result["passed"] else sha256(Path(v15["candidate_checkpoint"])),
            "deployment_eligible": False,
            "deployment_write_allowed": False,
            "deployment_sha256_expected": DEPLOYMENT_SHA256,
            "deployment_sha256_current": current_hash,
            "deployment_unchanged": True,
        }
        write_json_atomic(SUMMARY, summary)
        state.update(state="complete", stage="complete", updated_at=now_iso(), decision=summary["decision"], summary=str(SUMMARY.resolve()))
        write_json_atomic(STATUS, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        state.update(state="failed", updated_at=now_iso(), error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(STATUS, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
