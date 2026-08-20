"""One-shot final release gate on sealed human blind labels.

Use ``--prepare`` before annotation is complete to freeze the protocol.  The
actual evaluation refuses unsealed labels and consumes the blind budget once.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from calibrate_yolo_ensemble import load_yolo_ground_truth, match_instances
from deployment_fusion import infer_shadow_components, load_shadow_config, sha256
from release_blind_annotation import BlindAnnotationStore, DEFAULT_DATASET, now_iso, verify_selection
from run_feature_ablations import write_json_atomic
from yolo_multichannel_data import _load_selected_array, validate_channel_selection


ROOT = Path(__file__).resolve().parent
GUARD = ROOT / "logs" / "v27_release_blind_evaluation_guard.json"
SUMMARY = ROOT / "logs" / "v27_release_blind_final.summary.json"
SMOKE = ROOT / "logs" / "v27_shadow_deployment_smoke.summary.json"


def protocol() -> dict[str, object]:
    shadow = load_shadow_config()
    selection = verify_selection(DEFAULT_DATASET)
    return {
        "version": "v27_release_blind_final_protocol_v1",
        "preregistered_at": now_iso(),
        "candidate": shadow["model_name"],
        "primary_architecture": "YOLO11n-seg",
        "auxiliary_architecture": "YOLOv8n-seg",
        "primary_sha256": shadow["primary"]["sha256"],
        "auxiliary_sha256": shadow["auxiliary"]["sha256"],
        "fusion_config_sha256": shadow["fusion_config_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "images": 30,
        "ground_truth_requirement": "sealed manual RGB-only labels; no pseudo labels",
        "evaluation_limit": 1,
        "gate": {
            "fusion_f1_delta_vs_incumbent_min": 0.0,
            "precision_delta_vs_incumbent_min": -0.01,
            "recall_delta_vs_incumbent_min": -0.01,
            "mean_matched_iou_delta_vs_incumbent_min": -0.01,
        },
        "every_output_requires_yolo11": True,
        "configuration_search_allowed": False,
        "result_may_not_be_used_for_further_tuning": True,
    }


def prepare() -> dict:
    frozen = protocol()
    if GUARD.is_file():
        guard = json.loads(GUARD.read_text(encoding="utf-8"))
        existing = guard.get("protocol", {})
        immutable_keys = (
            "version", "candidate", "primary_sha256", "auxiliary_sha256",
            "fusion_config_sha256", "selection_manifest_sha256", "gate",
        )
        if any(existing.get(key) != frozen.get(key) for key in immutable_keys):
            raise RuntimeError("Existing blind-evaluation protocol does not match frozen V27 artifacts")
        return guard
    guard = {
        "version": "v27_release_blind_evaluation_guard_v1",
        "state": "registered",
        "evaluation_count": 0,
        "protocol": frozen,
        "updated_at": now_iso(),
    }
    write_json_atomic(GUARD, guard)
    return guard


def _finalize(counter: dict) -> dict[str, float | int]:
    tp, fp, fn = int(counter["tp"]), int(counter["fp"]), int(counter["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ious = counter["matched_ious"]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_matched_iou": statistics.fmean(ious) if ious else 0.0,
    }


def evaluate_once() -> dict:
    guard = prepare()
    if guard.get("evaluation_count") != 0 or guard.get("state") != "registered" or SUMMARY.exists():
        raise RuntimeError("V27 release blind evaluation budget has already been consumed")
    store = BlindAnnotationStore(DEFAULT_DATASET, verify_hashes=True)
    status = store.status()
    if not (
        status.get("total") == 30
        and status.get("completed") == 30
        and status.get("remaining") == 0
        and status.get("drafts") == 0
        and status.get("sealed") is True
        and status.get("can_undo") is False
    ):
        raise RuntimeError(f"Blind manual labels are not complete and sealed: {status}")
    seal = json.loads(store.seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("state") != "sealed_manual_ground_truth"
        or seal.get("annotation_mode") != "manual_rgb_only"
        or seal.get("model_predictions_exposed") is not False
        or seal.get("pseudo_labels_used") is not False
    ):
        raise RuntimeError("Blind annotation seal is not eligible for final release evaluation")
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    if smoke.get("state") != "passed" or smoke.get("production_unchanged") is not True:
        raise RuntimeError("Shadow deployment smoke gate must pass before the blind release gate")

    guard.update(
        state="running",
        evaluation_count=1,
        started_at=now_iso(),
        updated_at=now_iso(),
        annotation_seal_sha256=sha256(store.seal_path),
    )
    write_json_atomic(GUARD, guard)
    counters = {
        name: {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        for name in ("fusion", "primary_yolo11", "incumbent_yolov8")
    }
    try:
        shadow = load_shadow_config()
        feature_manifest = json.loads(
            (ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json").read_text(encoding="utf-8")
        )
        indices = validate_channel_selection(tuple(feature_manifest["channels"]), tuple(shadow["channels"]))
        source_count = 0
        for stem in store.stems:
            image_path = store.images_dir / f"{stem}.JPG"
            image_5ch = _load_selected_array(
                store.dataset_root / "test" / "features" / f"{stem}.npy",
                indices,
                len(feature_manifest["channels"]),
            )
            image_hw = tuple(image_5ch.shape[:2])
            targets = load_yolo_ground_truth(store.labels_dir / f"{stem}.txt", image_hw)
            components, fused, _metadata = infer_shadow_components(image_5ch)
            if any("yolo11_primary" not in item.sources for item in fused):
                raise RuntimeError(f"YOLO11 ownership invariant failed on {stem}")
            source_count += len(fused)
            variants = {
                "fusion": ([item.mask for item in fused], [item.bbox for item in fused]),
                "primary_yolo11": ([item.mask for item in components[0]], [item.bbox for item in components[0]]),
                "incumbent_yolov8": ([item.mask for item in components[1]], [item.bbox for item in components[1]]),
            }
            for name, (masks, bboxes) in variants.items():
                matched = match_instances(masks, targets, prediction_bboxes=bboxes)
                counters[name]["tp"] += int(matched["tp"])
                counters[name]["fp"] += int(matched["fp"])
                counters[name]["fn"] += int(matched["fn"])
                counters[name]["matched_ious"].extend(matched["matched_ious"])
        metrics = {name: _finalize(counter) for name, counter in counters.items()}
        fusion, incumbent = metrics["fusion"], metrics["incumbent_yolov8"]
        values = {
            "fusion_f1_delta_vs_incumbent": fusion["f1"] - incumbent["f1"],
            "precision_delta_vs_incumbent": fusion["precision"] - incumbent["precision"],
            "recall_delta_vs_incumbent": fusion["recall"] - incumbent["recall"],
            "mean_matched_iou_delta_vs_incumbent": fusion["mean_matched_iou"] - incumbent["mean_matched_iou"],
        }
        frozen_gate = guard["protocol"]["gate"]
        checks = {
            "f1_not_below_incumbent": values["fusion_f1_delta_vs_incumbent"] >= frozen_gate["fusion_f1_delta_vs_incumbent_min"],
            "precision_guard": values["precision_delta_vs_incumbent"] >= frozen_gate["precision_delta_vs_incumbent_min"],
            "recall_guard": values["recall_delta_vs_incumbent"] >= frozen_gate["recall_delta_vs_incumbent_min"],
            "matched_iou_guard": values["mean_matched_iou_delta_vs_incumbent"] >= frozen_gate["mean_matched_iou_delta_vs_incumbent_min"],
            "yolo11_ownership": source_count >= 0,
        }
        passed = all(checks.values())
        summary = {
            "version": "v27_release_blind_final_summary_v1",
            "created_at": now_iso(),
            "protocol": guard["protocol"],
            "annotation_seal_sha256": guard["annotation_seal_sha256"],
            "evaluation_count": 1,
            "images": len(store.stems),
            "metrics": metrics,
            "values": values,
            "checks": checks,
            "gate_passed": passed,
            "decision": "promote_v27_to_production" if passed else "reject_v27_keep_incumbent_terminal",
            "production_switch_allowed": passed,
            "result_may_not_be_used_for_further_tuning": True,
        }
        write_json_atomic(SUMMARY, summary)
        guard.update(
            state="complete",
            completed_at=now_iso(),
            updated_at=now_iso(),
            decision=summary["decision"],
            summary_sha256=sha256(SUMMARY),
        )
        write_json_atomic(GUARD, guard)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    except BaseException as exc:
        guard.update(
            state="terminal_failed",
            failed_at=now_iso(),
            updated_at=now_iso(),
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json_atomic(GUARD, guard)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
        return 0
    summary = evaluate_once()
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
