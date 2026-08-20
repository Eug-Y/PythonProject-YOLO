"""Export a valid-only confidence sweep under an explicit mask-overlap evaluation contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics.nn.tasks import load_checkpoint

from yolo_multichannel_data import MultichannelSegmentationValidator, validate_channel_selection


class CaptureStatsValidator(MultichannelSegmentationValidator):
    """Capture concatenated prediction statistics before Ultralytics clears them."""

    raw_stats: dict[str, np.ndarray]

    def get_stats(self):
        self.raw_stats = {
            key: np.concatenate(values, axis=0) if values else np.asarray([])
            for key, values in self.metrics.stats.items()
        }
        return super().get_stats()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf-min", type=float, default=0.001)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--device", default="0")
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--overlap-mask", dest="overlap_mask", action="store_true")
    overlap.add_argument("--no-overlap-mask", dest="overlap_mask", action="store_false")
    parser.set_defaults(overlap_mask=False)
    return parser.parse_args()


def threshold_rows(
    confidence: np.ndarray,
    true_positive_mask: np.ndarray,
    target_count: int,
    thresholds: np.ndarray,
) -> list[dict[str, float | int]]:
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    true_positive_mask = np.asarray(true_positive_mask, dtype=bool).reshape(-1)
    if confidence.shape != true_positive_mask.shape:
        raise ValueError("Confidence and true-positive arrays must have identical shapes")
    rows = []
    for threshold in thresholds:
        selected = confidence >= float(threshold)
        predictions = int(selected.sum())
        tp = int(np.logical_and(selected, true_positive_mask).sum())
        fp = predictions - tp
        fn = int(target_count) - tp
        precision = tp / predictions if predictions else 1.0
        recall = tp / target_count if target_count else 1.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        rows.append({
            "confidence": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def main() -> int:
    args = parse_args()
    if not 0.0 < args.conf_min < 1.0 or not 0.0 < args.step < 1.0:
        raise ValueError("--conf-min and --step must be in (0, 1)")
    checkpoint = args.checkpoint.resolve()
    data = args.data.resolve()
    manifest_path = args.manifest.resolve()
    feature_root = args.feature_root.resolve()
    output = args.output.resolve()
    for path in (checkpoint, data, manifest_path, feature_root):
        if not path.exists():
            raise FileNotFoundError(path)
    data_text = data.read_text(encoding="utf-8").lower()
    if "test:" in data_text:
        raise RuntimeError(f"Confidence sweep YAML must not expose test: {data}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("held_out_split_used") is not False or manifest.get("selection_splits") != ["train", "valid"]:
        raise RuntimeError("Feature manifest does not preserve the sealed-test contract")
    all_channels = tuple(manifest["channels"])
    selected_channels = tuple(args.channels)
    validate_channel_selection(all_channels, selected_channels)
    if f"channels: {len(selected_channels)}" not in data_text:
        raise RuntimeError("Data YAML channel count does not match the selected channels")

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    model, _ = load_checkpoint(str(checkpoint), device=device)
    actual_channels = int(getattr(model, "input_channels", model.model[0].conv.in_channels))
    if actual_channels != len(selected_channels):
        raise RuntimeError(f"Checkpoint expects {actual_channels} channels, selected {len(selected_channels)}")
    model.float().to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    validator = CaptureStatsValidator(
        save_dir=output,
        feature_root=feature_root,
        all_channels=all_channels,
        selected_channels=selected_channels,
        args={
            "model": str(checkpoint),
            "data": str(data),
            "split": "val",
            "imgsz": args.imgsz,
            "batch": 1,
            "workers": 0,
            "device": args.device,
            "conf": args.conf_min,
            "overlap_mask": args.overlap_mask,
            "task": "segment",
            "mode": "val",
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": True,
        },
    )
    metrics = validator(model=model)
    raw = validator.raw_stats
    confidence = raw["conf"]
    true_positive_mask = raw["tp_m"][:, 0] if raw["tp_m"].ndim == 2 else np.zeros(len(confidence), dtype=bool)
    target_count = int(len(raw["target_cls"]))
    thresholds = np.arange(args.conf_min, 1.0, args.step, dtype=np.float64)
    rows = threshold_rows(confidence, true_positive_mask, target_count, thresholds)
    with (output / "confidence_sweep.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {key: float(value) for key, value in metrics.items()}
    best_fixed = max(rows, key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["confidence"])))
    curve = {
        "confidence": np.asarray(validator.metrics.seg.px).tolist(),
        "precision": np.asarray(validator.metrics.seg.p_curve)[0].tolist(),
        "recall": np.asarray(validator.metrics.seg.r_curve)[0].tolist(),
        "f1": np.asarray(validator.metrics.seg.f1_curve)[0].tolist(),
    }
    summary = {
        "version": "multichannel_valid_confidence_sweep_v1",
        "checkpoint": str(checkpoint),
        "data": str(data),
        "manifest": str(manifest_path),
        "feature_root": str(feature_root),
        "channels": list(selected_channels),
        "split": "val",
        "held_out_test_used": False,
        "imgsz": args.imgsz,
        "inference_confidence_min": args.conf_min,
        "threshold_step": args.step,
        "validation_overlap_mask": args.overlap_mask,
        "target_instances": target_count,
        "retained_predictions": int(len(confidence)),
        "aggregate_metrics": aggregate,
        "ultralytics_operating_point": {
            "precision": float(validator.metrics.seg.mp),
            "recall": float(validator.metrics.seg.mr),
            "f1": float(validator.metrics.seg.f1.mean()),
        },
        "best_fixed_iou50_operating_point": best_fixed,
        "curve": curve,
    }
    (output / "confidence_sweep.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "mask_map50_95": aggregate.get("metrics/mAP50-95(M)"),
        "ultralytics_operating_point": summary["ultralytics_operating_point"],
        "best_fixed_iou50_operating_point": best_fixed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
