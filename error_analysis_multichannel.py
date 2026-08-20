"""Export validation-only per-image errors and plots for any manifest-selected channel group."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics.nn.tasks import load_checkpoint

from multispectral import PROJECT_ROOT
from yolo_multichannel_data import MultichannelSegmentationValidator, validate_channel_selection


class FullPlotMultichannelValidator(MultichannelSegmentationValidator):
    """Keep validation plots for every batch instead of only the first three."""

    def update_metrics(self, preds, batch) -> None:
        super().update_metrics(preds, batch)
        if self.args.plots and self.batch_i >= 3:
            self.plot_val_samples(batch, self.batch_i)
            self.plot_predictions(batch, preds, self.batch_i)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly authorize a locked one-time test export. Validation remains the default.",
    )
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--overlap-mask", dest="overlap_mask", action="store_true")
    overlap.add_argument("--no-overlap-mask", dest="overlap_mask", action="store_false")
    parser.set_defaults(overlap_mask=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    data = args.data.resolve()
    manifest_path = args.manifest.resolve()
    feature_root = args.feature_root.resolve()
    output = args.output.resolve()
    for path in (checkpoint, data, manifest_path, feature_root):
        if not path.exists():
            raise FileNotFoundError(path)
    data_text = data.read_text(encoding="utf-8").lower()
    if args.split == "val" and "test:" in data_text:
        raise RuntimeError(f"Validation analysis YAML must not expose test: {data}")
    if args.split == "test" and (not args.allow_test or "test:" not in data_text):
        raise RuntimeError("Test analysis requires both --allow-test and a YAML exposing test:")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.split == "val" and "test" in manifest.get("split_counts", {}):
        raise RuntimeError("Validation analysis manifest must not contain test features")
    if args.split == "test" and manifest.get("split_counts", {}).get("test") != 30:
        raise RuntimeError("Locked test analysis requires exactly 30 manifest-declared test features")
    all_channels = tuple(manifest["channels"])
    selected_channels = tuple(args.channels)
    validate_channel_selection(all_channels, selected_channels)
    expected = f"channels: {len(selected_channels)}"
    if expected not in data_text:
        raise RuntimeError(f"{data} must declare {expected!r}")

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    model, _ = load_checkpoint(str(checkpoint), device=device)
    actual_channels = int(getattr(model, "input_channels", model.model[0].conv.in_channels))
    if actual_channels != len(selected_channels):
        raise RuntimeError(
            f"Checkpoint expects {actual_channels} channels, selected {len(selected_channels)}: {checkpoint}"
        )
    model.float().to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    validator = FullPlotMultichannelValidator(
        save_dir=output,
        feature_root=feature_root,
        all_channels=all_channels,
        selected_channels=selected_channels,
        args={
            "model": str(checkpoint),
            "data": str(data),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": 1,
            "workers": 0,
            "device": args.device,
            "conf": args.conf,
            "overlap_mask": args.overlap_mask,
            "task": "segment",
            "mode": "val",
            "plots": True,
            "save_json": False,
            "save_txt": False,
            "verbose": True,
        },
    )
    metrics = validator(model=model)
    aggregate = {key: float(value) for key, value in metrics.items()}
    (output / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    config = {
        "version": "multichannel_validation_error_analysis_v1",
        "checkpoint": str(checkpoint),
        "data": str(data),
        "manifest": str(manifest_path),
        "feature_root": str(feature_root),
        "channels": list(selected_channels),
        "split": args.split,
        "held_out_test_used": args.split == "test",
        "imgsz": args.imgsz,
        "batch": 1,
        "confidence": args.conf,
        "validation_overlap_mask": args.overlap_mask,
    }
    (output / "analysis_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    box_by_image = validator.metrics.box.image_metrics
    mask_by_image = validator.metrics.seg.image_metrics
    names = sorted(set(box_by_image) | set(mask_by_image))
    per_image = {
        name: {"box": box_by_image.get(name, {}), "mask": mask_by_image.get(name, {})}
        for name in names
    }
    per_image = dict(sorted(per_image.items(), key=lambda item: (item[1]["mask"].get("f1", 1.0), item[0])))
    (output / "per_image_metrics.json").write_text(
        json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "image", "mask_f1", "mask_precision", "mask_recall", "mask_tp", "mask_fp", "mask_fn",
        "box_f1", "box_precision", "box_recall", "box_tp", "box_fp", "box_fn",
    ]
    with (output / "per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in per_image.items():
            mask = values["mask"]
            box = values["box"]
            writer.writerow({
                "image": name,
                **{f"mask_{key}": mask.get(key) for key in ("f1", "precision", "recall", "tp", "fp", "fn")},
                **{f"box_{key}": box.get(key) for key in ("f1", "precision", "recall", "tp", "fp", "fn")},
            })
    print(f"images={len(per_image)}")
    print(f"metrics={output / 'per_image_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
