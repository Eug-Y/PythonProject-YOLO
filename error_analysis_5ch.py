"""Export per-image box/mask errors and all validation plots for a 5-channel model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from multispectral import PROJECT_ROOT
from ultralytics.nn.tasks import load_checkpoint
from yolo_5ch_data import FiveChannelSegmentationValidator


class FullPlotFiveChannelValidator(FiveChannelSegmentationValidator):
    """Keep validation plots for every batch instead of only the first three."""

    def update_metrics(self, preds, batch) -> None:
        super().update_metrics(preds, batch)
        if self.args.plots and self.batch_i >= 3:
            self.plot_val_samples(batch, self.batch_i)
            self.plot_predictions(batch, preds, self.batch_i)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "segment"
        / "land_seg"
        / "exp_ca_5ch_grouped_reviewed_v1"
        / "weights"
        / "best.pt",
    )
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3" / "data_5ch_v3.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "Data_v3_recommended_model_errors")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25, help="Deployment confidence threshold for error ranking.")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    data = args.data.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if not data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data}")
    output.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    model, _ = load_checkpoint(str(checkpoint), device=device)
    actual_channels = int(model.model[0].conv.in_channels)
    if actual_channels != 5:
        raise RuntimeError(f"Expected a 5-channel checkpoint, got {actual_channels}: {checkpoint}")
    model.float().to(device).eval()
    validator = FullPlotFiveChannelValidator(
        save_dir=output,
        args={
            "model": str(checkpoint),
            "data": str(data),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": 1,
            "workers": 0,
            "device": args.device,
            "conf": args.conf,
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
    (output / "analysis_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "data": str(data),
                "split": args.split,
                "imgsz": args.imgsz,
                "batch": 1,
                "confidence": args.conf,
                "note": "Per-image TP/FP/FN and F1 use the deployment confidence threshold.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    box_by_image = validator.metrics.box.image_metrics
    mask_by_image = validator.metrics.seg.image_metrics
    names = sorted(set(box_by_image) | set(mask_by_image))
    per_image = {
        name: {
            "box": box_by_image.get(name, {}),
            "mask": mask_by_image.get(name, {}),
        }
        for name in names
    }
    per_image = dict(
        sorted(
            per_image.items(),
            key=lambda item: (item[1]["mask"].get("f1", 1.0), item[0]),
        )
    )
    (output / "per_image_metrics.json").write_text(
        json.dumps(per_image, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "image",
        "mask_f1",
        "mask_precision",
        "mask_recall",
        "mask_tp",
        "mask_fp",
        "mask_fn",
        "box_f1",
        "box_precision",
        "box_recall",
        "box_tp",
        "box_fp",
        "box_fn",
    ]
    with (output / "per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, values in per_image.items():
            mask = values["mask"]
            box = values["box"]
            writer.writerow(
                {
                    "image": name,
                    **{f"mask_{key}": mask.get(key) for key in ("f1", "precision", "recall", "tp", "fp", "fn")},
                    **{f"box_{key}": box.get(key) for key in ("f1", "precision", "recall", "tp", "fp", "fn")},
                }
            )

    print(f"images={len(per_image)}")
    print(f"metrics={output / 'per_image_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
