"""Validate a frozen configurable-channel checkpoint on one YOLO split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from custom_modules import CoordAtt, ECA  # noqa: F401 - needed when unpickling custom checkpoints
from multispectral import PROJECT_ROOT
from yolo_multichannel_data import MultichannelSegmentationValidator, validate_channel_selection


DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v4_candidates"
DEFAULT_MANIFEST = DEFAULT_FEATURE_ROOT / "manifest.json"
DEFAULT_GROUPS = PROJECT_ROOT / "configs" / "feature_ablation_groups.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--groups-config", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    from ultralytics.nn.tasks import load_checkpoint

    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    data_path = args.data.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data_path}")
    groups = read_json(args.groups_config)["groups"]
    if args.group not in groups:
        raise SystemExit(f"Unknown channel group: {args.group}")
    all_channels = tuple(read_json(args.manifest)["channels"])
    selected_channels = tuple(groups[args.group])
    validate_channel_selection(all_channels, selected_channels)
    data_text = data_path.read_text(encoding="utf-8")
    expected = f"channels: {len(selected_channels)}"
    if expected not in data_text:
        raise SystemExit(f"Dataset YAML does not declare {expected!r}: {data_path}")
    if args.split == "test" and "test:" not in data_text.lower():
        raise SystemExit(f"Test split is not exposed by {data_path}")

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    model, _ = load_checkpoint(str(checkpoint), device=device)
    model.criterion = None
    model.float().to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    validator = MultichannelSegmentationValidator(
        save_dir=output,
        args={
            "model": str(checkpoint),
            "data": str(data_path),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "task": "segment",
            "mode": "val",
            "plots": True,
            "save_json": False,
            "save_txt": False,
            "verbose": True,
        },
        feature_root=args.feature_root,
        all_channels=all_channels,
        selected_channels=selected_channels,
    )
    metrics = validator(model=model)
    serializable = {key: float(value) for key, value in metrics.items()}
    (output / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    evaluation = {
        "checkpoint": str(checkpoint),
        "checkpoint_feature_contract": str(checkpoint.parent.parent / "feature_contract.json"),
        "data": str(data_path),
        "feature_root": str(args.feature_root.resolve()),
        "split": args.split,
        "channels": list(selected_channels),
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

