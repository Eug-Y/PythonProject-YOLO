"""Validate the trained five-channel segmentation model on a YOLO split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_test_set import load_model
from multispectral import PROJECT_ROOT
from yolo_5ch_data import FiveChannelSegmentationValidator


DEFAULT_CHECKPOINT = Path(r"D:\PythonProject\runs\segment\land_seg\exp_ca_5ch_v11\weights\best.pt")
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "Data_v3" / "data_5ch_v3.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "Data_v3_model_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
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

    model = load_model(str(checkpoint), f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    validator = FiveChannelSegmentationValidator(
        save_dir=output,
        args={
            "model": str(checkpoint),
            "data": str(data),
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
    )
    metrics = validator(model=model)
    serializable = {key: float(value) for key, value in metrics.items()}
    (output / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))
    print(f"Metrics: {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
