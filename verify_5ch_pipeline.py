"""Verify the real five-channel Data_v3 train/validation input pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset

from multispectral import CHANNEL_NAMES, PROJECT_ROOT
from train_ca_5ch import AUGMENTATION_PROFILES
from yolo_5ch_data import FiveChannelPhotometric, build_five_channel_dataset


def _transform_names(dataset) -> list[str]:
    return [transform.__class__.__name__ for transform in dataset.transforms.transforms]


def _batch_summary(batch: dict, split: str) -> dict[str, object]:
    images = batch["img"]
    if images.ndim != 4 or images.shape[1] != len(CHANNEL_NAMES):
        raise RuntimeError(f"{split} batch must be Bx5xHxW, got {tuple(images.shape)}")
    if images.dtype != torch.uint8:
        raise RuntimeError(f"{split} images must stay uint8 until trainer preprocessing, got {images.dtype}")
    if int(images.min()) < 0 or int(images.max()) > 255:
        raise RuntimeError(f"{split} batch is outside uint8 image range")

    channel_means = images.float().mean(dim=(0, 2, 3)) / 255.0
    labels = int(batch["cls"].numel())
    masks = batch.get("masks")
    mask_planes = int(masks.shape[0]) if masks is not None else 0
    if labels <= 0 or mask_planes <= 0:
        raise RuntimeError(f"{split} batch has no usable segmentation labels: labels={labels}, masks={mask_planes}")

    return {
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "range": [int(images.min()), int(images.max())],
        "channel_means": {
            name: round(float(mean), 6) for name, mean in zip(CHANNEL_NAMES, channel_means, strict=True)
        },
        "labels": labels,
        "mask_planes": mask_planes,
    }


def verify_pipeline(
    data_yaml: str | Path,
    *,
    profile_name: str = "spectral_aug",
    imgsz: int = 640,
    batch_size: int = 2,
) -> dict[str, object]:
    profile = AUGMENTATION_PROFILES[profile_name]
    overrides = {
        "task": "segment",
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch_size,
        "workers": 0,
        "cache": False,
        "rect": False,
    }
    overrides.update(profile["training"])
    cfg = get_cfg(overrides=overrides)
    data = check_det_dataset(str(data_yaml), autodownload=False)

    train_dataset = build_five_channel_dataset(
        cfg,
        data["train"],
        batch_size,
        data,
        mode="train",
        stride=32,
        photometric_augment=profile["photometric"],
        photometric_kwargs=profile["photometric_kwargs"],
    )
    val_dataset = build_five_channel_dataset(
        cfg,
        data["val"],
        batch_size,
        data,
        mode="val",
        rect=True,
        stride=32,
        photometric_augment=profile["photometric"],
        photometric_kwargs=profile["photometric_kwargs"],
    )

    train_names = _transform_names(train_dataset)
    val_names = _transform_names(val_dataset)
    train_count = sum(isinstance(transform, FiveChannelPhotometric) for transform in train_dataset.transforms.transforms)
    val_count = sum(isinstance(transform, FiveChannelPhotometric) for transform in val_dataset.transforms.transforms)
    expected_train_count = 1 if profile["photometric"] else 0
    if train_count != expected_train_count:
        raise RuntimeError(
            f"train pipeline expected {expected_train_count} FiveChannelPhotometric transform, got {train_count}"
        )
    if val_count:
        raise RuntimeError("validation pipeline must not contain FiveChannelPhotometric")
    if profile["photometric"] and train_names[-2:] != ["FiveChannelPhotometric", "Format"]:
        raise RuntimeError(f"photometric transform must be immediately before Format, got {train_names[-2:]}")

    train_loader = build_dataloader(train_dataset, batch_size, workers=0, shuffle=False)
    val_loader = build_dataloader(val_dataset, batch_size, workers=0, shuffle=False)
    result = {
        "profile": profile_name,
        "data": str(Path(data_yaml).resolve()),
        "dataset_sizes": {"train": len(train_dataset), "val": len(val_dataset)},
        "transforms": {"train": train_names, "val": val_names},
        "batches": {
            "train": _batch_summary(next(iter(train_loader)), "train"),
            "val": _batch_summary(next(iter(val_loader)), "val"),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "datasets" / "Data_v3" / "data_5ch_v3.yaml"),
    )
    parser.add_argument("--profile", choices=tuple(AUGMENTATION_PROFILES), default="spectral_aug")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    report = verify_pipeline(cli.data, profile_name=cli.profile, imgsz=cli.imgsz, batch_size=cli.batch)
    print(json.dumps(report, ensure_ascii=False, indent=2))
