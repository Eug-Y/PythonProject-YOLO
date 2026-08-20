"""Train one locked multichannel group ablation on train/valid only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from multispectral import PROJECT_ROOT
from train_ca_5ch import (
    SUPPORTED_SEGMENTATION_MODELS,
    build_ca_model_from_base,
    resolve_base_checkpoint,
    validate_attention_compatibility,
)
from yolo_multichannel_data import MultichannelSegmentationTrainer, validate_channel_selection


DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v4_candidates"
DEFAULT_MANIFEST = DEFAULT_FEATURE_ROOT / "manifest.json"
DEFAULT_GROUPS = PROJECT_ROOT / "configs" / "feature_ablation_groups.json"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "segment" / "land_seg"


ABLATION_AUGMENTATION = {
    # Identical geometry for every group. Built-in HSV is disabled because
    # Data_v4 already stores RGB and three-channel-only HSV would be unfair.
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.0,
    "close_mosaic": 10,
    "scale": 0.5,
    "translate": 0.1,
}


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_train_sampling_weights(path: Path) -> tuple[dict[str, float], dict[str, object]]:
    payload = read_json(path)
    if payload.get("source_split") != "train":
        raise ValueError(f"Sampling weights must be train-only: source_split={payload.get('source_split')!r}")
    if payload.get("held_out_split_used") is not False or payload.get("validation_split_used") is not False:
        raise ValueError("Sampling weights must explicitly confirm that valid/test were not used")
    raw_weights = payload.get("weights_by_image")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(f"Sampling weights are missing weights_by_image: {path}")
    weights = {str(name): float(weight) for name, weight in raw_weights.items()}
    values = np.asarray(list(weights.values()), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Sampling weights must be finite and strictly positive")
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": file_sha256,
        "version": payload.get("version"),
        "source_split": "train",
        "held_out_split_used": False,
        "validation_split_used": False,
        "sampling": payload.get("sampling"),
    }
    return weights, metadata


def adapt_first_conv_channels(model, channels: int, extra_init: str = "mean"):
    """Expand a pretrained stem while preserving every existing input kernel."""
    if channels < 3:
        raise ValueError(f"At least three RGB channels are required, got {channels}")
    if extra_init not in {"mean", "zero", "kaiming"}:
        raise ValueError(f"Unsupported extra-channel initialization: {extra_init}")
    first = model.model[0]
    old_conv = first.conv
    old_channels = int(old_conv.in_channels)
    if old_channels < 3:
        raise ValueError(f"Expected a stem with at least three RGB channels, got {old_channels}")
    if channels < old_channels:
        raise ValueError(f"Channel expansion cannot shrink a {old_channels}-channel stem to {channels}")
    if channels == old_channels:
        print(f"First Conv: keep {old_channels} input channels unchanged")
        return model

    old_weight = old_conv.weight.detach()
    new_conv = nn.Conv2d(
        channels,
        old_conv.out_channels,
        old_conv.kernel_size,
        old_conv.stride,
        old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight[:, :old_channels].copy_(old_weight)
        extra = new_conv.weight[:, old_channels:]
        if extra_init == "mean":
            extra.copy_(old_weight.mean(dim=1, keepdim=True).expand_as(extra))
        elif extra_init == "zero":
            extra.zero_()
        else:
            nn.init.kaiming_uniform_(extra, a=1.0)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    first.conv = new_conv
    print(
        f"First Conv: {old_channels} -> {channels} channels; preserved {old_channels} kernels and "
        f"initialized {channels - old_channels} new kernels with {extra_init}"
    )
    return model


def verify_forward(model, channels: int, imgsz: int) -> None:
    print(f"验证模型前向传播: input=(1, {channels}, {imgsz}, {imgsz})")
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, channels, imgsz, imgsz))
    if output is None:
        raise RuntimeError("Model forward returned None")
    print("模型前向传播 OK")


def write_feature_contract(
    save_dir: Path,
    *,
    group: str,
    channels: Sequence[str],
    manifest: Path,
    groups_config: Path,
    args: dict,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = read_json(manifest)
    manifest_version = str(manifest_payload.get("version", "multichannel_candidates"))
    contract = {
        "version": f"{manifest_version}_model_feature_contract_v1",
        "candidate_manifest_version": manifest_version,
        "selection_group": group,
        "channels": list(channels),
        "channel_count": len(channels),
        "candidate_manifest": str(manifest.resolve()),
        "group_config": str(groups_config.resolve()),
        "selection_splits": ["train", "valid"],
        "held_out_split_used": False,
        "held_out_split": "test",
        "rgb_storage_order": "RGB",
        "extra_channel_initialization": args["extra_init"],
        "attention": args["attention"],
        "base_model": args["base_model"],
        "initialization_checkpoint": args.get("initialization_checkpoint"),
        "seed": args["seed"],
        "imgsz": args["imgsz"],
        "epochs": args["epochs"],
        "train_sampling": args.get("train_sampling", {"enabled": False}),
        "augmentation": args.get("augmentation"),
        "mask_supervision": args.get("mask_supervision", {"mask_ratio": 4, "overlap_mask": True}),
    }
    with (save_dir / "feature_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, help="Group key in configs/feature_ablation_groups.json")
    parser.add_argument("--groups-config", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Channel-count YAML; selected automatically from the group by default.",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--architecture", choices=("yolo11", "yolov8"), default="yolo11")
    parser.add_argument("--model-size", choices=("n", "s"), default="n")
    parser.add_argument(
        "--base-model",
        choices=SUPPORTED_SEGMENTATION_MODELS,
        default=None,
        help="Explicit base model; overrides --architecture and --model-size.",
    )
    parser.add_argument("--attention", choices=("ca_eca", "none"), default="ca_eca")
    parser.add_argument("--extra-init", choices=("mean", "zero", "kaiming"), default="mean")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None, help="Defaults to a conservative value for the base model.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--mosaic", type=float, default=None, help="Override the locked Mosaic probability.")
    parser.add_argument("--mixup", type=float, default=None, help="Override the locked MixUp probability.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Override the final no-Mosaic epoch count.")
    parser.add_argument(
        "--mask-ratio",
        type=int,
        choices=(1, 2, 4, 8),
        default=4,
        help="Mask target downsampling ratio; lower values retain finer boundary supervision.",
    )
    parser.add_argument(
        "--overlap-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train overlapping instances as a single indexed mask (Ultralytics default).",
    )
    parser.add_argument(
        "--sampling-weights",
        type=Path,
        default=None,
        help="Train-only JSON weights for deterministic WeightedRandomSampler; validation remains unchanged.",
    )
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize a fresh run from a trained checkpoint; unlike --resume, optimizer and epoch state are reset.",
    )
    return parser.parse_args()


def resolve_selection_yaml(feature_root: Path, manifest: dict, channel_count: int) -> Path:
    """Resolve a versioned train/valid-only YAML without assuming Data_v4."""
    version = str(manifest.get("version", "data_v4_candidates"))
    prefix = version.split("_candidates", 1)[0].lower()
    candidates = (
        feature_root / f"{prefix}_selection_{channel_count}ch.yaml",
        PROJECT_ROOT / "configs" / f"{prefix}_selection_{channel_count}ch.yaml",
        feature_root / f"data_v4_selection_{channel_count}ch.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No train/valid selection YAML for {channel_count} channels; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def main() -> None:
    from ultralytics.nn.tasks import SegmentationModel, load_checkpoint

    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    if cli.resume and cli.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    sampling_weights: dict[str, float] | None = None
    sampling_metadata: dict[str, object] = {"enabled": False}
    if cli.sampling_weights is not None:
        if not cli.sampling_weights.is_file():
            raise FileNotFoundError(f"Sampling weights not found: {cli.sampling_weights}")
        sampling_weights, sampling_metadata = load_train_sampling_weights(cli.sampling_weights.resolve())
    manifest = read_json(cli.manifest)
    group_config = read_json(cli.groups_config)
    groups = group_config.get("groups", {})
    if cli.group not in groups:
        raise ValueError(f"Unknown group {cli.group!r}; available={sorted(groups)}")
    all_channels = tuple(manifest["channels"])
    selected_channels = tuple(groups[cli.group])
    validate_channel_selection(all_channels, selected_channels)
    data_path = cli.data or resolve_selection_yaml(cli.feature_root, manifest, len(selected_channels))
    data_text = data_path.read_text(encoding="utf-8")
    if data_text.lower().find("test:") >= 0:
        raise RuntimeError(f"Feature-selection YAML must not expose test: {data_path}")
    expected_channel_line = f"channels: {len(selected_channels)}"
    if expected_channel_line not in data_text:
        raise RuntimeError(f"{data_path} must declare {expected_channel_line!r}")
    fraction = cli.fraction if cli.fraction is not None else (0.05 if cli.smoke_test else 1.0)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {fraction}")
    for name, value in (("mosaic", cli.mosaic), ("mixup", cli.mixup)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}")
    if cli.close_mosaic is not None and cli.close_mosaic < 0:
        raise ValueError(f"--close-mosaic must be non-negative, got {cli.close_mosaic}")

    epochs = 1 if cli.smoke_test else cli.epochs
    cache = False if cli.smoke_test else cli.cache
    workers = 0 if cli.smoke_test else cli.workers
    base_checkpoint = resolve_base_checkpoint(cli.base_model, cli.architecture, cli.model_size)
    validate_attention_compatibility(base_checkpoint, cli.attention)
    if cli.init_checkpoint and not cli.init_checkpoint.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {cli.init_checkpoint}")
    model_tag = Path(base_checkpoint).stem
    if cli.batch is not None and cli.batch < 1:
        raise ValueError(f"--batch must be positive, got {cli.batch}")
    if cli.batch is not None:
        batch = cli.batch
    elif base_checkpoint.startswith("yolov9") or "s-seg" in base_checkpoint:
        batch = 1
    else:
        batch = 4
    experiment_name = cli.name or f"ablate_{cli.group}_{model_tag}_seed{cli.seed}_v1"
    if cli.smoke_test and cli.name is None:
        experiment_name = f"smoke_{experiment_name}"
    overrides = {
        "model": base_checkpoint,
        "data": str(data_path.resolve()),
        "epochs": epochs,
        "imgsz": cli.imgsz,
        "batch": batch,
        "device": cli.device,
        "workers": workers,
        "patience": cli.patience,
        "optimizer": "SGD",
        "lr0": 0.002,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": (
            cli.warmup_epochs if cli.warmup_epochs is not None else min(3.0, max(1.0, epochs / 10.0))
        ),
        "cos_lr": True,
        "amp": True,
        "cache": cache,
        "fraction": fraction,
        "save": True,
        "save_period": 10,
        "project": str(cli.project.resolve()),
        "name": experiment_name,
        "exist_ok": False,
        "seed": cli.seed,
        "deterministic": cli.deterministic,
        "mask_ratio": cli.mask_ratio,
        "overlap_mask": cli.overlap_mask,
        "verbose": True,
    }
    resolved_augmentation = dict(ABLATION_AUGMENTATION)
    if cli.mosaic is not None:
        resolved_augmentation["mosaic"] = cli.mosaic
    if cli.mixup is not None:
        resolved_augmentation["mixup"] = cli.mixup
    if cli.close_mosaic is not None:
        resolved_augmentation["close_mosaic"] = cli.close_mosaic
    overrides.update(resolved_augmentation)

    print(f"\n===== {manifest.get('version', 'multichannel')} 分组消融 =====")
    print(f"组: {cli.group}  通道({len(selected_channels)}): {', '.join(selected_channels)}")
    print(f"数据: train + valid；test 未暴露、未读取")
    print(
        f"模型: {base_checkpoint}  attention={cli.attention}  init={cli.extra_init}  "
        f"imgsz={cli.imgsz} epochs={epochs} batch={batch} seed={cli.seed}"
    )

    trainer_kwargs = {
        "feature_root": cli.feature_root,
        "all_channels": all_channels,
        "selected_channels": selected_channels,
        "photometric_augment": False,
        "train_sampling_weights": sampling_weights,
    }
    if cli.resume:
        overrides["resume"] = str(cli.resume.resolve())
        trainer = MultichannelSegmentationTrainer(overrides=overrides, **trainer_kwargs)
        write_feature_contract(
            Path(trainer.save_dir),
            group=cli.group,
            channels=selected_channels,
            manifest=cli.manifest,
            groups_config=cli.groups_config,
            args={
                "extra_init": cli.extra_init,
                "attention": cli.attention,
                "base_model": base_checkpoint,
                "initialization_checkpoint": None,
                "seed": cli.seed,
                "imgsz": cli.imgsz,
                "epochs": epochs,
                "train_sampling": sampling_metadata,
                "augmentation": resolved_augmentation,
                "mask_supervision": {"mask_ratio": cli.mask_ratio, "overlap_mask": cli.overlap_mask},
            },
        )
        trainer.train()
        return

    if cli.init_checkpoint:
        model, _ = load_checkpoint(str(cli.init_checkpoint.resolve()))
        actual_channels = int(model.model[0].conv.in_channels)
        if actual_channels > len(selected_channels):
            raise ValueError(
                f"Initialization checkpoint expects {actual_channels} channels, "
                f"but group {cli.group!r} selects only {len(selected_channels)}"
            )
        checkpoint_has_attention = any(module.__class__.__name__ in {"CoordAtt", "ECA"} for module in model.modules())
        requested_attention = cli.attention == "ca_eca"
        if checkpoint_has_attention != requested_attention:
            raise ValueError(
                "Initialization checkpoint attention structure does not match --attention "
                f"({checkpoint_has_attention=} versus {cli.attention!r})"
            )
        model.float()
        if actual_channels < len(selected_channels):
            model = adapt_first_conv_channels(model, len(selected_channels), cli.extra_init)
    else:
        checkpoint_model, _ = load_checkpoint(base_checkpoint)
        model = SegmentationModel(checkpoint_model.yaml, nc=1)
        model.load(checkpoint_model)
        if cli.attention == "ca_eca":
            model = build_ca_model_from_base(model)
        model = adapt_first_conv_channels(model, len(selected_channels), cli.extra_init)
    verify_forward(model, len(selected_channels), min(cli.imgsz, 320 if cli.verify_only else 640))
    if cli.verify_only:
        return

    trainer = MultichannelSegmentationTrainer(overrides=overrides, **trainer_kwargs)
    trainer.model = model
    write_feature_contract(
        Path(trainer.save_dir),
        group=cli.group,
        channels=selected_channels,
        manifest=cli.manifest,
        groups_config=cli.groups_config,
        args={
            "extra_init": cli.extra_init,
            "attention": cli.attention,
            "base_model": base_checkpoint,
            "initialization_checkpoint": str(cli.init_checkpoint.resolve()) if cli.init_checkpoint else None,
            "seed": cli.seed,
            "imgsz": cli.imgsz,
            "epochs": epochs,
            "train_sampling": sampling_metadata,
            "augmentation": resolved_augmentation,
            "mask_supervision": {"mask_ratio": cli.mask_ratio, "overlap_mask": cli.overlap_mask},
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
