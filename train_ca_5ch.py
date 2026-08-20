"""
YOLO11 + CoordAtt + ECA 5 通道多光谱训练脚本
通道: [R, G, B, NDVI, NDRE]
加载 _5ch.npy 替代 JPG，第一层 Conv 3→5 通道
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from custom_modules import CoordAtt, ECA
from multispectral import PROJECT_ROOT
from yolo_5ch_data import FiveChannelSegmentationTrainer


SUPPORTED_SEGMENTATION_MODELS = (
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolo12n-seg.pt",
    "yolo12s-seg.pt",
    "yolo26n-seg.pt",
    "yolo26s-seg.pt",
    "yolov9c-seg.pt",
)


def resolve_base_checkpoint(base_model: str | None, architecture: str, model_size: str) -> str:
    """Resolve a supported pretrained segmentation checkpoint name."""
    checkpoint = base_model or f"{architecture}{model_size}-seg.pt"
    if checkpoint not in SUPPORTED_SEGMENTATION_MODELS:
        raise ValueError(
            f"Unsupported segmentation base model {checkpoint!r}; "
            f"expected one of {SUPPORTED_SEGMENTATION_MODELS}"
        )
    return checkpoint


def validate_attention_compatibility(base_checkpoint: str, attention: str) -> None:
    """Reject known backbone/attention combinations before a long setup."""
    if base_checkpoint.startswith("yolov9") and attention == "ca_eca":
        raise ValueError(
            "YOLOv9-seg uses RepNCSPELAN4 blocks and is not compatible with the current "
            "C2f/C3k2 CoordAtt+ECA insertion; use --attention none"
        )


AUGMENTATION_PROFILES = {
    "baseline": {
        "training": {
            # Kept for exact comparison with the previous run. Ultralytics
            # skips RandomHSV for five-channel arrays.
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.0,
            "close_mosaic": 10,
            "scale": 0.5,
            "translate": 0.1,
        },
        "photometric": False,
        "photometric_kwargs": {},
    },
    "spectral_only_mild": {
        "training": {
            # Match the successful baseline geometry exactly. Built-in HSV
            # remains ineffective for HxWx5 inputs, so the only active change
            # from baseline is the mild index-channel transform below.
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.0,
            "close_mosaic": 10,
            "scale": 0.5,
            "translate": 0.1,
        },
        "photometric": True,
        "photometric_kwargs": {
            "rgb_probability": 0.0,
            "spectral_probability": 0.3,
            "spectral_gain": 0.015,
            "spectral_bias": 0.005,
            "spectral_noise": 0.002,
        },
    },
    "spectral_aug": {
        "training": {
            # Disable the built-in RGB-only transform and use the explicit
            # five-channel transform in yolo_5ch_data.py instead.
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 10.0,
            "fliplr": 0.5,
            "flipud": 0.5,
            "mosaic": 0.8,
            "mixup": 0.0,
            "copy_paste": 0.1,
            "close_mosaic": 15,
            "scale": 0.4,
            "translate": 0.1,
        },
        "photometric": True,
        "photometric_kwargs": {
            "rgb_probability": 0.8,
            "hue_gain": 0.01,
            "saturation_gain": 0.25,
            "value_gain": 0.20,
            "spectral_probability": 0.5,
            "spectral_gain": 0.03,
            "spectral_bias": 0.01,
            "spectral_noise": 0.005,
        },
    },
}


# ============================================================================
# 模型层工具
# ============================================================================
def _get_output_channels(layer):
    if hasattr(layer, "out_channels"):
        return layer.out_channels
    if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"):
        return layer.cv2.conv.out_channels
    if hasattr(layer, "cv3") and hasattr(layer.cv3, "conv"):
        return layer.cv3.conv.out_channels
    if hasattr(layer, "conv"):
        return layer.conv.out_channels
    return None


def build_ca_model_from_base(detection_model):
    """在 YOLO11 C3k2 或 YOLOv8 C2f 主干块后插入 CoordAtt + ECA。"""
    old_layers = list(detection_model.model)
    backbone_end = 7  # 两个架构的主干目标块均位于索引 2/4/6

    print("插入 CoordAtt + ECA（残差版）...")
    new_layers = []
    old_to_new = {}
    inserted_blocks = []

    for oi, layer in enumerate(old_layers):
        old_to_new[oi] = len(new_layers)
        new_layers.append(layer)

        name = layer.__class__.__name__
        if name in {"C3k2", "C2f"} and oi < backbone_end:
            ch = _get_output_channels(layer)
            if ch:
                ca = CoordAtt(ch, ch)
                ca.f = -1
                ca.i = -1
                new_layers.append(ca)

                eca = ECA(ch, k_size=3)
                eca.f = -1
                eca.i = -1
                new_layers.append(eca)
                inserted_blocks.append((oi, name, ch))

    if not inserted_blocks:
        block_names = sorted({layer.__class__.__name__ for layer in old_layers[:backbone_end]})
        raise RuntimeError(
            "CoordAtt+ECA did not find a compatible YOLO11 C3k2 or YOLOv8 C2f backbone block; "
            f"found={block_names}"
        )

    print(f"注意力插入点: {inserted_blocks}")
    print(f"层数: {len(old_layers)} → {len(new_layers)}")

    # 修复 f / i 索引
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "f"):
            f = layer.f
            if isinstance(f, (int, np.integer)):
                if f != -1:
                    layer.f = old_to_new.get(int(f), int(f))
            else:
                new_f = []
                for idx in f:
                    if idx == -1:
                        new_f.append(-1)
                    else:
                        new_f.append(old_to_new.get(int(idx), int(idx)))
                layer.f = new_f
        layer.i = i

    detection_model.model = nn.Sequential(*new_layers)

    # 重建 save 列表
    save = set()
    for layer in new_layers:
        f = getattr(layer, "f", -1)
        if f == -1:
            continue
        indices = [f] if isinstance(f, (int, np.integer)) else f
        for idx in indices:
            if idx != -1:
                save.add(int(idx))
    detection_model.save = sorted(save)

    return detection_model


# ============================================================================
# 5 通道适配：修改第一层 Conv
# ============================================================================
def adapt_first_conv_5ch(model, spectral_init="mean"):
    """将首层卷积改为5通道，并初始化NDVI/NDRE卷积核。"""
    if spectral_init not in {"mean", "zero", "kaiming"}:
        raise ValueError(f"Unsupported spectral initialization: {spectral_init}")
    first = model.model[0]
    old_conv = first.conv
    old_weight = old_conv.weight.detach()

    new_conv = nn.Conv2d(
        5,
        old_conv.out_channels,
        old_conv.kernel_size,
        old_conv.stride,
        old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_weight)
        if spectral_init == "mean":
            mean_kernel = old_weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:].copy_(mean_kernel.expand(-1, 2, -1, -1))
        elif spectral_init == "zero":
            new_conv.weight[:, 3:].zero_()
        else:
            nn.init.kaiming_uniform_(new_conv.weight[:, 3:], a=1.0)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    first.conv = new_conv
    print(f"第一层 Conv: {old_conv.in_channels}→{new_conv.in_channels} 通道, "
          f"ch0-2: 预训练, ch3-4: {spectral_init} 初始化")
    return model


def print_architecture(model):
    print("\n===== 模型架构 =====")
    for i, layer in enumerate(model.model):
        name = layer.__class__.__name__
        f = getattr(layer, "f", None)
        ch = _get_output_channels(layer)
        params = sum(p.numel() for p in layer.parameters())
        print(f"  {i:3d}: {name:15s} ch={str(ch):>6s}  "
              f"params={params:>8,d}  f={f}")


def verify_forward(model, imgsz=640):
    print("\n===== 验证前向传播 (5ch) =====")
    model.model.eval()
    x = torch.randn(1, 5, imgsz, imgsz)
    with torch.no_grad():
        y = model(x)
    if isinstance(y, (list, tuple)):
        for i, yi in enumerate(y):
            if isinstance(yi, torch.Tensor):
                print(f"  输出[{i}]: shape={yi.shape}")
    print("前向传播 OK!")


# ============================================================================
# 训练入口
# ============================================================================
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train the reviewed five-channel segmentation dataset.")
    parser.add_argument("--name", default=None, help="Experiment name; generated from profile/model/seed by default.")
    parser.add_argument("--architecture", choices=("yolo11", "yolov8"), default="yolo11")
    parser.add_argument("--model-size", choices=("n", "s"), default="n")
    parser.add_argument(
        "--base-model",
        choices=SUPPORTED_SEGMENTATION_MODELS,
        default=None,
        help="Explicit base model; overrides --architecture and --model-size.",
    )
    parser.add_argument("--profile", choices=tuple(AUGMENTATION_PROFILES), default=None)
    parser.add_argument("--attention", choices=("ca_eca", "none"), default="ca_eca")
    parser.add_argument("--spectral-init", choices=("mean", "zero", "kaiming"), default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reproducible PyTorch algorithms (default: enabled).",
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training images to use.")
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache decoded five-channel images in RAM (default: enabled).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one full-data epoch without RAM caching or data-loader workers.",
    )
    parser.add_argument("--verify-only", action="store_true", help="Build the model and run a 5ch forward pass only.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from last.pt while preserving the serialized custom architecture.",
    )
    parser.add_argument(
        "--project",
        default=str(PROJECT_ROOT / "runs" / "segment" / "land_seg"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    from ultralytics.nn.tasks import SegmentationModel, load_checkpoint

    cli = parse_cli_args()
    if not 0.0 < cli.fraction <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {cli.fraction}")
    profile_name = cli.profile or ("baseline" if cli.resume else "spectral_aug")
    profile = AUGMENTATION_PROFILES[profile_name]
    spectral_init = cli.spectral_init or ("mean" if profile_name == "spectral_aug" else "kaiming")
    base_checkpoint = resolve_base_checkpoint(cli.base_model, cli.architecture, cli.model_size)
    validate_attention_compatibility(base_checkpoint, cli.attention)
    compact_model = any(token in base_checkpoint for token in ("11n", "v8n", "12n", "26n"))
    batch = cli.batch if cli.batch is not None else (2 if compact_model else 1)
    model_tag = Path(base_checkpoint).stem
    default_name = (
        f"exp_{cli.attention}_5ch_{model_tag}_{profile_name}_seed{cli.seed}_v2"
    )
    experiment_name = cli.name or (f"smoke_{default_name}" if cli.smoke_test else default_name)
    epochs = 1 if cli.smoke_test else cli.epochs
    fraction = cli.fraction
    cache = False if cli.smoke_test else cli.cache
    workers = 0 if cli.smoke_test else cli.workers
    args = {
        "model": base_checkpoint,
        "data": str(PROJECT_ROOT / "datasets" / "Data_v3" / "data_5ch_v3.yaml"),
        "epochs": epochs,
        "imgsz": cli.imgsz,
        "batch": batch,
        "device": cli.device,
        "workers": workers,
        "patience": 50,
        "optimizer": "SGD",
        "lr0": 0.002,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 10,
        "cos_lr": True,
        "amp": True,
        "cache": cache,
        "fraction": fraction,
        "save": True,
        "save_period": 10,
        "project": cli.project,
        "name": experiment_name,
        "exist_ok": False,
        "seed": cli.seed,
        "deterministic": cli.deterministic,
        "verbose": True,
    }
    args.update(profile["training"])

    print("\n===== 实验配置 =====")
    print(f"模型: {base_checkpoint}  attention={cli.attention}")
    print(f"增强: {profile_name}  5ch_photometric={profile['photometric']}")
    print(f"光谱卷积初始化: {spectral_init}")
    print(
        f"实验名: {experiment_name}  seed={cli.seed}  batch={batch}  "
        f"epochs={epochs}  fraction={fraction:g}  cache={cache}"
    )

    if cli.resume:
        args["resume"] = cli.resume
        trainer = FiveChannelSegmentationTrainer(
            overrides=args,
            five_channel_photometric=profile["photometric"],
            five_channel_photometric_kwargs=profile["photometric_kwargs"],
        )
        trainer.train()
        raise SystemExit(0)

    # 加载预训练权重，用 nc=1 重建模型
    ckpt_model, _ = load_checkpoint(base_checkpoint)
    base_model = SegmentationModel(ckpt_model.yaml, nc=1)
    base_model.load(ckpt_model)

    # 插入 CoordAtt + ECA
    model = build_ca_model_from_base(base_model) if cli.attention == "ca_eca" else base_model

    # 5 通道适配
    model = adapt_first_conv_5ch(model, spectral_init=spectral_init)

    print_architecture(model)
    verify_forward(model, imgsz=min(cli.imgsz, 640))
    if cli.verify_only:
        raise SystemExit(0)

    # 注入修改后的模型
    trainer = FiveChannelSegmentationTrainer(
        overrides=args,
        five_channel_photometric=profile["photometric"],
        five_channel_photometric_kwargs=profile["photometric_kwargs"],
    )
    trainer.model = model
    trainer.train()
