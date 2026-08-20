"""Lightweight multispectral fusion modules for the Data_v8 screen.

The Ultralytics graph is kept intact: the first layer splits RGB from the
spectral channels, and wrappers at selected backbone stages consume spectral
features from a per-forward context.  The context is empty again after the
last fusion stage so checkpoints never retain batch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class FusionRuntimeContext:
    """Ephemeral features shared by the input stem and fusion wrappers."""

    features: dict[str, torch.Tensor] = field(default_factory=dict)

    def reset(self, features: dict[str, torch.Tensor] | None = None) -> None:
        self.features.clear()
        if features:
            self.features.update(features)

    def pop(self, level: str) -> torch.Tensor:
        if level not in self.features:
            raise RuntimeError(f"Missing spectral feature {level}; available={sorted(self.features)}")
        return self.features.pop(level)


class ConvBNAct(nn.Sequential):
    def __init__(self, c1: int, c2: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(c1, c2, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparable(nn.Module):
    def __init__(self, c1: int, c2: int, *, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, 3, stride, 1, groups=c1, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(c1)
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.depthwise_bn(self.depthwise(x)))
        return self.act(self.pointwise_bn(self.pointwise(x)))


class LightweightSpectralEncoder(nn.Module):
    """Encode NIR/RedEdge (and optional indices) at P3/P4/P5 resolutions."""

    def __init__(self, in_channels: int, requested_levels: Iterable[str]) -> None:
        super().__init__()
        requested = tuple(dict.fromkeys(requested_levels))
        unknown = sorted(set(requested) - {"P3", "P4", "P5"})
        if unknown:
            raise ValueError(f"Unknown spectral levels: {unknown}")
        if not requested:
            raise ValueError("At least one spectral level is required")
        self.requested_levels = requested
        self.stage1 = ConvBNAct(in_channels, 8, stride=2)       # /2
        self.stage2 = DepthwiseSeparable(8, 8, stride=2)       # /4
        self.stage3 = DepthwiseSeparable(8, 16, stride=2)      # /8  P3
        self.stage4 = DepthwiseSeparable(16, 32, stride=2)     # /16 P4
        self.stage5 = (
            DepthwiseSeparable(32, 64, stride=2) if "P5" in requested else None
        )                                                       # /32 P5

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        all_features = {"P3": p3, "P4": p4}
        if self.stage5 is not None:
            all_features["P5"] = self.stage5(p4)
        return {level: all_features[level] for level in self.requested_levels}


class RGBSpectralInputStem(nn.Module):
    """Run the pretrained RGB stem and prepare spectral pyramid features."""

    def __init__(
        self,
        rgb_stem: nn.Module,
        context: FusionRuntimeContext,
        *,
        input_channels: int,
        spectral_indices: tuple[int, ...],
        requested_levels: tuple[str, ...],
    ) -> None:
        super().__init__()
        if input_channels < 5:
            raise ValueError(f"Dual-branch fusion requires at least 5 channels, got {input_channels}")
        if not spectral_indices:
            raise ValueError("spectral_indices must not be empty")
        if max(spectral_indices) >= input_channels or min(spectral_indices) < 3:
            raise ValueError(
                f"Invalid spectral indices {spectral_indices} for {input_channels}-channel RGB-first input"
            )
        self.rgb_stem = rgb_stem
        self.spectral_encoder = LightweightSpectralEncoder(len(spectral_indices), requested_levels)
        self.context = context
        self.input_channels = int(input_channels)
        self.spectral_indices = tuple(int(i) for i in spectral_indices)

    @property
    def conv(self):
        """Expose the pretrained conv for existing architecture diagnostics."""
        return self.rgb_stem.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != self.input_channels:
            raise RuntimeError(
                f"Fusion stem expected BCHW with {self.input_channels} channels, got {tuple(x.shape)}"
            )
        self.context.reset()
        rgb = x[:, :3]
        spectral = x[:, self.spectral_indices]
        self.context.reset(self.spectral_encoder(spectral))
        return self.rgb_stem(rgb)


class SimpleConcatFusion(nn.Module):
    """Minimal concat + 1x1 fusion initialized near an RGB identity."""

    def __init__(self, rgb_channels: int, spectral_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(rgb_channels + spectral_channels, rgb_channels, 1, bias=True)
        with torch.no_grad():
            self.fuse.weight.zero_()
            eye = torch.eye(rgb_channels).view(rgb_channels, rgb_channels, 1, 1)
            self.fuse.weight[:, :rgb_channels].copy_(eye)
            nn.init.normal_(self.fuse.weight[:, rgb_channels:], mean=0.0, std=0.01)
            self.fuse.bias.zero_()

    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat((rgb, spectral), dim=1))


class GatedResidualFusion(nn.Module):
    """Inject projected spectral features through a spatial/channel gate."""

    def __init__(self, rgb_channels: int, spectral_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(spectral_channels, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Conv2d(rgb_channels * 2, rgb_channels, 1, bias=True)
        nn.init.kaiming_uniform_(self.project[0].weight, a=1.0)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, rgb: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        projected = self.project(spectral)
        gate = torch.sigmoid(self.gate(torch.cat((rgb, projected), dim=1)))
        return rgb + gate * projected


class BackboneFusionWrapper(nn.Module):
    """Wrap one YOLO backbone layer and fuse its output with one spectral level."""

    def __init__(
        self,
        block: nn.Module,
        context: FusionRuntimeContext,
        *,
        level: str,
        rgb_channels: int,
        spectral_channels: int,
        method: str,
    ) -> None:
        super().__init__()
        self.block = block
        self.context = context
        self.level = level
        if method == "concat":
            self.fusion = SimpleConcatFusion(rgb_channels, spectral_channels)
        elif method == "gated":
            self.fusion = GatedResidualFusion(rgb_channels, spectral_channels)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")
        self.method = method

        for name in ("i", "f", "type", "np"):
            if hasattr(block, name):
                setattr(self, name, getattr(block, name))

    def forward(self, x):
        rgb = self.block(x)
        spectral = self.context.pop(self.level)
        if rgb.shape[-2:] != spectral.shape[-2:]:
            raise RuntimeError(
                f"{self.level} spatial mismatch: rgb={tuple(rgb.shape)}, spectral={tuple(spectral.shape)}"
            )
        return self.fusion(rgb, spectral)


_YOLOV8_LEVELS = {
    "P3": (4, 64, 16),
    "P4": (6, 128, 32),
    "P5": (9, 256, 64),
}


def build_yolov8_multispectral_fusion(model, *, mode: str, input_channels: int):
    """Patch a YOLOv8n segmentation model with F1/F2/F3 fusion."""

    if mode not in {"f1_p4_concat", "f2_multiscale_gated", "f3_index_guided_gated"}:
        raise ValueError(f"Unsupported Data_v8 fusion mode: {mode}")
    layers = list(model.model)
    if len(layers) != 23 or layers[2].__class__.__name__ != "C2f":
        raise RuntimeError("Data_v8 screening fusion currently supports the 23-layer YOLOv8n-seg graph only")

    if mode == "f1_p4_concat":
        levels = ("P4",)
        method = "concat"
        spectral_indices = (3, 4)
    elif mode == "f2_multiscale_gated":
        levels = ("P3", "P4", "P5")
        method = "gated"
        spectral_indices = (3, 4)
    else:
        levels = ("P3", "P4", "P5")
        method = "gated"
        spectral_indices = (3, 4, 5, 6)
        if input_channels != 7:
            raise ValueError(f"F3 requires 7 channels, got {input_channels}")

    context = FusionRuntimeContext()
    original_stem = layers[0]
    stem = RGBSpectralInputStem(
        original_stem,
        context,
        input_channels=input_channels,
        spectral_indices=spectral_indices,
        requested_levels=levels,
    )
    for name in ("i", "f", "type", "np"):
        if hasattr(original_stem, name):
            setattr(stem, name, getattr(original_stem, name))
    layers[0] = stem

    for level in levels:
        index, rgb_channels, spectral_channels = _YOLOV8_LEVELS[level]
        layers[index] = BackboneFusionWrapper(
            layers[index],
            context,
            level=level,
            rgb_channels=rgb_channels,
            spectral_channels=spectral_channels,
            method=method,
        )
    model.model = nn.Sequential(*layers)
    model.fusion_mode = mode
    model.input_channels = int(input_channels)
    return model


def fusion_parameter_count(model: nn.Module) -> int:
    names = ("spectral_encoder", "fusion")
    parameters = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        if any(token in module_name for token in names):
            for parameter in module.parameters(recurse=False):
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter.numel())
    return int(sum(parameters))


def assert_fusion_context_empty(model: nn.Module) -> None:
    contexts: list[FusionRuntimeContext] = []
    for module in model.modules():
        context = getattr(module, "context", None)
        if isinstance(context, FusionRuntimeContext) and all(context is not item for item in contexts):
            contexts.append(context)
    leftovers = {i: sorted(context.features) for i, context in enumerate(contexts) if context.features}
    if leftovers:
        raise RuntimeError(f"Fusion runtime context retained batch tensors: {leftovers}")
