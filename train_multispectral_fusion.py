"""Train one Data_v8 multispectral fusion group on train/valid only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from custom_fusion_modules import (
    assert_fusion_context_empty,
    build_yolov8_multispectral_fusion,
    fusion_parameter_count,
)
from multispectral import PROJECT_ROOT
from train_feature_ablation import ABLATION_AUGMENTATION, adapt_first_conv_channels
from yolo_multichannel_data import MultichannelSegmentationTrainer, validate_channel_selection


MANIFEST = PROJECT_ROOT / "datasets" / "Data_v8_multispectral_fusion" / "manifest.json"
GROUPS_CONFIG = PROJECT_ROOT / "configs" / "data_v8_fusion_groups.json"
DATA_BY_CHANNELS = {
    5: PROJECT_ROOT / "configs" / "data_v8_fusion_selection_5ch.yaml",
    7: PROJECT_ROOT / "configs" / "data_v8_fusion_selection_7ch.yaml",
}
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "segment" / "land_seg"
DEPLOYMENT = DEFAULT_PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    config = read_json(GROUPS_CONFIG) if GROUPS_CONFIG.is_file() else {"groups": {}}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=sorted(config.get("groups", {})))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--groups-config", type=Path, default=GROUPS_CONFIG)
    parser.add_argument("--base-model", default="yolov8n-seg.pt")
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def validate_contract(cli: argparse.Namespace) -> tuple[dict, dict, tuple[str, ...], Path, Path]:
    required = (cli.manifest, cli.groups_config, DEPLOYMENT)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError(
            f"Deployment SHA256 mismatch: expected {DEPLOYMENT_SHA256}, got {deployment_hash}"
        )
    manifest = read_json(cli.manifest)
    if manifest.get("selection_splits") != ["train", "valid"]:
        raise RuntimeError("Data_v8 manifest must expose exactly train/valid for selection")
    if manifest.get("held_out_split_used") is not False:
        raise RuntimeError("Data_v8 manifest must declare held_out_split_used=false")
    if "test" in manifest.get("split_contracts", {}):
        raise RuntimeError("Data_v8 manifest must not contain a test split contract")
    groups = read_json(cli.groups_config)
    entry = groups.get("groups", {}).get(cli.group)
    if not entry:
        raise ValueError(f"Unknown fusion group: {cli.group}")
    channels = tuple(entry["channels"])
    validate_channel_selection(tuple(manifest["channels"]), channels)
    data = DATA_BY_CHANNELS.get(len(channels))
    if data is None or not data.is_file():
        raise FileNotFoundError(f"No train/valid YAML for {len(channels)} channels")
    data_text = data.read_text(encoding="utf-8").lower()
    if "test:" in data_text or f"channels: {len(channels)}" not in data_text:
        raise RuntimeError(f"Unsafe or inconsistent selection YAML: {data}")
    feature_root = Path(manifest["source_feature_root"])
    if not feature_root.is_dir():
        raise FileNotFoundError(feature_root)
    return manifest, entry, channels, data, feature_root


def build_model(base_model: str, fusion_mode: str, channels: tuple[str, ...], seed: int):
    from ultralytics.nn.tasks import SegmentationModel, load_checkpoint

    if Path(base_model).name != "yolov8n-seg.pt":
        raise ValueError("Seed-0 fusion screening is locked to yolov8n-seg.pt")
    seed_everything(seed)
    checkpoint_model, _ = load_checkpoint(base_model)
    model = SegmentationModel(checkpoint_model.yaml, nc=1)
    model.load(checkpoint_model)
    if fusion_mode == "none":
        model = adapt_first_conv_channels(model, len(channels), "mean")
        model.fusion_mode = "none"
        model.input_channels = len(channels)
    else:
        model = build_yolov8_multispectral_fusion(
            model,
            mode=fusion_mode,
            input_channels=len(channels),
        )
    return model


def verify_model(model, channels: tuple[str, ...], size: int) -> None:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, len(channels), size, size, device=device))
    if output is None:
        raise RuntimeError("Fusion model returned None")
    assert_fusion_context_empty(model)
    model.train(was_training)
    print(f"[fusion] forward verified: input=(1,{len(channels)},{size},{size})")


def write_contract(
    save_dir: Path,
    *,
    cli: argparse.Namespace,
    manifest: dict,
    entry: dict,
    channels: tuple[str, ...],
    model,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "data_v8_multispectral_fusion_run_contract_v1",
        "group": cli.group,
        "role": entry["role"],
        "fusion": entry["fusion"],
        "channels": list(channels),
        "channel_count": len(channels),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "manifest_version": manifest["version"],
        "manifest_path": str(cli.manifest.resolve()),
        "base_model": cli.base_model,
        "attention": "none",
        "seed": cli.seed,
        "epochs": 1 if cli.smoke_test else cli.epochs,
        "imgsz": cli.imgsz,
        "batch": 1 if cli.smoke_test else cli.batch,
        "fraction": 0.05 if cli.smoke_test else cli.fraction,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fusion_parameters": fusion_parameter_count(model),
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_before": sha256(DEPLOYMENT),
        "deployment_write_allowed": False,
    }
    (save_dir / "fusion_contract.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    cli = parse_args()
    if cli.epochs < 1 or cli.patience < 1 or cli.batch < 1 or not 0.0 < cli.fraction <= 1.0:
        raise ValueError("epochs, patience, batch and fraction must be positive and fraction <= 1")
    if cli.resume and (cli.verify_only or cli.smoke_test):
        raise ValueError("--resume cannot be combined with --verify-only or --smoke-test")
    manifest, entry, channels, data, feature_root = validate_contract(cli)
    name = cli.name or f"v8_{cli.group}_yolov8n_none_1024_seed{cli.seed}_screen_v1"
    if cli.smoke_test and cli.name is None:
        name = f"smoke_{cli.group}_seed{cli.seed}_v1"

    if cli.resume:
        from ultralytics.nn.tasks import load_checkpoint

        model, _ = load_checkpoint(str(cli.resume.resolve()))
        if getattr(model, "fusion_mode", "none") != entry["fusion"]:
            raise RuntimeError("Resume checkpoint fusion mode does not match requested group")
    else:
        model = build_model(cli.base_model, entry["fusion"], channels, cli.seed)
    verify_model(model, channels, min(cli.imgsz, 320))
    if cli.verify_only:
        print(
            json.dumps(
                {
                    "group": cli.group,
                    "fusion": entry["fusion"],
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "fusion_parameters": fusion_parameter_count(model),
                },
                ensure_ascii=False,
            )
        )
        return

    smoke = cli.smoke_test
    overrides = {
        "model": cli.base_model,
        "data": str(data.resolve()),
        "epochs": 1 if smoke else cli.epochs,
        "imgsz": cli.imgsz,
        "batch": 1 if smoke else cli.batch,
        "device": cli.device,
        "workers": 0 if smoke else cli.workers,
        "patience": cli.patience,
        "optimizer": "SGD",
        "lr0": 0.002,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "cos_lr": True,
        "amp": True,
        "cache": False if smoke else True,
        "fraction": 0.05 if smoke else cli.fraction,
        "save": True,
        "save_period": 10,
        "project": str(cli.project.resolve()),
        "name": name,
        "exist_ok": False,
        "seed": cli.seed,
        "deterministic": True,
        "verbose": True,
    }
    overrides.update(ABLATION_AUGMENTATION)
    if cli.resume:
        overrides["resume"] = str(cli.resume.resolve())

    trainer = MultichannelSegmentationTrainer(
        overrides=overrides,
        feature_root=feature_root,
        all_channels=tuple(manifest["channels"]),
        selected_channels=channels,
        photometric_augment=False,
    )
    trainer.model = model
    write_contract(
        Path(trainer.save_dir),
        cli=cli,
        manifest=manifest,
        entry=entry,
        channels=channels,
        model=model,
    )
    print(
        f"[fusion] group={cli.group} channels={channels} mode={entry['fusion']} "
        f"train/valid only; deployment writes forbidden"
    )
    trainer.train()
    after = sha256(DEPLOYMENT)
    if after != DEPLOYMENT_SHA256:
        raise RuntimeError(f"Deployment checkpoint changed during fusion run: {after}")


if __name__ == "__main__":
    main()
