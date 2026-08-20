"""Build one uniform V15 three-seed YOLO11 model soup without touching deployment."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch
from ultralytics.nn.tasks import load_checkpoint

from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, sha256


ROOT = Path(__file__).resolve().parent
SOURCES = tuple(
    ROOT / "runs" / "segment" / "land_seg" / f"v15_yolo11_c0_nooverlap_standardized_seed{seed}_fresh_v1" / "weights" / "best.pt"
    for seed in (0, 1, 2)
)
OUTPUT = ROOT / "runs" / "segment" / "land_seg" / "v16_yolo11_v15_seed012_uniform_soup_v1" / "weights" / "best.pt"
MANIFEST = OUTPUT.parent.parent / "soup_manifest.json"
V15_SUMMARY = ROOT / "logs" / "v15_yolo11_standardized_multiseed.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs=3, type=Path, default=list(SOURCES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser.parse_args()


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not states:
        raise ValueError("At least one state dict is required")
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("Source model state dict keys do not match")
    averaged: dict[str, torch.Tensor] = {}
    nonfloating_mismatches: list[str] = []
    for key in keys:
        values = [state[key].detach().cpu() for state in states]
        if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values[1:]):
            raise ValueError(f"Source tensor contract differs for {key}")
        if values[0].is_floating_point():
            mean = torch.stack([value.float() for value in values], dim=0).mean(dim=0)
            averaged[key] = mean.to(dtype=values[0].dtype)
        else:
            if any(not torch.equal(value, values[0]) for value in values[1:]):
                nonfloating_mismatches.append(key)
            averaged[key] = values[0].clone()
    return averaged, nonfloating_mismatches


def first_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except ValueError:
                pass
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return first_tensor(item)
            except ValueError:
                pass
    raise ValueError("Forward output contains no tensor")


def main() -> int:
    args = parse_args()
    sources = [path.resolve() for path in args.sources]
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    v15 = json.loads(V15_SUMMARY.read_text(encoding="utf-8"))
    if v15.get("gate_passed") is not True or v15.get("held_out_test_used") is not False:
        raise RuntimeError("V15 did not authorize a valid-only post-gate candidate review")
    if v15.get("validation_overlap_mask") is not False:
        raise RuntimeError("V15 standardized validation contract changed")
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V16 soup construction")

    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in sources]
    models = [checkpoint.get("model") for checkpoint in checkpoints]
    if any(model is None for model in models):
        raise RuntimeError("A V15 source checkpoint has no serialized model")
    states = [model.float().state_dict() for model in models]
    averaged, nonfloating_mismatches = average_state_dicts(states)
    soup_model = copy.deepcopy(models[2]).float()
    soup_model.load_state_dict(averaged, strict=True)
    soup_model.eval()
    with torch.inference_mode():
        output_shape = first_tensor(soup_model(torch.zeros(1, 5, 320, 320))).shape
    checkpoint = copy.deepcopy(checkpoints[2])
    checkpoint["model"] = soup_model.half()
    checkpoint["ema"] = None
    checkpoint["optimizer"] = None
    checkpoint["scaler"] = None
    checkpoint["epoch"] = -1
    checkpoint["best_fitness"] = None
    checkpoint["date"] = datetime.now().astimezone().isoformat(timespec="seconds")
    train_args = dict(checkpoint.get("train_args") or {})
    train_args.update(
        name="v16_yolo11_v15_seed012_uniform_soup_v1",
        model=str(output),
        mode="val",
        overlap_mask=False,
    )
    checkpoint["train_args"] = train_args
    checkpoint["soup_contract"] = {
        "version": "v16_uniform_model_soup_v1",
        "sources": [str(path) for path in sources],
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_write_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    loaded, _ = load_checkpoint(str(output), device="cpu")
    actual_channels = int(getattr(loaded, "input_channels", loaded.model[0].conv.in_channels))
    if actual_channels != 5:
        raise RuntimeError(f"V16 soup expects 5 channels, got {actual_channels}")
    if sha256(DEPLOYMENT) != deployment_hash:
        raise RuntimeError("Deployment checkpoint changed during V16 soup construction")
    manifest = {
        "version": "v16_uniform_model_soup_manifest_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": [
            {"checkpoint": str(path), "sha256": sha256(path), "weight": 1 / 3}
            for path in sources
        ],
        "output": str(output),
        "output_sha256": sha256(output),
        "state_tensors": len(averaged),
        "nonfloating_mismatches": nonfloating_mismatches,
        "forward_output_shape": list(output_shape),
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "attention": "3x CoordAtt + 3x ECA",
        "held_out_test_used": False,
        "deployment_write_allowed": False,
        "deployment_sha256_before_after": deployment_hash,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
