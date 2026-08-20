"""Build and validate a train/valid-only contract for Data_v8 fusion.

This script deliberately names only the train and valid directories.  It does
not enumerate, hash, load, or copy any held-out test image or feature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = ROOT / "datasets" / "Data_v4_candidates"
OUTPUT_ROOT = ROOT / "datasets" / "Data_v8_multispectral_fusion"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest.json"
CHANNEL_CONFIG = ROOT / "configs" / "data_v4_channels.json"
TRAIN_IMAGES = ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed" / "train" / "images"
TRAIN_LABELS = ROOT / "datasets" / "Data_v6_teacher_refined_yolo11_final_reviewed" / "train" / "labels"
VALID_IMAGES = ROOT / "datasets" / "Data_v3" / "valid" / "images"
VALID_LABELS = ROOT / "datasets" / "Data_v3" / "valid" / "labels"
ALLOWED_SPLITS = ("train", "valid")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def image_stems(path: Path) -> set[str]:
    return {item.stem for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES}


def validate_split(split: str, images: Path, labels: Path, channel_count: int) -> dict:
    if split not in ALLOWED_SPLITS:
        raise RuntimeError(f"Refusing to inspect non-selection split: {split}")
    feature_dir = FEATURE_ROOT / split / "features"
    for required in (images, labels, feature_dir):
        if not required.is_dir():
            raise FileNotFoundError(required)

    stems = image_stems(images)
    label_paths = sorted(path for path in labels.glob("*.txt") if path.stem in stems)
    feature_paths = sorted(path for path in feature_dir.glob("*.npy") if path.stem in stems)
    label_stems = {path.stem for path in label_paths}
    feature_stems = {path.stem for path in feature_paths}
    if stems != label_stems:
        raise RuntimeError(
            f"{split} image/label mismatch: missing_labels={sorted(stems-label_stems)[:5]} "
            f"extra_labels={sorted(label_stems-stems)[:5]}"
        )
    if stems != feature_stems:
        raise RuntimeError(
            f"{split} image/feature mismatch: missing_features={sorted(stems-feature_stems)[:5]} "
            f"extra_features={sorted(feature_stems-stems)[:5]}"
        )

    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    low_clip = 0
    high_clip = 0
    values = 0
    for path in feature_paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or int(array.shape[-1]) != channel_count:
            raise RuntimeError(f"Unexpected feature shape {array.shape}: {path}")
        if array.dtype != np.uint8:
            raise RuntimeError(f"Expected uint8 feature storage, got {array.dtype}: {path}")
        shapes.add(tuple(int(v) for v in array.shape))
        dtypes.add(str(array.dtype))
        # Deterministic sparse sampling avoids loading every full-resolution array.
        sample = np.asarray(array[::64, ::64, :])
        low_clip += int((sample == 0).sum())
        high_clip += int((sample == 255).sum())
        values += int(sample.size)

    return {
        "images": len(stems),
        "labels": len(label_paths),
        "features": len(feature_paths),
        "stem_signature": hashlib.sha256("\n".join(sorted(stems)).encode("utf-8")).hexdigest(),
        "labels_sha256": aggregate_hash(label_paths),
        "feature_shapes": [list(shape) for shape in sorted(shapes)],
        "feature_dtypes": sorted(dtypes),
        "sampled_zero_clip_rate": low_clip / max(values, 1),
        "sampled_255_clip_rate": high_clip / max(values, 1),
    }


def main() -> None:
    channel_config = json.loads(CHANNEL_CONFIG.read_text(encoding="utf-8"))
    channels = list(channel_config["channels"])
    split_contracts = {
        "train": validate_split("train", TRAIN_IMAGES, TRAIN_LABELS, len(channels)),
        "valid": validate_split("valid", VALID_IMAGES, VALID_LABELS, len(channels)),
    }
    payload = {
        "version": "data_v8_multispectral_fusion_contract_v1",
        "created_from": "existing Data_v4 train/valid feature arrays; no feature copies",
        "sensor": channel_config["sensor"],
        "storage": channel_config["storage"],
        "source_bands": channel_config["source_bands"],
        "channels": channels,
        "selection_splits": list(ALLOWED_SPLITS),
        "held_out_split": "test",
        "held_out_split_used": False,
        "source_feature_root": str(FEATURE_ROOT.resolve()),
        "split_contracts": split_contracts,
        "channel_config_sha256": sha256_file(CHANNEL_CONFIG),
        "notes": [
            "Only train and valid were enumerated and checked.",
            "The frozen test split was not enumerated, loaded, hashed, copied, or materialized.",
            "RGB storage order is R,G,B; selected channel order is defined per fusion group.",
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
