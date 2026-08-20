"""Freeze an untouched 30-image holdout from the unlabeled raw pool.

Selection uses only filename continuity.  It never reads image pixels for
selection and never creates labels or model predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

from multispectral import PROJECT_ROOT, build_multichannel, save_multichannel_uint8


RAW_ROOT = PROJECT_ROOT / "多光谱图片"
USED_DATASET = PROJECT_ROOT / "datasets" / "Data_v3"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "Data_release_blind_v1"
CHANNEL_CONFIG = PROJECT_ROOT / "configs" / "data_v4_channels.json"
TARGET_COUNT = 30


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def image_number(name: str) -> int:
    match = re.fullmatch(r"DJI_(\d+)\.JPG", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected raw image name: {name}")
    return int(match.group(1))


def contiguous_blocks(numbers: list[int], step: int = 10) -> list[list[int]]:
    blocks: list[list[int]] = []
    for number in sorted(numbers):
        if not blocks or number - blocks[-1][-1] != step:
            blocks.append([])
        blocks[-1].append(number)
    return blocks


def select_largest_complete_blocks(blocks: list[list[int]], target_count: int) -> list[list[int]]:
    selected: list[list[int]] = []
    total = 0
    for block in sorted(blocks, key=lambda values: (-len(values), values[0])):
        if total + len(block) <= target_count:
            selected.append(list(block))
            total += len(block)
        if total == target_count:
            break
    if total != target_count:
        raise RuntimeError(f"Could not select exactly {target_count} complete blocks; got {total}")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen holdout: {OUTPUT_ROOT}")
    raw_images = sorted(
        (path for path in RAW_ROOT.glob("*.JPG") if re.fullmatch(r"DJI_\d+", path.stem, flags=re.IGNORECASE)),
        key=lambda path: image_number(path.name),
    )
    used_stems = {
        path.stem
        for path in USED_DATASET.rglob("*.JPG")
        if re.fullmatch(r"DJI_\d+", path.stem, flags=re.IGNORECASE)
    }
    unused = [path for path in raw_images if path.stem not in used_stems]
    if len(raw_images) != 429 or len(used_stems) != 300 or len(unused) != 129:
        raise RuntimeError(f"Raw/used/unused contract drifted: {len(raw_images)}/{len(used_stems)}/{len(unused)}")
    selected_blocks = select_largest_complete_blocks(
        contiguous_blocks([image_number(path.name) for path in unused]), TARGET_COUNT
    )
    selected_numbers = [number for block in selected_blocks for number in block]
    selected = [RAW_ROOT / f"DJI_{number:04d}.JPG" for number in selected_numbers]
    manual_dirs = (RAW_ROOT / "labels", PROJECT_ROOT / "待标注_100张" / "labels")
    pseudo_dir = RAW_ROOT / "pseudo_labels"
    for path in selected:
        if any((directory / f"{path.stem}.txt").exists() for directory in manual_dirs) or (pseudo_dir / f"{path.stem}.txt").exists():
            raise RuntimeError(f"Selected holdout is not label-blind: {path.stem}")

    config = json.loads(CHANNEL_CONFIG.read_text(encoding="utf-8"))
    channels = tuple(config["channels"])
    if len(channels) != 17 or config.get("storage", {}).get("dtype") != "uint8":
        raise RuntimeError("Blind feature channel contract drifted")
    records: list[dict[str, object]] = []
    for index, image_path in enumerate(selected, 1):
        output_image = OUTPUT_ROOT / "test" / "images" / image_path.name
        output_feature = OUTPUT_ROOT / "test" / "features" / f"{image_path.stem}.npy"
        hardlink(image_path, output_image)
        feature = build_multichannel(image_path, channels, RAW_ROOT)
        save_multichannel_uint8(output_feature, feature, channels)
        records.append(
            {
                "index": index,
                "stem": image_path.stem,
                "image_sha256": sha256(output_image),
                "feature_sha256": sha256(output_feature),
                "feature_shape": list(feature.shape),
            }
        )
        print(f"[{index}/{len(selected)}] {image_path.stem}")

    (OUTPUT_ROOT / "test" / "labels").mkdir(parents=True, exist_ok=True)
    stem_signature = hashlib.sha256("\n".join(path.stem for path in selected).encode("utf-8")).hexdigest()
    manifest = {
        "version": "release_blind_holdout_v1",
        "created_at": now_iso(),
        "state": "frozen_pending_manual_annotation",
        "selection_rule": "largest complete contiguous unused filename blocks; descending block length then ascending first number",
        "selection_uses_image_pixels": False,
        "selection_uses_labels": False,
        "selection_uses_model_predictions": False,
        "source_raw_images": 429,
        "previously_used_stems": 300,
        "unused_pool_stems": 129,
        "selected_blocks": selected_blocks,
        "selected_images": TARGET_COUNT,
        "selected_stem_signature": stem_signature,
        "channels": list(channels),
        "feature_storage": "uint8",
        "records": records,
        "manual_labels_present": 0,
        "pseudo_labels_present": 0,
        "held_out_from_future_training_and_validation": True,
        "old_test_results_allowed_for_candidate_tuning": False,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state": manifest["state"], "selected_images": TARGET_COUNT, "manifest": str((OUTPUT_ROOT / 'manifest.json').resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
