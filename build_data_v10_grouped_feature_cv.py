"""Build grouped-CV YAMLs for soil/vegetation feature screening and verify paired inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from multispectral import PROJECT_ROOT


V9_ROOT = PROJECT_ROOT / "configs" / "generated" / "data_v9_grouped_cv"
V9_CONTRACT = V9_ROOT / "contract.json"
BASE_FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v4_candidates"
CANDIDATE_FEATURE_ROOT = PROJECT_ROOT / "datasets" / "Data_v7_soil_vegetation_candidates"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "generated" / "data_v10_grouped_feature_cv"
BASE_CHANNEL_INDICES = (0, 1, 2, 5, 6)


def verify_shared_first_five() -> dict[str, object]:
    digest = hashlib.sha256()
    checked = 0
    for split in ("train", "valid"):
        candidate_dir = CANDIDATE_FEATURE_ROOT / split / "features"
        for candidate_path in sorted(candidate_dir.glob("*.npy"), key=lambda path: path.name.lower()):
            base_path = BASE_FEATURE_ROOT / split / "features" / candidate_path.name
            if not base_path.is_file():
                raise FileNotFoundError(base_path)
            base = np.load(base_path, mmap_mode="r", allow_pickle=False)
            candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
            if base.shape[:2] != candidate.shape[:2] or base.shape[2] < 7 or candidate.shape[2] != 7:
                raise RuntimeError(f"Unexpected paired feature shapes: {base_path}={base.shape}, {candidate_path}={candidate.shape}")
            base_five = np.ascontiguousarray(base[:, :, BASE_CHANNEL_INDICES])
            candidate_five = np.ascontiguousarray(candidate[:, :, :5])
            if not np.array_equal(base_five, candidate_five):
                difference = np.abs(base_five.astype(np.int16) - candidate_five.astype(np.int16))
                raise RuntimeError(
                    f"Shared five-channel values differ for {split}/{candidate_path.name}: "
                    f"max={int(difference.max())}, mean={float(difference.mean())}"
                )
            digest.update(split.encode("utf-8"))
            digest.update(b"\0")
            digest.update(candidate_path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(candidate_five.tobytes())
            checked += 1
    if checked != 270:
        raise RuntimeError(f"Expected 270 paired train+valid arrays, got {checked}")
    return {
        "checked_arrays": checked,
        "shared_channels": ["R", "G", "B", "NDVI", "NDRE"],
        "base_channel_indices": list(BASE_CHANNEL_INDICES),
        "candidate_channel_indices": [0, 1, 2, 3, 4],
        "exact_array_match": True,
        "shared_values_sha256": digest.hexdigest(),
    }


def write_yaml(path: Path, train_list: Path, val_list: Path, channels: int) -> None:
    path.write_text(
        "\n".join(
            (
                f"train: {train_list.resolve().as_posix()}",
                f"val: {val_list.resolve().as_posix()}",
                "",
                "nc: 1",
                "names: ['land']",
                f"channels: {channels}",
                "",
            )
        ),
        encoding="utf-8",
    )


def build(output: Path) -> dict[str, object]:
    v9 = json.loads(V9_CONTRACT.read_text(encoding="utf-8"))
    if v9.get("held_out_test_used") or v9.get("test_paths_enumerated"):
        raise RuntimeError("V9 grouped-CV contract indicates sealed-test access")
    pairing = verify_shared_first_five()
    output.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, object]] = []
    for source_fold in v9["fold_contracts"]:
        fold = int(source_fold["fold"])
        fold_dir = output / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_list = Path(source_fold["train_list"])
        val_list = Path(source_fold["validation_list"])
        data_6ch = fold_dir / "data_6ch.yaml"
        data_7ch = fold_dir / "data_7ch.yaml"
        write_yaml(data_6ch, train_list, val_list, 6)
        write_yaml(data_7ch, train_list, val_list, 7)
        folds.append(
            {
                "fold": fold,
                "train_images": int(source_fold["train_images"]),
                "validation_images": int(source_fold["validation_images"]),
                "train_list": str(train_list.resolve()),
                "validation_list": str(val_list.resolve()),
                "data_6ch": str(data_6ch.resolve()),
                "data_7ch": str(data_7ch.resolve()),
            }
        )
    payload = {
        "version": "data_v10_grouped_feature_cv_contract_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_contract": str(V9_CONTRACT.resolve()),
        "source_splits": ["train", "valid"],
        "held_out_split": "test",
        "held_out_test_used": False,
        "test_paths_enumerated": False,
        "training_label_policy": "h0_teacher_refined_mixed",
        "validation_label_policy": "human_only",
        "feature_pairing": pairing,
        "fold_contracts": folds,
    }
    (output / "contract.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.output.resolve())
    print(f"paired_arrays={payload['feature_pairing']['checked_arrays']}")
    print(f"exact_match={payload['feature_pairing']['exact_array_match']}")
    print(f"contract={args.output.resolve() / 'contract.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
