"""Freeze the Data_v3 split/label/model baseline before Data_v4 feature selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from multispectral import PROJECT_ROOT
from spectral_indices import registry_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--channels", type=Path, default=PROJECT_ROOT / "configs" / "data_v4_channels.json")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "runs" / "segment" / "land_seg" / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v3_grouped_reviewed_v1_test_after_error_review" / "metrics.json",
    )
    parser.add_argument(
        "--label-audit",
        type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v3_label_quality_trainval_p2_reviewed" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v4_feature_factor_analysis" / "baseline_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    required_files = (args.channels, args.model, args.metrics, args.label_audit)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing baseline inputs: {missing}")

    splits: dict[str, dict[str, object]] = {}
    for split in ("train", "valid", "test"):
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        images = sorted(path for path in image_dir.glob("*.JPG"))
        labels = sorted(label_dir.glob("*.txt"))
        label_hashes = {path.stem: sha256_file(path) for path in labels}
        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "stems": [path.stem for path in images],
            "labels_sha256": label_hashes,
            "split_signature": hashlib.sha256(
                "\n".join(f"{stem}:{label_hashes.get(stem, 'MISSING')}" for stem in sorted(path.stem for path in images)).encode()
            ).hexdigest(),
        }

    manifest = {
        "version": 1,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dataset": str(dataset),
        "selection_splits": ["train", "valid"],
        "held_out_split": "test",
        "test_policy": "Do not use test images, labels, or metrics for feature selection; evaluate once after the final channel set is frozen.",
        "splits": splits,
        "channel_config": {
            "path": str(args.channels.resolve()),
            "sha256": sha256_file(args.channels),
            "content": json.loads(args.channels.read_text(encoding="utf-8")),
        },
        "index_registry": registry_manifest(),
        "deployment_baseline": {
            "model_name": args.model.parent.parent.name,
            "checkpoint": str(args.model.resolve()),
            "checkpoint_sha256": sha256_file(args.model),
            "test_metrics": json.loads(args.metrics.read_text(encoding="utf-8")),
            "test_metrics_path": str(args.metrics.resolve()),
        },
        "label_quality": json.loads(args.label_audit.read_text(encoding="utf-8")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Baseline frozen: {args.output.resolve()}")
    for split, values in splits.items():
        print(f"{split}: images={values['images']} labels={values['labels']} signature={values['split_signature'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
