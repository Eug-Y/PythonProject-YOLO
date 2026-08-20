"""Export valid-split mask predictions from one frozen multichannel YOLO checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask

from custom_modules import CoordAtt, ECA  # noqa: F401 - checkpoint deserialization
from multispectral import PROJECT_ROOT, read_image
from prediction_bundle import save_prediction_bundle
from yolo_multichannel_data import _load_selected_array, feature_path_for_image, validate_channel_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def infer_masks(model, image: np.ndarray, *, imgsz: int, conf: float, iou: float, device: torch.device) -> dict:
    height, width = image.shape[:2]
    stride = int(model.stride.max())
    if imgsz <= 0 or imgsz % stride:
        raise ValueError(f"imgsz must be a positive multiple of model stride {stride}, got {imgsz}")
    ratio = min(imgsz / height, imgsz / width)
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_left = (imgsz - resized_width) // 2
    pad_top = (imgsz - resized_height) // 2
    canvas = np.full((imgsz, imgsz, image.shape[2]), 114, dtype=image.dtype)
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    tensor = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32) / 255.0
    with torch.no_grad():
        output = model._predict_once(tensor)
    prediction, proto = output[0]
    detections = non_max_suppression(prediction, conf, iou, nc=1)[0]
    if detections is None or not len(detections):
        return {
            "masks": np.zeros((0, height, width), dtype=bool),
            "scores": np.zeros(0, dtype=np.float32),
            "class_ids": np.zeros(0, dtype=np.int16),
            "boxes": np.zeros((0, 4), dtype=np.float32),
        }
    masks = process_mask(proto[0], detections[:, 6:], detections[:, :4], (imgsz, imgsz), upsample=True)
    masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    original_masks = np.stack(
        [
            cv2.resize(
                mask[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks_np
        ]
    )
    boxes = detections[:, :4].detach().cpu().numpy().astype(np.float32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - pad_left, 0, resized_width) / ratio
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - pad_top, 0, resized_height) / ratio
    scores = detections[:, 4].detach().cpu().numpy().astype(np.float32)
    class_ids = detections[:, 5].detach().cpu().numpy().astype(np.int16)
    nonempty = original_masks.any(axis=(1, 2))
    return {
        "masks": original_masks[nonempty],
        "scores": scores[nonempty],
        "class_ids": class_ids[nonempty],
        "boxes": boxes[nonempty],
    }


def main() -> int:
    from ultralytics.nn.tasks import load_checkpoint

    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_channels = tuple(manifest["channels"])
    channels = tuple(args.channels)
    channel_indices = validate_channel_selection(all_channels, channels)
    images = sorted((args.dataset / args.split / "images").glob("*.JPG"))
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found in {args.dataset / args.split / 'images'}")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--conf and --iou must be in [0, 1]")
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model, _ = load_checkpoint(str(args.checkpoint.resolve()), device=device)
    actual_channels = int(model.model[0].conv.in_channels)
    if actual_channels != len(channels):
        raise ValueError(f"Checkpoint expects {actual_channels} channels, selection contains {len(channels)}")
    model.float().to(device).eval()

    output = args.output.resolve()
    bundle_dir = output / "predictions"
    failures: list[dict[str, str]] = []
    total_instances = 0
    for index, image_path in enumerate(images, 1):
        try:
            feature_path = feature_path_for_image(image_path, args.feature_root)
            array = _load_selected_array(feature_path, channel_indices, len(all_channels))
            image_hw = read_image(image_path).shape[:2]
            if array.shape[:2] != image_hw:
                raise ValueError(f"Feature/image shape mismatch: {array.shape[:2]} versus {image_hw}")
            result = infer_masks(model, array, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device)
            save_prediction_bundle(bundle_dir / f"{image_path.stem}.npz", **result)
            total_instances += len(result["scores"])
        except Exception as exc:
            failures.append({"image": image_path.name, "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0 or index == len(images):
            print(f"[{args.split}] {index}/{len(images)} instances={total_instances} failures={len(failures)}")

    export_manifest = {
        "version": "multichannel_yolo_prediction_export_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "feature_root": str(args.feature_root.resolve()),
        "feature_manifest": str(args.manifest.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "held_out_test_used": False,
        "channels": list(channels),
        "image_count": len(images),
        "total_instances": total_instances,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {output / 'manifest.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
