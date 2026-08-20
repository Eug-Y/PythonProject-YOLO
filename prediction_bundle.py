"""Compact, validated storage for original-resolution instance-mask predictions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def save_prediction_bundle(
    path: str | Path,
    masks: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    boxes: np.ndarray | None = None,
) -> Path:
    path = Path(path)
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids, dtype=np.int16)
    if masks.ndim != 3:
        raise ValueError(f"masks must be NxHxW, got {masks.shape}")
    count, height, width = masks.shape
    if scores.shape != (count,) or class_ids.shape != (count,):
        raise ValueError("scores and class_ids must have one value per mask")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be finite and in [0, 1]")
    if boxes is None:
        boxes = np.zeros((count, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape != (count, 4) or not np.isfinite(boxes).all():
        raise ValueError(f"boxes must be finite Nx4, got {boxes.shape}")
    packed = np.packbits(masks.reshape(count, height * width), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        packed_masks=packed,
        scores=scores,
        class_ids=class_ids,
        boxes=boxes,
        image_hw=np.asarray([height, width], dtype=np.int32),
        mask_pixels=np.asarray([height * width], dtype=np.int64),
    )
    os.replace(temporary, path)
    return path


def load_prediction_bundle(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"packed_masks", "scores", "class_ids", "boxes", "image_hw", "mask_pixels"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Prediction bundle is missing arrays {sorted(missing)}: {path}")
        packed = payload["packed_masks"]
        scores = payload["scores"].astype(np.float32, copy=False)
        class_ids = payload["class_ids"].astype(np.int16, copy=False)
        boxes = payload["boxes"].astype(np.float32, copy=False)
        image_hw = payload["image_hw"].astype(np.int32, copy=False)
        mask_pixels = int(payload["mask_pixels"][0])
    if image_hw.shape != (2,) or np.any(image_hw <= 0):
        raise ValueError(f"Invalid image_hw in prediction bundle: {image_hw}")
    height, width = map(int, image_hw)
    if mask_pixels != height * width or packed.ndim != 2:
        raise ValueError(f"Invalid packed mask dimensions in {path}")
    count = packed.shape[0]
    if scores.shape != (count,) or class_ids.shape != (count,) or boxes.shape != (count, 4):
        raise ValueError(f"Prediction arrays have inconsistent counts in {path}")
    unpacked = np.unpackbits(packed, axis=1, count=mask_pixels).reshape(count, height, width).astype(bool)
    return {
        "masks": unpacked,
        "scores": scores,
        "class_ids": class_ids,
        "boxes": boxes,
        "image_hw": image_hw,
    }

