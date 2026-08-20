"""Late-fuse instance masks from YOLO segmentation models with different inputs.

The fusion layer is deliberately architecture-agnostic: YOLO11 and YOLOv8,
and RGB/spectral/terrain channel layouts, may be combined after each model has
produced masks in the original image coordinate system. This avoids unsafe
weight averaging between incompatible networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class MaskPrediction:
    mask: np.ndarray
    score: float
    class_id: int = 0
    source: str = "model"
    bbox: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"MaskPrediction expects a 2D mask, got {mask.shape}")
        if not mask.any():
            raise ValueError("MaskPrediction cannot contain an empty mask")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(f"Prediction score must be in [0, 1], got {self.score}")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "class_id", int(self.class_id))
        if self.bbox is None:
            object.__setattr__(self, "bbox", bbox_from_mask(mask))


@dataclass(frozen=True)
class FusedMaskPrediction:
    mask: np.ndarray
    score: float
    class_id: int
    sources: tuple[str, ...]
    bbox: tuple[int, int, int, int]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"Mask shapes differ: {first.shape} versus {second.shape}")
    intersection = np.logical_and(first, second).sum(dtype=np.int64)
    union = np.logical_or(first, second).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Cannot derive a bounding box from an empty mask")
    return int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)


def cropped_mask_iou(
    first: np.ndarray,
    second: np.ndarray,
    first_bbox: tuple[int, int, int, int],
    second_bbox: tuple[int, int, int, int],
) -> float:
    """Compute exact IoU inside the union bbox, avoiding full-frame scans."""
    ix1 = max(first_bbox[0], second_bbox[0])
    iy1 = max(first_bbox[1], second_bbox[1])
    ix2 = min(first_bbox[2], second_bbox[2])
    iy2 = min(first_bbox[3], second_bbox[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    ux1 = min(first_bbox[0], second_bbox[0])
    uy1 = min(first_bbox[1], second_bbox[1])
    ux2 = max(first_bbox[2], second_bbox[2])
    uy2 = max(first_bbox[3], second_bbox[3])
    first_crop = first[uy1:uy2, ux1:ux2]
    second_crop = second[uy1:uy2, ux1:ux2]
    intersection = np.logical_and(first_crop, second_crop).sum(dtype=np.int64)
    union = np.logical_or(first_crop, second_crop).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def _representative(cluster: Sequence[tuple[MaskPrediction, float]]) -> np.ndarray:
    vote = np.zeros(cluster[0][0].mask.shape, dtype=np.float32)
    total = 0.0
    for prediction, model_weight in cluster:
        weight = model_weight * max(prediction.score, 1e-6)
        vote += prediction.mask.astype(np.float32) * weight
        total += weight
    return vote >= total * 0.5


def fuse_mask_predictions(
    predictions_by_model: Sequence[Sequence[MaskPrediction]],
    *,
    model_weights: Sequence[float] | None = None,
    match_iou: float = 0.5,
    mask_vote: float = 0.5,
    score_threshold: float = 0.25,
    min_models: int = 1,
    required_model_indices: Sequence[int] | None = None,
    unconfirmed_score_threshold: float | None = None,
) -> list[FusedMaskPrediction]:
    """Greedily cluster same-class instances and confidence-weight their masks."""
    if not predictions_by_model:
        return []
    weights = tuple(float(value) for value in (model_weights or [1.0] * len(predictions_by_model)))
    if len(weights) != len(predictions_by_model):
        raise ValueError("model_weights length must match predictions_by_model")
    if any(not np.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Every model weight must be positive and finite")
    if not 0.0 <= match_iou <= 1.0 or not 0.0 <= mask_vote <= 1.0:
        raise ValueError("match_iou and mask_vote must be in [0, 1]")
    if min_models < 1:
        raise ValueError("min_models must be at least 1")
    if unconfirmed_score_threshold is not None and not 0.0 <= unconfirmed_score_threshold <= 1.0:
        raise ValueError("unconfirmed_score_threshold must be in [0, 1]")
    required_models = frozenset(int(index) for index in (required_model_indices or ()))
    if any(index < 0 or index >= len(predictions_by_model) for index in required_models):
        raise ValueError("required_model_indices contains an out-of-range model index")

    flattened: list[tuple[MaskPrediction, float, int, tuple[int, int, int, int]]] = []
    expected_shape: tuple[int, int] | None = None
    for model_index, (predictions, model_weight) in enumerate(zip(predictions_by_model, weights)):
        for prediction in predictions:
            if expected_shape is None:
                expected_shape = prediction.mask.shape
            elif prediction.mask.shape != expected_shape:
                raise ValueError(
                    f"All models must output original-size masks; got {prediction.mask.shape} and {expected_shape}"
                )
            assert prediction.bbox is not None
            flattened.append((prediction, model_weight, model_index, prediction.bbox))
    flattened.sort(key=lambda item: item[0].score * item[1], reverse=True)

    clusters: list[list[tuple[MaskPrediction, float, int, tuple[int, int, int, int]]]] = []
    for prediction, model_weight, model_index, prediction_bbox in flattened:
        best_index = None
        best_iou = match_iou
        for cluster_index, cluster in enumerate(clusters):
            if cluster[0][0].class_id != prediction.class_id:
                continue
            # Never merge two detections emitted by the same model. They have
            # already passed that model's NMS and should remain separate.
            if any(existing_model == model_index for _, _, existing_model, _ in cluster):
                continue
            overlap = max(
                cropped_mask_iou(prediction.mask, existing.mask, prediction_bbox, existing_bbox)
                for existing, _, _, existing_bbox in cluster
            )
            if overlap >= best_iou:
                best_iou = overlap
                best_index = cluster_index
        if best_index is None:
            clusters.append([(prediction, model_weight, model_index, prediction_bbox)])
        else:
            clusters[best_index].append((prediction, model_weight, model_index, prediction_bbox))

    fused: list[FusedMaskPrediction] = []
    for cluster in clusters:
        cluster_models = {model_index for _, _, model_index, _ in cluster}
        model_count = len(cluster_models)
        if model_count < min_models:
            continue
        if not required_models.issubset(cluster_models):
            continue
        score_total = 0.0
        model_weight_total = 0.0
        for prediction, model_weight, _, _ in cluster:
            score_total += prediction.score * model_weight
            model_weight_total += model_weight
        if len(cluster) == 1:
            mask = cluster[0][0].mask
            fused_bbox = cluster[0][3]
        else:
            x1 = min(item[3][0] for item in cluster)
            y1 = min(item[3][1] for item in cluster)
            x2 = max(item[3][2] for item in cluster)
            y2 = max(item[3][3] for item in cluster)
            vote = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
            vote_total = 0.0
            for prediction, model_weight, _, _ in cluster:
                confidence_weight = model_weight * max(prediction.score, 1e-6)
                vote += prediction.mask[y1:y2, x1:x2].astype(np.float32) * confidence_weight
                vote_total += confidence_weight
            crop = vote >= vote_total * mask_vote
            mask = np.zeros(cluster[0][0].mask.shape, dtype=bool)
            mask[y1:y2, x1:x2] = crop
            fused_bbox = bbox_from_mask(mask) if mask.any() else (0, 0, 0, 0)
        if not mask.any():
            continue
        score = score_total / model_weight_total
        effective_threshold = (
            unconfirmed_score_threshold
            if model_count == 1 and unconfirmed_score_threshold is not None
            else score_threshold
        )
        if score < effective_threshold:
            continue
        fused.append(
            FusedMaskPrediction(
                mask=mask,
                score=float(score),
                class_id=cluster[0][0].class_id,
                sources=tuple(sorted({prediction.source for prediction, _, _, _ in cluster})),
                bbox=fused_bbox,
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)
