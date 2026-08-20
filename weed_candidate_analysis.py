"""Crop-row constrained weed-candidate analysis for multispectral field images.

This is a review aid, not a weed species classifier.  It separates vegetation
from soil, estimates the dominant crop-row direction, builds row corridors,
and scores vegetation components using off-row position plus red-edge,
texture, and shape novelty.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


WEED_CANDIDATE_WARNING = (
    "杂草候选基于作物行约束及光谱/纹理/形状异常生成，尚无杂草专用人工真值验证；"
    "结果用于巡田复核，不能直接作为施药处方。"
)


def _as_layer(name: str, value: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    layer = np.asarray(value, dtype=np.float32)
    if layer.shape != image_hw or not np.isfinite(layer).all():
        raise ValueError(f"{name} must be a finite HxW layer matching the RGB image")
    return np.clip(layer, 0.0, 1.0)


def vegetation_mask(ndvi: np.ndarray, exg: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
    ndvi_u8 = np.rint(np.clip(ndvi, 0.0, 1.0) * 255.0).astype(np.uint8)
    ndvi_threshold, _ = cv2.threshold(ndvi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ndvi_cut = max(float(ndvi_threshold), 118.0)
    mask = ndvi_u8 >= ndvi_cut
    exg_threshold = None
    if exg is not None:
        exg_u8 = np.rint(np.clip(exg, 0.0, 1.0) * 255.0).astype(np.uint8)
        raw_threshold, _ = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        exg_threshold = max(float(raw_threshold), 126.0)
        green = exg_u8 >= exg_threshold
        # Strong NIR vegetation remains eligible; moderate NDVI needs visible
        # green support.  This suppresses bright soil without losing shadowed crop.
        strong_ndvi = ndvi_u8 >= min(255.0, ndvi_cut + 18.0)
        mask = strong_ndvi | (mask & cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned.astype(bool), {
        "ndvi_threshold_encoded": round(ndvi_cut / 255.0, 4),
        "exg_threshold_encoded": round(exg_threshold / 255.0, 4) if exg_threshold is not None else None,
    }


def _normalize_angle(angle: float) -> float:
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return angle


def estimate_row_orientation(mask: np.ndarray) -> tuple[float, float, int]:
    height, width = mask.shape
    scale = min(1.0, 640.0 / max(height, width))
    small = cv2.resize(
        mask.astype(np.uint8) * 255,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    edges = cv2.Canny(small, 30, 90)
    minimum = max(18, round(min(small.shape) * 0.045))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=max(15, minimum // 2),
        minLineLength=minimum, maxLineGap=max(6, minimum // 2),
    )
    angles: list[float] = []
    weights: list[float] = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, raw)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < minimum:
                continue
            angles.append(_normalize_angle(math.degrees(math.atan2(y2 - y1, x2 - x1))))
            weights.append(length)
    if angles:
        radians = np.deg2rad(np.asarray(angles) * 2.0)
        weighted = np.asarray(weights)
        x = float(np.sum(np.cos(radians) * weighted))
        y = float(np.sum(np.sin(radians) * weighted))
        angle = _normalize_angle(math.degrees(math.atan2(y, x)) / 2.0)
        confidence = math.hypot(x, y) / max(float(np.sum(weighted)), 1e-6)
        return angle, float(np.clip(confidence, 0.0, 1.0)), len(angles)

    rows, columns = np.nonzero(mask)
    if len(rows) < 20:
        return 90.0, 0.0, 0
    points = np.column_stack((columns, rows)).astype(np.float64)
    covariance = np.cov(points - points.mean(axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    angle = _normalize_angle(math.degrees(math.atan2(direction[1], direction[0])))
    confidence = 1.0 - float(values.min() / max(values.max(), 1e-6))
    return angle, float(np.clip(confidence * 0.5, 0.0, 0.5)), 0


def _line_kernel(length: int, angle_deg: float, thickness: int = 3) -> np.ndarray:
    size = max(7, int(length) | 1)
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = (size - 1) / 2.0
    radius = (size - 3) / 2.0
    radians = math.radians(angle_deg)
    dx, dy = math.cos(radians) * radius, math.sin(radians) * radius
    first = (round(center - dx), round(center - dy))
    second = (round(center + dx), round(center + dy))
    cv2.line(kernel, first, second, 1, max(1, thickness), cv2.LINE_8)
    return kernel


def build_crop_row_corridors(mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray, int]:
    height, width = mask.shape
    line_length = max(15, round(min(height, width) * 0.045)) | 1
    line = _line_kernel(line_length, angle_deg, thickness=max(1, line_length // 15))
    binary = mask.astype(np.uint8)
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, line)
    row_core = cv2.morphologyEx(connected, cv2.MORPH_OPEN, line)
    # A compact weed blob can survive a line opening if its diameter happens
    # to exceed the kernel.  Keep only components with meaningful extent in
    # the dominant row direction.
    core_count, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(row_core, connectivity=8)
    filtered_core = np.zeros_like(row_core)
    radians = math.radians(angle_deg)
    direction = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    minimum_span = max(line_length * 2.0, min(height, width) * 0.09)
    for label in range(1, core_count):
        if int(core_stats[label, cv2.CC_STAT_AREA]) < line_length:
            continue
        rows, columns = np.nonzero(core_labels == label)
        projection = np.column_stack((columns, rows)) @ direction
        if float(projection.max() - projection.min()) >= minimum_span:
            filtered_core[core_labels == label] = 1
    row_core = filtered_core
    radius = max(2, round(line_length * 0.12))
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    corridors = cv2.dilate(row_core, band_kernel).astype(bool)
    return row_core.astype(bool), corridors, line_length


def _component_shape_score(columns: np.ndarray, rows: np.ndarray) -> float:
    if len(rows) < 3:
        return 1.0
    points = np.column_stack((columns, rows)).astype(np.float64)
    values = np.linalg.eigvalsh(np.cov(points - points.mean(axis=0), rowvar=False))
    if values.max() <= 1e-6:
        return 1.0
    return float(np.clip(math.sqrt(max(values.min(), 0.0) / values.max()), 0.0, 1.0))


def _robust_novelty(value: float, baseline: list[float], *, high_only: bool = False) -> float:
    if len(baseline) < 3:
        return 0.0
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(np.asarray(baseline) - median)))
    delta = value - median if high_only else abs(value - median)
    if high_only:
        delta = max(delta, 0.0)
    return float(np.clip(delta / max(3.0 * 1.4826 * mad, 1e-4), 0.0, 1.0))


def analyze_weed_candidates(
    image_rgb: np.ndarray,
    ndvi: np.ndarray,
    ndre: np.ndarray,
    *,
    exg: np.ndarray | None = None,
    gsd: float = 0.028,
) -> dict[str, Any]:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8 HxWx3")
    height, width = image.shape[:2]
    ndvi_layer = _as_layer("NDVI", ndvi, (height, width))
    ndre_layer = _as_layer("NDRE", ndre, (height, width))
    exg_layer = _as_layer("ExG", exg, (height, width)) if exg is not None else None
    vegetation, thresholds = vegetation_mask(ndvi_layer, exg_layer)
    angle, orientation_confidence, supporting_lines = estimate_row_orientation(vegetation)
    row_core, row_corridors, line_length = build_crop_row_corridors(vegetation, angle)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_mean = cv2.boxFilter(gray, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT101)
    local_square = cv2.boxFilter(gray * gray, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT101)
    texture = np.sqrt(np.maximum(local_square - local_mean * local_mean, 0.0))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        vegetation.astype(np.uint8), connectivity=8
    )
    minimum_pixels = max(6, round(height * width * 1e-6))
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_pixels:
            continue
        rows, columns = np.nonzero(labels == label)
        off_fraction = float(np.mean(~row_corridors[rows, columns]))
        components.append({
            "label": label,
            "area_pixels": area,
            "off_row_fraction": off_fraction,
            "ndre_mean": float(np.mean(ndre_layer[rows, columns])),
            "texture_mean": float(np.mean(texture[rows, columns])),
            "shape_score": _component_shape_score(columns, rows),
        })

    baseline = [item for item in components if item["off_row_fraction"] <= 0.2]
    ndre_baseline = [float(item["ndre_mean"]) for item in baseline]
    texture_baseline = [float(item["texture_mean"]) for item in baseline]
    candidate_mask = np.zeros((height, width), dtype=bool)
    off_row_count = in_row_count = 0
    candidate_rows: list[dict[str, Any]] = []
    for item in components:
        off = float(item["off_row_fraction"])
        spectral = _robust_novelty(float(item["ndre_mean"]), ndre_baseline)
        texture_score = _robust_novelty(float(item["texture_mean"]), texture_baseline, high_only=True)
        shape = float(item["shape_score"])
        in_row_risk = 0.4 * spectral + 0.3 * texture_score + 0.3 * shape
        risk = 0.65 * off + 0.15 * spectral + 0.1 * texture_score + 0.1 * shape
        is_off_row = off >= 0.45
        is_in_row_suspect = off < 0.45 and in_row_risk >= 0.72 and orientation_confidence >= 0.25
        if not (is_off_row or is_in_row_suspect):
            continue
        selected = labels == int(item["label"])
        candidate_mask |= selected
        off_row_count += int(is_off_row)
        in_row_count += int(is_in_row_suspect)
        candidate_rows.append({
            "area_pixels": int(item["area_pixels"]),
            "area_m2": round(float(item["area_pixels"]) * gsd * gsd, 4),
            "off_row_fraction": round(off, 4),
            "spectral_novelty": round(spectral, 4),
            "texture_novelty": round(texture_score, 4),
            "shape_score": round(shape, 4),
            "risk_score": round(max(risk, in_row_risk if is_in_row_suspect else 0.0), 4),
            "reason": "off_row_vegetation" if is_off_row else "in_row_multifeature_anomaly",
        })

    overlay = image.copy()
    tinted = overlay.copy()
    tinted[row_corridors] = (40, 155, 190)
    crop_vegetation = vegetation & row_corridors & ~candidate_mask
    tinted[crop_vegetation] = (40, 205, 95)
    tinted[candidate_mask] = (245, 65, 90)
    overlay = cv2.addWeighted(overlay, 0.58, tinted, 0.42, 0)
    contours, _ = cv2.findContours(candidate_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 230, 50), 2, cv2.LINE_AA)
    center = (width // 2, height // 2)
    radius = max(35, min(height, width) // 8)
    radians = math.radians(angle)
    delta = (round(math.cos(radians) * radius), round(math.sin(radians) * radius))
    cv2.line(
        overlay, (center[0] - delta[0], center[1] - delta[1]),
        (center[0] + delta[0], center[1] + delta[1]), (255, 255, 255), 4, cv2.LINE_AA,
    )
    cv2.line(
        overlay, (center[0] - delta[0], center[1] - delta[1]),
        (center[0] + delta[0], center[1] + delta[1]), (30, 80, 220), 2, cv2.LINE_AA,
    )
    banner_height = 58
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (18, 24, 28), -1)
    cv2.putText(
        overlay, f"Crop rows {angle:+.1f} deg | weed candidates {len(candidate_rows)}",
        (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        overlay, "cyan=row corridor  green=crop-row vegetation  red=review candidate",
        (14, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (205, 220, 225), 1, cv2.LINE_AA,
    )

    image_pixels = max(height * width, 1)
    vegetation_pixels = int(vegetation.sum())
    candidate_pixels = int(candidate_mask.sum())
    metrics = {
        **thresholds,
        "row_angle_deg": round(angle, 3),
        "row_orientation_confidence": round(orientation_confidence, 4),
        "supporting_line_count": supporting_lines,
        "row_line_kernel_pixels": line_length,
        "vegetation_pct": round(100.0 * vegetation_pixels / image_pixels, 3),
        "crop_row_vegetation_pct": round(100.0 * int(crop_vegetation.sum()) / image_pixels, 3),
        "weed_candidate_pct": round(100.0 * candidate_pixels / image_pixels, 3),
        "weed_candidate_area_m2": round(candidate_pixels * gsd * gsd, 3),
        "weed_candidate_count": len(candidate_rows),
        "off_row_candidate_count": off_row_count,
        "in_row_suspect_count": in_row_count,
        "method": "crop_row_constraint_plus_spectral_texture_shape_novelty_v1",
        "review_required": True,
    }
    return {
        "overlay": overlay,
        "vegetation_mask": vegetation,
        "row_core_mask": row_core,
        "row_corridor_mask": row_corridors,
        "candidate_mask": candidate_mask,
        "candidates": sorted(candidate_rows, key=lambda row: row["risk_score"], reverse=True),
        "metrics": metrics,
        "warning": WEED_CANDIDATE_WARNING,
    }
