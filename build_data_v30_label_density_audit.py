"""Audit instance density and fragmentation in the Data_v3 train/valid labels.

Motivation
----------
The SAM2-assisted batch inside Data_v3 (100 images) carries roughly 2.1x the
polygons per image of the hand-labelled base batch (about 79.5 versus 37.9), and
the densest images are exactly the recall-priority scenes that V28/V29 could not
fix. Two very different diagnoses fit that observation:

  (a) over-segmentation - one field/plant region was split into several
      contours, so the label count is inflated and the model is punished for
      merging pieces that a human considers one instance;
  (b) real density - the scene genuinely contains that many instances, so the
      only fixes are resolution, tiling or capacity.

This module decides between them with indicators that were fixed *before*
looking at the real data.

Pre-registered indicators (SAM2 batch measured against the base batch control)
---------------------------------------------------------------------------
  A1 overlap      share of polygons that have at least one same-image partner
                  with mask IoU > ``OVERLAP_IOU`` (0.30).
  A2 tiling       share of polygons touching at least ``TOUCH_NEIGHBOURS`` (2)
                  neighbours, where a neighbour is a same-image polygon whose
                  ``DILATE_PX``-dilated mask covers more than
                  ``TOUCH_FRACTION`` (5%) of its own area.
  A3 fragmentation median polygon equivalent radius ``sqrt(area/pi)`` in grid
                  units, relative to the base batch.

Decision rule (fixed in advance)
--------------------------------
  * ``over_segmentation`` if the median polygon count ratio is at least
    ``POLYGON_RATIO_MIN`` (1.5) and at least two indicators fire, where an
    indicator fires when the SAM2 value is at least ``INDICATOR_RATIO`` (2.0)
    times the base value (for A3, when the radius is at most half the base).
  * ``real_density`` if the polygon ratio is at least 1.5 but fewer than two
    indicators fire.
  * ``inconclusive`` otherwise.

Scope
-----
Read-only. Only ``train`` and ``valid`` can be requested; ``test`` is not even
an accepted value, and ``datasets/Data_release_blind_v1`` is never opened.

Outputs
-------
``reports/Data_v30_label_density_audit/summary.json``
``reports/Data_v30_label_density_audit/per_image_density.csv``
``reports/Data_v30_label_density_audit/index.html``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "Data_v3"
DEFAULT_SAM2_LABELS = PROJECT_ROOT / "待标注_100张" / "labels"
DEFAULT_METRICS = PROJECT_ROOT / "reports" / "Project_progress_report" / "per_image_metrics.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "Data_v30_label_density_audit"

GRID_WIDTH = 640
GRID_HEIGHT = 480
AUDITED_SPLITS = ("train", "valid")

OVERLAP_IOU = 0.30
DILATE_PX = 2
TOUCH_FRACTION = 0.05
TOUCH_NEIGHBOURS = 2
INDICATOR_RATIO = 2.0
POLYGON_RATIO_MIN = 1.5

VERSION = "data_v30_label_density_dataset_audit_v1"


# --------------------------------------------------------------------------- #
# label parsing and geometry
# --------------------------------------------------------------------------- #
def read_polygons(label_path: Path) -> tuple[list[np.ndarray], int]:
    """Return normalized polygons plus the number of malformed lines skipped."""
    polygons: list[np.ndarray] = []
    malformed = 0
    if not label_path.is_file():
        return polygons, malformed
    for raw_line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            malformed += 1
            continue
        if len(values) < 6 or len(values) % 2:
            malformed += 1
            continue
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        if not np.all(np.isfinite(points)):
            malformed += 1
            continue
        polygons.append(points)
    return polygons, malformed


def normalized_to_grid(points: np.ndarray) -> np.ndarray:
    """Map normalized coordinates onto the audit grid, clipped to its bounds."""
    grid = np.empty_like(points)
    grid[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * (GRID_WIDTH - 1)
    grid[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * (GRID_HEIGHT - 1)
    return grid


def equivalent_radius(area: float) -> float:
    """Radius of the circle with the given area, in grid units."""
    return float(np.sqrt(max(area, 0.0) / np.pi))


def polygon_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Mask IoU of two boolean masks that share a common ROI."""
    intersection = int(np.count_nonzero(mask_a & mask_b))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(mask_a)) + int(np.count_nonzero(mask_b)) - intersection
    return intersection / union if union else 0.0


def dilate_mask(mask: np.ndarray, radius: int = DILATE_PX) -> np.ndarray:
    """Binary-dilate a boolean mask with an elliptical kernel."""
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


class Instance:
    """One rasterized instance plus the geometry needed for pairing."""

    __slots__ = ("index", "points", "mask", "bbox", "area")

    def __init__(self, index: int, points: np.ndarray) -> None:
        self.index = index
        self.points = points
        self.mask, self.bbox = rasterize(points)
        self.area = float(np.count_nonzero(self.mask))

    @property
    def radius(self) -> float:
        return equivalent_radius(self.area)


def rasterize(points: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rasterize one polygon and crop it to its bounding box."""
    canvas = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    contour = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [contour], 1)
    ys, xs = np.nonzero(canvas)
    if xs.size == 0:
        return np.zeros((1, 1), dtype=bool), (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return canvas[y0:y1 + 1, x0:x1 + 1].astype(bool), (x0, y0, x1, y1)


def _bboxes_near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _place(mask: np.ndarray, bbox: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    """Place a cropped mask inside a larger ROI boolean mask."""
    x0, y0, x1, y1 = roi
    out = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    out[
        bbox[1] - y0: bbox[1] - y0 + mask.shape[0],
        bbox[0] - x0: bbox[0] - x0 + mask.shape[1],
    ] = mask
    return out


def pair_metrics(instances: Sequence[Instance]) -> dict[str, object]:
    """Compute overlap, tiling and union statistics for one image."""
    count = len(instances)
    result: dict[str, object] = {
        "polygons": count,
        "overlap_pairs": 0,
        "overlap_polygons": 0,
        "tiling_polygons": 0,
        "union_coverage": 0.0,
        "union_over_sum": 1.0,
        "median_radius": 0.0,
        "median_area": 0.0,
        "max_iou": 0.0,
        "overlap_share": 0.0,
        "tiling_share": 0.0,
        "malformed": 0,
    }
    if count == 0:
        return result

    union_canvas = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    for instance in instances:
        x0, y0, x1, y1 = instance.bbox
        if x1 >= x0 and y1 >= y0:
            union_canvas[y0:y1 + 1, x0:x1 + 1] |= instance.mask.astype(np.uint8)

    has_overlap = [False] * count
    touch_counts = [0] * count
    overlap_pairs = 0
    max_iou = 0.0
    for i in range(count):
        for j in range(i + 1, count):
            first, second = instances[i], instances[j]
            if not _bboxes_near(first.bbox, second.bbox, DILATE_PX):
                continue
            roi = (
                min(first.bbox[0], second.bbox[0]),
                min(first.bbox[1], second.bbox[1]),
                max(first.bbox[2], second.bbox[2]),
                max(first.bbox[3], second.bbox[3]),
            )
            mask_i = _place(first.mask, first.bbox, roi)
            mask_j = _place(second.mask, second.bbox, roi)
            iou = polygon_iou(mask_i, mask_j)
            max_iou = max(max_iou, iou)
            if iou > OVERLAP_IOU:
                overlap_pairs += 1
                has_overlap[i] = has_overlap[j] = True
            dilated_j = dilate_mask(mask_j)
            dilated_i = dilate_mask(mask_i)
            area_i = int(np.count_nonzero(mask_i))
            area_j = int(np.count_nonzero(mask_j))
            if area_i and np.count_nonzero(mask_i & dilated_j) / area_i > TOUCH_FRACTION:
                touch_counts[i] += 1
            if area_j and np.count_nonzero(mask_j & dilated_i) / area_j > TOUCH_FRACTION:
                touch_counts[j] += 1

    areas = [instance.area for instance in instances]
    radar = sorted(instance.radius for instance in instances)
    middle = len(radar) // 2
    result.update(
        {
            "overlap_pairs": overlap_pairs,
            "overlap_polygons": int(sum(has_overlap)),
            "tiling_polygons": int(sum(1 for value in touch_counts if value >= TOUCH_NEIGHBOURS)),
            "union_coverage": float(np.count_nonzero(union_canvas)) / (GRID_WIDTH * GRID_HEIGHT),
            "union_over_sum": float(np.count_nonzero(union_canvas)) / float(sum(areas)),
            "median_radius": float(np.median(radar)),
            "median_area": float(np.median(areas)),
            "max_iou": float(max_iou),
            "overlap_share": float(sum(has_overlap)) / count,
            "tiling_share": float(sum(1 for value in touch_counts if value >= TOUCH_NEIGHBOURS)) / count,
        }
    )
    return result


def image_metrics(label_path: Path) -> dict[str, object]:
    """Rasterize one image's labels and score its instance density."""
    polygons, malformed = read_polygons(label_path)
    instances = [Instance(index, normalized_to_grid(points)) for index, points in enumerate(polygons)]
    instances = [instance for instance in instances if instance.area > 0]
    metrics = pair_metrics(instances)
    metrics["malformed"] = malformed
    metrics["polygons"] = len(instances)
    return metrics


# --------------------------------------------------------------------------- #
# dataset walking
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_labels(paths: Iterable[Path]) -> str:
    """Hash a label set by content, so the audit is reproducible from filenames."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def label_paths(dataset_root: Path, split: str) -> list[Path]:
    directory = dataset_root / split / "labels"
    return sorted(directory.glob("*.txt")) if directory.is_dir() else []


def audit_source(split: str, dataset_root: Path, sam2_stems: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in label_paths(dataset_root, split):
        source = "sam2_new" if path.stem in sam2_stems else "base"
        rows.append({"split": split, "stem": path.stem, "source": source, **image_metrics(path)})
    return rows


# --------------------------------------------------------------------------- #
# aggregation and decision
# --------------------------------------------------------------------------- #
def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_batch(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    polygons = [float(row["polygons"]) for row in rows]
    return {
        "images": len(rows),
        "polygons_total": int(sum(polygons)),
        "polygons_per_image_mean": _mean(polygons),
        "polygons_per_image_median": _median(polygons),
        "polygons_per_image_max": int(max(polygons)) if polygons else 0,
        "median_radius": _median([float(row["median_radius"]) for row in rows]),
        "overlap_share_mean": _mean([float(row["overlap_share"]) for row in rows]),
        "tiling_share_mean": _mean([float(row["tiling_share"]) for row in rows]),
        "union_over_sum_mean": _mean([float(row["union_over_sum"]) for row in rows]),
        "max_iou_mean": _mean([float(row["max_iou"]) for row in rows]),
    }


def inconclusive_decision() -> dict[str, object]:
    """Decision payload used when a control batch is unavailable."""
    return {
        "polygon_ratio": None,
        "indicators": {},
        "indicators_fired": 0,
        "decision": "inconclusive",
        "thresholds": {
            "overlap_iou": OVERLAP_IOU,
            "dilate_px": DILATE_PX,
            "touch_fraction": TOUCH_FRACTION,
            "touch_neighbours": TOUCH_NEIGHBOURS,
            "indicator_ratio": INDICATOR_RATIO,
            "polygon_ratio_min": POLYGON_RATIO_MIN,
            "fingerprint": {},
        },
    }


def decide(sam2: dict[str, object], base: dict[str, object]) -> dict[str, object]:
    """Apply the pre-registered indicator and decision rule."""
    base_polygons = float(base["polygons_per_image_median"])
    sam2_polygons = float(sam2["polygons_per_image_median"])
    polygon_ratio = sam2_polygons / base_polygons if base_polygons else float("inf")

    def fires(sam2_value: float, base_value: float) -> bool:
        if base_value <= 0:
            return sam2_value > TOUCH_FRACTION
        return sam2_value >= INDICATOR_RATIO * base_value

    base_radius = float(base["median_radius"])
    sam2_radius = float(sam2["median_radius"])
    indicators = {
        "A1_overlap": fires(float(sam2["overlap_share_mean"]), float(base["overlap_share_mean"])),
        "A2_tiling": fires(float(sam2["tiling_share_mean"]), float(base["tiling_share_mean"])),
        "A3_fragmentation": bool(base_radius > 0 and sam2_radius <= base_radius / INDICATOR_RATIO),
    }
    fired = sum(indicators.values())

    if polygon_ratio >= POLYGON_RATIO_MIN and fired >= 2:
        decision = "over_segmentation"
    elif polygon_ratio >= POLYGON_RATIO_MIN:
        decision = "real_density"
    else:
        decision = "inconclusive"

    return {
        "polygon_ratio": float(polygon_ratio) if np.isfinite(polygon_ratio) else None,
        "indicators": indicators,
        "indicators_fired": fired,
        "decision": decision,
        "thresholds": {
            "overlap_iou": OVERLAP_IOU,
            "dilate_px": DILATE_PX,
            "touch_fraction": TOUCH_FRACTION,
            "touch_neighbours": TOUCH_NEIGHBOURS,
            "indicator_ratio": INDICATOR_RATIO,
            "polygon_ratio_min": POLYGON_RATIO_MIN,
            "fingerprint": {
                "overlap_share": {
                    "sam2": float(sam2["overlap_share_mean"]),
                    "base": float(base["overlap_share_mean"]),
                },
                "tiling_share": {
                    "sam2": float(sam2["tiling_share_mean"]),
                    "base": float(base["tiling_share_mean"]),
                },
                "median_radius": {"sam2": sam2_radius, "base": base_radius},
            },
        },
    }


def rank_correlation(pairs: Sequence[tuple[float, float]]) -> float:
    """Spearman rank correlation without a SciPy dependency."""
    if len(pairs) < 3:
        return 0.0

    def ranks(values: Sequence[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
        out = np.empty(len(values), dtype=np.float64)
        out[order] = np.arange(len(values), dtype=np.float64)
        return out

    first = ranks([pair[0] for pair in pairs])
    second = ranks([pair[1] for pair in pairs])
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def recall_linkage(rows: Sequence[dict[str, object]], metrics_path: Path) -> dict[str, object]:
    """Correlate label density on valid with the frozen production recall."""
    if not metrics_path.is_file():
        return {"available": False, "reason": f"missing {metrics_path.name}"}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("held_out_test_used"):
        return {"available": False, "reason": "metrics file declares held-out test usage"}
    by_stem = {str(item["stem"]): item for item in payload.get("images", [])}
    valid_rows = [row for row in rows if row["split"] == "valid" and row["stem"] in by_stem]
    if not valid_rows:
        return {"available": False, "reason": "no overlapping valid images"}

    joined = []
    for row in valid_rows:
        fusion = by_stem[str(row["stem"])].get("fusion", {})
        ground_truth = float(fusion.get("ground_truth") or by_stem[str(row["stem"])].get("ground_truth") or 0)
        fn = float(fusion.get("fn", 0))
        joined.append(
            {
                "stem": row["stem"],
                "source": row["source"],
                "polygons": float(row["polygons"]),
                "ground_truth": ground_truth,
                "fn": fn,
                "recall": float(fusion.get("recall", 0.0)),
                "fn_rate": (fn / ground_truth) if ground_truth else 0.0,
            }
        )

    order = np.argsort([item["polygons"] for item in joined], kind="mergesort")
    bucketed: list[dict[str, object]] = []
    for chunk in np.array_split(np.asarray(order), 3):
        group = [joined[int(index)] for index in chunk]
        bucketed.append(
            {
                "images": len(group),
                "polygons_mean": _mean([item["polygons"] for item in group]),
                "ground_truth_mean": _mean([item["ground_truth"] for item in group]),
                "fn_mean": _mean([item["fn"] for item in group]),
                "fn_rate_mean": _mean([item["fn_rate"] for item in group]),
                "recall_mean": _mean([item["recall"] for item in group]),
            }
        )
    return {
        "available": True,
        "source": str(metrics_path),
        "source_sha256": sha256_file(metrics_path),
        "images": len(joined),
        "spearman_polygons_vs_fn_rate": rank_correlation(
            [(item["polygons"], item["fn_rate"]) for item in joined]
        ),
        "spearman_polygons_vs_recall": rank_correlation(
            [(item["polygons"], item["recall"]) for item in joined]
        ),
        "terciles": bucketed,
        "by_source": {
            source: {
                "images": len(group),
                "polygons_mean": _mean([item["polygons"] for item in group]),
                "fn_rate_mean": _mean([item["fn_rate"] for item in group]),
                "recall_mean": _mean([item["recall"] for item in group]),
            }
            for source in ("sam2_new", "base")
            for group in [[item for item in joined if item["source"] == source]]
            if group
        },
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _bar(value: float, scale: float, digits: int = 3) -> str:
    width = 0.0 if scale <= 0 else min(100.0, max(0.0, value / scale * 100.0))
    return (
        f'<span class="bar"><span style="width:{width:.1f}%"></span></span>'
        f'<span class="num">{value:.{digits}f}</span>'
    )


def render_html(summary: dict[str, object]) -> str:
    decision = summary["decision"]["decision"]
    headline = {
        "over_segmentation": (
            "结论：SAM2 批的高轮廓数主要是<b>过分割</b> —— 同一株/同一田块被切成多个轮廓，"
            "标签侧修复（合并、去重）优先于分辨率改造。"
        ),
        "real_density": (
            "结论：SAM2 批的高轮廓数更像<b>真实高密度</b> —— 轮廓彼此分离且尺寸与人工批相当，"
            "应转向分辨率/切片路线，而不是继续做标签合并。"
        ),
        "inconclusive": "结论：证据不足以二选一，需扩大样本或改用人工抽检判定。",
    }[decision]

    def batch_table(batches: dict[str, object]) -> str:
        rows = []
        for name, batch in batches.items():
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(name))}</td>"
                f"<td class=\"num\">{batch['images']}</td>"
                f"<td class=\"num\">{batch['polygons_total']}</td>"
                f"<td class=\"num\">{batch['polygons_per_image_mean']:.1f}</td>"
                f"<td class=\"num\">{batch['polygons_per_image_median']:.0f}</td>"
                f"<td class=\"num\">{batch['polygons_per_image_max']}</td>"
                f"<td class=\"num\">{batch['median_radius']:.2f}</td>"
                f"<td>{_bar(float(batch['overlap_share_mean']), 0.5)}</td>"
                f"<td>{_bar(float(batch['tiling_share_mean']), 0.5)}</td>"
                f"<td class=\"num\">{batch['union_over_sum_mean']:.3f}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    per_split = summary["per_split"]
    split_sections = []
    for split, payload in per_split.items():
        split_sections.append(
            f"<h3>{html.escape(split)}（{payload['decision']['decision']}）</h3>"
            "<table><thead><tr><th>批次</th><th>图像</th><th>轮廓总数</th><th>均值/图</th>"
            "<th>中位/图</th><th>最大/图</th><th>中位等效半径</th><th>重叠轮廓占比</th>"
            "<th>拼接轮廓占比</th><th>并集/面积和</th></tr></thead><tbody>"
            f"{batch_table(payload['batches'])}</tbody></table>"
        )

    linkage = summary["recall_linkage"]
    if linkage.get("available"):
        terciles = "".join(
            "<tr>"
            f"<td class=\"num\">{index + 1}</td>"
            f"<td class=\"num\">{item['images']}</td>"
            f"<td class=\"num\">{item['polygons_mean']:.1f}</td>"
            f"<td class=\"num\">{item['ground_truth_mean']:.1f}</td>"
            f"<td class=\"num\">{item['fn_mean']:.1f}</td>"
            f"<td class=\"num\">{item['fn_rate_mean']:.3f}</td>"
            f"<td class=\"num\">{item['recall_mean']:.3f}</td>"
            "</tr>"
            for index, item in enumerate(linkage["terciles"])
        )
        linkage_html = (
            "<p>Spearman（轮廓数 vs FN 率）："
            f"<b>{linkage['spearman_polygons_vs_fn_rate']:+.3f}</b>；"
            "（轮廓数 vs 召回）："
            f"<b>{linkage['spearman_polygons_vs_recall']:+.3f}</b>，"
            f"基于 {linkage['images']} 张 valid 图。</p>"
            "<table><thead><tr><th>等分位</th><th>图像</th><th>平均轮廓数</th><th>平均真值</th>"
            "<th>平均 FN</th><th>平均 FN 率</th><th>平均召回</th></tr></thead><tbody>"
            f"{terciles}</tbody></table>"
        )
    else:
        linkage_html = f"<p>未接入冻结生产指标：{html.escape(str(linkage.get('reason')))}</p>"

    densest = summary["densest_images"]
    densest_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['stem']))}</td>"
        f"<td>{html.escape(str(item['split']))}</td>"
        f"<td>{html.escape(str(item['source']))}</td>"
        f"<td class=\"num\">{item['polygons']}</td>"
        f"<td class=\"num\">{item['median_radius']:.2f}</td>"
        f"<td class=\"num\">{item['overlap_share']:.3f}</td>"
        f"<td class=\"num\">{item['tiling_share']:.3f}</td>"
        "</tr>"
        for item in densest
    )

    indicators = summary["decision"]["indicators"]
    ratio = summary["decision"].get("polygon_ratio")
    ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
    indicator_html = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'命中' if value else '未命中'}</td></tr>"
        for name, value in indicators.items()
    )
    fingerprint = summary["decision"]["thresholds"]["fingerprint"]
    fingerprint_html = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td class=\"num\">{values['sam2']:.3f}</td>"
        f"<td class=\"num\">{values['base']:.3f}</td></tr>"
        for name, values in fingerprint.items()
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Data_v30 标签密度与过分割审计</title>
<style>
 body {{ font-family: "Microsoft YaHei", "Segoe UI", sans-serif; margin: 28px; color: #1b2430; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 28px; }}
 h3 {{ font-size: 15px; margin-top: 20px; }}
 table {{ border-collapse: collapse; margin: 10px 0 18px; font-size: 13px; }}
 th, td {{ border: 1px solid #d7dee6; padding: 5px 9px; text-align: left; }}
 th {{ background: #f2f5f8; }}
 .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 .bar {{ display: inline-block; width: 90px; height: 9px; background: #eaeef2; vertical-align: middle; }}
 .bar span {{ display: inline-block; height: 9px; background: #2f80ed; }}
 .headline {{ background: #f7f9fb; border-left: 4px solid #2f80ed; padding: 12px 16px; margin: 16px 0; }}
 .note {{ color: #5b6b7c; font-size: 13px; }}
 code {{ background: #f2f5f8; padding: 1px 4px; }}
</style>
</head>
<body>
<h1>Data_v30 标签密度与过分割审计</h1>
<p class="note">生成时间 {html.escape(summary['created_at'])} ｜ 数据集 <code>{html.escape(summary['dataset_root'])}</code>
 ｜ 审计划分 {html.escape(', '.join(summary['splits']))} ｜
 held_out_test_used = {str(summary['held_out_test_used']).lower()} ｜
 sealed_blind_used = {str(summary['sealed_blind_used']).lower()}</p>
<div class="headline">{headline}</div>

<h2>一、判定依据（预注册指标）</h2>
<table><thead><tr><th>指标</th><th>是否命中</th></tr></thead><tbody>{indicator_html}</tbody></table>
<table><thead><tr><th>指纹量</th><th>SAM2 批</th><th>人工基础批</th></tr></thead><tbody>{fingerprint_html}</tbody></table>
<p class="note">轮廓数中位比 = {ratio_text}；
判定阈值：重叠 IoU &gt; {summary['decision']['thresholds']['overlap_iou']}、
膨胀 {summary['decision']['thresholds']['dilate_px']} px、接触面积占比 &gt;
{summary['decision']['thresholds']['touch_fraction']}、至少
{summary['decision']['thresholds']['touch_neighbours']} 个邻居、
指标倍率 ≥ {summary['decision']['thresholds']['indicator_ratio']}、
轮廓数比 ≥ {summary['decision']['thresholds']['polygon_ratio_min']}。</p>

<h2>二、分批统计</h2>
{''.join(split_sections)}

<h2>三、与冻结生产召回的关联（仅 valid）</h2>
{linkage_html}

<h2>四、轮廓数最高的图像</h2>
<table><thead><tr><th>图像</th><th>划分</th><th>来源</th><th>轮廓数</th><th>中位等效半径</th>
<th>重叠占比</th><th>拼接占比</th></tr></thead><tbody>{densest_html}</tbody></table>

<p class="note">本审计只读 <code>datasets/Data_v3/{{train,valid}}</code>；
<code>datasets/Data_v3/test</code> 与 <code>datasets/Data_release_blind_v1</code> 未被读取。
标签集哈希：{html.escape(json.dumps(summary['inputs']['label_set_sha256'], ensure_ascii=False))}</p>
</body>
</html>
"""


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    columns = [
        "split", "stem", "source", "polygons", "overlap_pairs", "overlap_polygons",
        "tiling_polygons", "overlap_share", "tiling_share", "union_coverage",
        "union_over_sum", "median_radius", "median_area", "max_iou", "malformed",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_summary(
    dataset_root: Path,
    splits: Sequence[str],
    sam2_stems: set[str],
    sam2_dir: Path,
    metrics_path: Path,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for split in splits:
        rows.extend(audit_source(split, dataset_root, sam2_stems))

    per_split: dict[str, object] = {}
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        batches = {
            source: summarize_batch([row for row in split_rows if row["source"] == source])
            for source in ("base", "sam2_new")
            if any(row["source"] == source for row in split_rows)
        }
        decision = (
            decide(batches["sam2_new"], batches["base"])
            if {"base", "sam2_new"} <= set(batches)
            else inconclusive_decision()
        )
        per_split[split] = {"batches": batches, "decision": decision}

    pooled = {
        source: summarize_batch([row for row in rows if row["source"] == source])
        for source in ("base", "sam2_new")
        if any(row["source"] == source for row in rows)
    }
    overall = (
        decide(pooled["sam2_new"], pooled["base"])
        if {"base", "sam2_new"} <= set(pooled)
        else inconclusive_decision()
    )

    densest = sorted(rows, key=lambda row: (-float(row["polygons"]), str(row["stem"])))[:10]
    summary = {
        "version": VERSION,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "splits": list(splits),
        "held_out_test_used": False,
        "sealed_blind_used": False,
        "policy": "train and valid only; the frozen test split and the release blind holdout are never opened",
        "inputs": {
            "sam2_label_dir": str(sam2_dir),
            "sam2_stems": len(sam2_stems),
            "label_set_sha256": {
                split: sha256_labels(label_paths(dataset_root, split)) for split in splits
            },
        },
        "grid": {"width": GRID_WIDTH, "height": GRID_HEIGHT},
        "per_split": per_split,
        "pooled_batches": pooled,
        "decision": overall,
        "recall_linkage": recall_linkage([row for row in rows if row["split"] == "valid"], metrics_path),
        "densest_images": densest,
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--splits", nargs="+", choices=AUDITED_SPLITS, default=list(AUDITED_SPLITS),
        help="Only train and valid are accepted; the frozen test split cannot be requested.",
    )
    parser.add_argument("--sam2-labels", type=Path, default=DEFAULT_SAM2_LABELS)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset does not exist: {dataset_root}")
    sam2_dir = args.sam2_labels.resolve()
    sam2_stems = {path.stem for path in sam2_dir.glob("*.txt")} if sam2_dir.is_dir() else set()

    summary, rows = build_summary(
        dataset_root, tuple(args.splits), sam2_stems, sam2_dir, args.metrics_json.resolve()
    )

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(rows, output_dir / "per_image_density.csv")
    (output_dir / "index.html").write_text(render_html(summary), encoding="utf-8")

    report = {
        "decision": summary["decision"]["decision"],
        "indicators": summary["decision"]["indicators"],
        "polygon_ratio": summary["decision"]["polygon_ratio"],
        "images": len(rows),
        "output": str(output_dir),
        "held_out_test_used": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())