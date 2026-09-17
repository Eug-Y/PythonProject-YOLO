"""Build the project progress web report with annotated production-effect figures.

The report renders the *frozen production* V27 fusion (YOLO11-primary +
YOLOv8 auxiliary) on the validation split only, using the prediction bundles
exported at original resolution.  The sealed test split and the one-shot release
blind holdout are never read.

Outputs
-------
reports/Project_progress_report/index.html
reports/Project_progress_report/summary.json
reports/Project_progress_report/per_image_metrics.json
reports/Project_progress_report/assets/*.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibrate_yolo_ensemble import cached_bbox, load_yolo_ground_truth, match_instances
from multispectral import PROJECT_ROOT, read_image
from prediction_bundle import load_prediction_bundle
from yolo_mask_ensemble import MaskPrediction, cropped_mask_iou, fuse_mask_predictions

DEPLOYMENT = PROJECT_ROOT / "configs" / "deployment.json"
DATASET = PROJECT_ROOT / "datasets" / "Data_v3"
VALID_IMAGES = DATASET / "valid" / "images"
VALID_LABELS = DATASET / "valid" / "labels"
PRIMARY_BUNDLES = PROJECT_ROOT / "reports" / "Data_v23_yolo11_fusion_valid" / "predictions_yolo11_primary"
AUXILIARY_BUNDLES = PROJECT_ROOT / "reports" / "Data_v23_yolo11_fusion_valid" / "predictions_yolov8_auxiliary"
BASELINE_SUMMARY = PROJECT_ROOT / "logs" / "v29_yolo11_recall_finetune.summary.json"
BLIND_SUMMARY = PROJECT_ROOT / "logs" / "v27_release_blind_final.summary.json"
RUNTIME_SUMMARY = PROJECT_ROOT / "logs" / "v28_runtime_optimization.summary.json"
HARD_EXAMPLES_SUMMARY = PROJECT_ROOT / "logs" / "v28_yolo11_hard_examples.summary.json"
RECALL_SUMMARY = PROJECT_ROOT / "logs" / "v29_yolo11_recall_finetune.summary.json"
V19_REPORT = PROJECT_ROOT / "reports" / "Data_v19_final_report" / "README.md"
CORRECTION_SCREENSHOT = PROJECT_ROOT / "reports" / "difficult_sample_correction_quality_v2.png"
DEPLOYED_HOME_SCREENSHOT = PROJECT_ROOT / "reports" / "v27_deployed_web_home.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "Project_progress_report"

PANEL_WIDTH = 800
PANEL_HEIGHT = 650
GAP = 8
HEADER_HEIGHT = 96
CROP_WIDTH = 400
CROP_HEIGHT = 325
CROP_SCALE = 2

# BGR colours for the OpenCV layers.
GREEN = (60, 190, 60)
RED = (50, 50, 235)
ORANGE = (0, 165, 255)
CYAN = (230, 220, 40)
MAGENTA = (200, 80, 215)
# RGB colours for the PIL text pass.
RGB_DARK = (24, 30, 36)
RGB_WHITE = (248, 250, 252)
RGB_GREEN = (52, 199, 89)
RGB_RED = (235, 87, 87)
RGB_ORANGE = (255, 149, 0)
RGB_CYAN = (64, 224, 230)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int):
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    for candidate in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "DejaVuSans.ttf"):
        try:
            _FONT_CACHE[size] = ImageFont.truetype(candidate, size)
            return _FONT_CACHE[size]
        except Exception:  # noqa: BLE001 - font availability is environment-specific
            continue
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_texts(bgr: np.ndarray, items: list[tuple[int, int, str, int, tuple, tuple | None]]) -> np.ndarray:
    """Draw CJK-capable text on a BGR image and return the new image."""
    if not items:
        return bgr
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil)
    for x, y, text, size, color, background in items:
        font = _font(size)
        if background is not None:
            left, top, right, bottom = drawer.textbbox((x, y), text, font=font)
            drawer.rectangle((left - 7, top - 5, right + 7, bottom + 5), fill=background)
        drawer.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def blend(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    """Alpha-blend a flat colour into the masked pixels only."""
    if not mask.any():
        return img
    out = img.astype(np.float32)
    target = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + target * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def outline(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return img


def dim(img: np.ndarray, factor: float = 0.55) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def fit_panel(img: np.ndarray, mask: np.ndarray | None = None):
    """Resize an image to the panel size and a mask with it."""
    resized = cv2.resize(img, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    if mask is None:
        return resized, None
    small = cv2.resize(mask.astype(np.uint8), (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return resized, small.astype(bool)


def match_detail(
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
    threshold: float,
    prediction_bboxes: list[tuple[int, int, int, int]],
):
    """Greedy matcher that returns the assignment, mirroring match_instances."""
    target_bboxes = [cached_bbox(target) for target in ground_truth]
    matched: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for prediction_index, (prediction, prediction_bbox) in enumerate(zip(predictions, prediction_bboxes)):
        best_index = None
        best_iou = threshold
        for index, (target, target_bbox) in enumerate(zip(ground_truth, target_bboxes)):
            if index in matched:
                continue
            overlap = cropped_mask_iou(prediction, target, prediction_bbox, target_bbox)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None:
            matched.add(best_index)
            pairs.append((prediction_index, best_index, best_iou))
    return pairs, matched


def load_production() -> dict:
    payload = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    if payload.get("state") != "active":
        raise RuntimeError("Production deployment config is not active")
    if payload.get("every_output_requires_yolo11") is not True:
        raise RuntimeError("Production config does not guarantee YOLO11 ownership")
    if payload.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
        raise RuntimeError("Unexpected production feature contract")
    fusion_path = PROJECT_ROOT / payload["fusion_config"]
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    if fusion.get("role") != "strict_yolo11_primary":
        raise RuntimeError("Production fusion must use the strict YOLO11-primary role")
    if fusion.get("required_model_indices") != [0] or fusion.get("model_weights") != [1.0, 1.0]:
        raise RuntimeError("Production fusion must be 1:1 with a mandatory YOLO11 source")
    if fusion.get("score_threshold") != payload["confidence"]:
        raise RuntimeError("Fusion score threshold and deployment confidence disagree")
    for role in ("primary", "auxiliary"):
        checkpoint = PROJECT_ROOT / payload[role]["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual = sha256(checkpoint)
        if actual != payload[role]["sha256"]:
            raise RuntimeError(f"{role} checkpoint hash drifted: {actual}")
    for root in (PRIMARY_BUNDLES, AUXILIARY_BUNDLES):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("held_out_test_used") is not False:
            raise RuntimeError(f"Prediction export {root} does not declare a sealed test split")
        if manifest.get("split") != "valid":
            raise RuntimeError(f"Prediction export {root} is not a validation export")
        if manifest.get("channels") != ["R", "G", "B", "NDVI", "NDRE"]:
            raise RuntimeError(f"Prediction export {root} has an unexpected channel contract")
    return payload, fusion


def predict_for_image(stem: str, image_hw: tuple[int, int], names: tuple[str, str], config: dict):
    per_model: list[list[MaskPrediction]] = []
    for name, root in zip(names, (PRIMARY_BUNDLES, AUXILIARY_BUNDLES)):
        bundle = load_prediction_bundle(root / "predictions" / f"{stem}.npz")
        if tuple(bundle["image_hw"]) != image_hw:
            raise ValueError(f"Prediction/image shape mismatch for {name}/{stem}")
        per_model.append([
            MaskPrediction(mask, float(score), int(class_id), name)
            for mask, score, class_id in zip(bundle["masks"], bundle["scores"], bundle["class_ids"])
        ])
    fused = fuse_mask_predictions(
        per_model,
        model_weights=tuple(config["model_weights"]),
        match_iou=float(config["match_iou"]),
        mask_vote=float(config["mask_vote"]),
        score_threshold=float(config["score_threshold"]),
        min_models=int(config["min_models"]),
        required_model_indices=tuple(config.get("required_model_indices", ())),
        unconfirmed_score_threshold=config.get("unconfirmed_score_threshold"),
    )
    if any("yolo11_primary" not in item.sources for item in fused):
        raise RuntimeError(f"YOLO11 ownership invariant failed for {stem}")
    threshold = float(config["score_threshold"])
    return {
        "fusion": ([item.mask for item in fused], [item.bbox for item in fused]),
        "primary_yolo11": (
            [item.mask for item in per_model[0] if item.score >= threshold],
            [item.bbox for item in per_model[0] if item.score >= threshold],
        ),
        "auxiliary_yolov8": (
            [item.mask for item in per_model[1] if item.score >= threshold],
            [item.bbox for item in per_model[1] if item.score >= threshold],
        ),
    }


def evaluate_valid(config: dict, names: tuple[str, str]) -> tuple[list[dict], dict]:
    images = sorted(VALID_IMAGES.glob("*.JPG"))
    if not images:
        raise FileNotFoundError(f"No validation images in {VALID_IMAGES}")
    counters = {
        key: {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        for key in ("fusion", "primary_yolo11", "auxiliary_yolov8")
    }
    per_image: list[dict] = []
    for index, image_path in enumerate(images, 1):
        image_hw = tuple(read_image(image_path).shape[:2])
        targets = load_yolo_ground_truth(VALID_LABELS / f"{image_path.stem}.txt", image_hw)
        variants = predict_for_image(image_path.stem, image_hw, names, config)
        row = {"image": image_path.name, "stem": image_path.stem, "ground_truth": len(targets)}
        for key, (masks, bboxes) in variants.items():
            matched = match_instances(masks, targets, prediction_bboxes=bboxes)
            counters[key]["tp"] += int(matched["tp"])
            counters[key]["fp"] += int(matched["fp"])
            counters[key]["fn"] += int(matched["fn"])
            counters[key]["matched_ious"].extend(matched["matched_ious"])
            tp, fp, fn = int(matched["tp"]), int(matched["fp"]), int(matched["fn"])
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            row[key] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "predictions": len(masks),
            }
        per_image.append(row)
        print(f"[{index:>2}/{len(images)}] {image_path.name} "
              f"GT={row['ground_truth']:>3} fused={row['fusion']['predictions']:>3} "
              f"TP={row['fusion']['tp']:>3} FP={row['fusion']['fp']:>3} FN={row['fusion']['fn']:>3} "
              f"F1={row['fusion']['f1']:.4f}", flush=True)
    aggregate = {}
    for key, value in counters.items():
        tp, fp, fn = int(value["tp"]), int(value["fp"]), int(value["fn"])
        ious = value["matched_ious"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        aggregate[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_matched_iou": float(np.mean(ious)) if ious else 0.0,
        }
    return per_image, aggregate


def select_showcase(per_image: list[dict], limit: int = 5) -> list[dict]:
    """Pick illustrative images.  Images with zero ground truth are excluded:
    their F1 is 0 by convention (every prediction counts as a false positive),
    which would otherwise dominate the "worst" slot with an empty truth panel."""
    scored = [row for row in per_image if row["ground_truth"] > 0] or list(per_image)
    ordered = sorted(scored, key=lambda row: row["fusion"]["f1"])
    picks: list[dict] = []
    seen: set[str] = set()

    def take(row: dict | None) -> None:
        if row is not None and row["stem"] not in seen and len(picks) < limit:
            seen.add(row["stem"])
            picks.append(row)

    take(ordered[0])                                                    # worst F1
    take(ordered[-1])                                                   # best F1
    take(max(scored, key=lambda r: r["fusion"]["fp"]))                  # most false positives
    take(max(scored, key=lambda r: r["fusion"]["fn"]))                  # most misses
    take(ordered[len(ordered) // 2])                                    # median
    for row in ordered:                                                 # deterministic top-up
        if len(picks) >= limit:
            break
        take(row)
    return picks


def zero_ground_truth(per_image: list[dict]) -> list[dict]:
    return [row for row in per_image if row["ground_truth"] == 0]


def error_focus(pairs, ground_truth, predictions, image_hw, matched) -> tuple[int, int, int, int]:
    """Find a native-resolution crop around the most serious error."""
    missed = [index for index in range(len(ground_truth)) if index not in matched]
    false_positives = [index for index in range(len(predictions)) if index not in {p for p, _, _ in pairs}]
    candidates: list[np.ndarray] = []
    for index in missed:
        candidates.append(ground_truth[index])
    for index in false_positives:
        candidates.append(predictions[index])
    if not candidates:
        return 0, 0, CROP_WIDTH, CROP_HEIGHT
    height, width = image_hw
    best = max(candidates, key=lambda mask: int(mask.sum()))
    rows, columns = np.nonzero(best)
    centre_x = int(columns.mean())
    centre_y = int(rows.mean())
    crop_w = min(CROP_WIDTH, width)
    crop_h = min(CROP_HEIGHT, height)
    left = int(np.clip(centre_x - crop_w // 2, 0, width - crop_w))
    top = int(np.clip(centre_y - crop_h // 2, 0, height - crop_h))
    return left, top, crop_w, crop_h


def render_figure(row: dict, config: dict, names: tuple[str, str], assets: Path) -> str:
    image_path = VALID_IMAGES / row["image"]
    image = read_image(image_path)
    image_hw = tuple(image.shape[:2])
    targets = load_yolo_ground_truth(VALID_LABELS / f"{row['stem']}.txt", image_hw)
    variants = predict_for_image(row["stem"], image_hw, names, config)
    masks, bboxes = variants["fusion"]
    pairs, matched = match_detail(masks, targets, 0.5, bboxes)
    matched_predictions = {prediction for prediction, _, _ in pairs}

    tp_mask = np.zeros(image_hw, dtype=bool)
    for prediction_index, target_index, _ in pairs:
        tp_mask |= masks[prediction_index]
    fp_mask = np.zeros(image_hw, dtype=bool)
    for index, mask in enumerate(masks):
        if index not in matched_predictions:
            fp_mask |= mask
    fn_mask = np.zeros(image_hw, dtype=bool)
    for index, target in enumerate(targets):
        if index not in matched:
            fn_mask |= target
    gt_mask = np.zeros(image_hw, dtype=bool)
    for target in targets:
        gt_mask |= target

    # Panel 1: human ground truth.
    truth_panel, truth_small = fit_panel(image, gt_mask)
    truth_panel = outline(truth_panel, truth_small, CYAN, 2)

    # Panel 2: production fusion output.
    pred_panel, _ = fit_panel(image)
    _, tp_small = fit_panel(image, tp_mask)
    _, fp_small = fit_panel(image, fp_mask)
    _, gt_small = fit_panel(image, gt_mask)
    pred_panel = outline(pred_panel, gt_small, (170, 170, 170), 1)
    pred_panel = blend(pred_panel, tp_small, GREEN, 0.45)
    pred_panel = blend(pred_panel, fp_small, RED, 0.45)
    pred_panel = outline(pred_panel, tp_small, (20, 120, 20), 1)
    pred_panel = outline(pred_panel, fp_small, (30, 30, 150), 1)

    # Panel 3: error map.
    error_panel, _ = fit_panel(dim(image, 0.5))
    _, fn_small = fit_panel(image, fn_mask)
    error_panel = blend(error_panel, tp_small, GREEN, 0.55)
    error_panel = blend(error_panel, fp_small, RED, 0.55)
    error_panel = blend(error_panel, fn_small, ORANGE, 0.60)

    # Panel 4: native-resolution zoom on the most serious error.
    if targets or masks:
        left, top, crop_w, crop_h = error_focus(pairs, targets, masks, image_hw, matched)
    else:
        left, top, crop_w, crop_h = 0, 0, min(CROP_WIDTH, image_hw[1]), min(CROP_HEIGHT, image_hw[0])
    crop = image[top:top + crop_h, left:left + crop_w].copy()
    crop_tp = tp_mask[top:top + crop_h, left:left + crop_w]
    crop_fp = fp_mask[top:top + crop_h, left:left + crop_w]
    crop_gt = gt_mask[top:top + crop_h, left:left + crop_w]
    crop = outline(crop, crop_gt, CYAN, 1)
    crop = blend(crop, crop_tp, GREEN, 0.45)
    crop = blend(crop, crop_fp, RED, 0.45)
    crop = cv2.resize(crop, (crop_w * CROP_SCALE, crop_h * CROP_SCALE), interpolation=cv2.INTER_NEAREST)
    if crop.shape[1] != PANEL_WIDTH or crop.shape[0] != PANEL_HEIGHT:
        canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 18, dtype=np.uint8)
        height = min(PANEL_HEIGHT, crop.shape[0])
        width = min(PANEL_WIDTH, crop.shape[1])
        canvas[:height, :width] = crop[:height, :width]
        crop = canvas

    total_width = PANEL_WIDTH * 2 + GAP * 3
    figure = np.full((HEADER_HEIGHT + PANEL_HEIGHT * 2 + GAP * 3, total_width, 3), 245, dtype=np.uint8)
    cells = [
        (0, 0, truth_panel),
        (1, 0, pred_panel),
        (0, 1, error_panel),
        (1, 1, crop),
    ]
    for column, row_index, cell in cells:
        y = HEADER_HEIGHT + GAP + row_index * (PANEL_HEIGHT + GAP)
        x = GAP + column * (PANEL_WIDTH + GAP)
        figure[y:y + PANEL_HEIGHT, x:x + PANEL_WIDTH] = cell

    metrics = row["fusion"]
    texts = [
        (GAP + 4, 10, f"{row['image']}  ·  生产融合 V27（YOLO11 主 + YOLOv8 辅）", 26, RGB_DARK, None),
        (
            GAP + 4,
            48,
            f"有效集单图：F1 {metrics['f1']:.4f}   Precision {metrics['precision']:.4f}   "
            f"Recall {metrics['recall']:.4f}   TP {metrics['tp']}   FP {metrics['fp']}   FN {metrics['fn']}",
            20,
            (70, 82, 96),
            None,
        ),
        (GAP + 12, HEADER_HEIGHT + GAP + 10, "① 人工真值 GT（青色轮廓）", 22, RGB_WHITE, (0, 0, 0)),
        (GAP * 2 + PANEL_WIDTH + 12, HEADER_HEIGHT + GAP + 10, "② 生产融合预测（绿=命中 TP，红=误检 FP，灰=真值参考）", 20, RGB_WHITE, (0, 0, 0)),
        (GAP + 12, HEADER_HEIGHT + GAP * 2 + PANEL_HEIGHT + 10, "③ 误差分布（绿=TP，红=FP 误检，橙=FN 漏检）", 22, RGB_WHITE, (0, 0, 0)),
        (GAP * 2 + PANEL_WIDTH + 12, HEADER_HEIGHT + GAP * 2 + PANEL_HEIGHT + 10, "④ 局部放大（2×，最严重错误区域，青=真值，绿=TP，红=FP）", 20, RGB_WHITE, (0, 0, 0)),
    ]
    figure = draw_texts(figure, texts)
    assets.mkdir(parents=True, exist_ok=True)
    output_path = assets / f"{row['stem']}_effect.jpg"
    ok, encoded = cv2.imencode(".jpg", figure, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"Cannot encode {output_path}")
    encoded.tofile(output_path)
    return output_path.name


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_html(
    output: Path,
    production: dict,
    fusion: dict,
    aggregate: dict,
    verification: dict,
    showcase: list[dict],
    figures: dict[str, str],
    extras: dict[str, str],
    names: tuple[str, str],
    zero_gt: list[dict] | None = None,
) -> str:
    blind = _read_json(BLIND_SUMMARY, {}) or {}
    runtime = _read_json(RUNTIME_SUMMARY, {}) or {}
    hard = _read_json(HARD_EXAMPLES_SUMMARY, {}) or {}
    recall = _read_json(RECALL_SUMMARY, {}) or {}

    def fmt(value, digits: int = 4) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    def delta(candidate: dict, baseline: dict, digits: int = 4) -> str:
        value = float(candidate) - float(baseline)
        return f"{value:+.{digits}f}"

    cards = []
    for row in showcase:
        metrics = row["fusion"]
        cards.append(f'''    <article class="sample">
      <header>
        <h3>{html.escape(row['image'])}</h3>
        <p class="mini">真值 <b>{row['ground_truth']}</b> 个实例 · 融合输出 <b>{metrics['predictions']}</b> 个 ·
          命中 <b class="good">{metrics['tp']}</b> · 误检 <b class="bad">{metrics['fp']}</b> · 漏检 <b class="warn">{metrics['fn']}</b> ·
          F1 <b>{fmt(metrics['f1'])}</b> · P <b>{fmt(metrics['precision'])}</b> · R <b>{fmt(metrics['recall'])}</b></p>
      </header>
      <img src="assets/{html.escape(figures[row['stem']])}" alt="{html.escape(row['image'])} 生产融合效果标注图">
    </article>''')

    blind_fusion = (blind.get("metrics") or {}).get("fusion") or {}
    blind_incumbent = (blind.get("metrics") or {}).get("incumbent_yolov8") or {}
    runtime_before = (runtime.get("before") or {}).get("median_seconds")
    runtime_after = (runtime.get("after") or {}).get("median_seconds")
    hard_metrics = (hard.get("v28_locked_valid_metrics") or {}).get("fusion") or {}
    hard_baseline = hard.get("v27_seed0_baseline_fusion") or {}
    hard_gate = hard.get("promotion_gate") or {}
    hard_progress = hard.get("training_progress") or {}
    recall_metrics = (recall.get("v29_locked_valid_metrics") or {}).get("fusion") or {}
    recall_baseline = recall.get("v27_seed0_baseline_fusion") or {}
    recall_gate = recall.get("recall_priority_gate") or {}
    recall_progress = recall.get("training_progress") or {}

    extra_blocks = []
    if "correction" in extras:
        extra_blocks.append(f'''  <section>
    <h2>标注工具效果（困难样本纠正台）</h2>
    <p class="note">纠正台新增整图质量检查：错误（紫色）阻止保存、警告（橙色）提示人工复核、轮廓编号、
      顶点级微调与完全重复实例拦截。队列 20 张验证图已全部完成纠正（状态 corrected 20/20）。</p>
    <img class="shot" src="assets/{html.escape(extras['correction'])}" alt="困难样本纠正台质量检查界面">
  </section>''')
    if "home" in extras:
        extra_blocks.append(f'''  <section>
    <h2>生产站点效果（V27 部署页）</h2>
    <p class="note">生产服务使用 waitress 启动于 <code>http://127.0.0.1:5055</code>；
      页面提供原始影像、NDVI/NDRE/GNDVI/EVI2 指数层、土壤代理层、热力图、分割结果与杂草候选复核层。</p>
    <img class="shot" src="assets/{html.escape(extras['home'])}" alt="V27 生产部署页面截图">
  </section>''')

    checks = "".join(
        f'<tr><td>{html.escape(name)}</td><td class="{"good" if passed else "bad"}">{"通过" if passed else "未通过"}</td></tr>'
        for name, passed in verification["checks"].items()
    )

    zero_block = ""
    if zero_gt:
        items = "、".join(
            f"<code>{html.escape(row['image'])}</code>（模型输出 {row['fusion']['predictions']} 个实例，"
            f"全部只能计为误检 FP {row['fusion']['fp']}）"
            for row in zero_gt
        )
        zero_block = f'''  <div class="callout">
    <b>发现一处标注缺口：</b>有效集有 {len(zero_gt)} 张图的标签文件为空，人工真值为 0 个实例 —— {items}。
    按目标检测约定这类图像无法定义 F1（任何预测都只能记为误检），因此已排除在效果图选取之外。
    该帧可能确实没有可标注目标，也可能是漏标；建议纳入标注复核队列后再参与评测。
  </div>'''

    page = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO 五通道农田分割 · 项目进度与最新效果报告</title>
<style>
  :root {{ color-scheme: light; font-family: "Microsoft YaHei", Inter, system-ui, sans-serif; --ink:#16202b; --sub:#5b6b7c; --line:#dde5ec; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f1f4f7; color:var(--ink); line-height:1.65; }}
  main {{ width:min(1560px, calc(100% - 32px)); margin:26px auto 72px; }}
  h1 {{ font-size:30px; margin:0 0 6px; }}
  h2 {{ font-size:21px; margin:34px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--line); }}
  h3 {{ margin:0 0 6px; font-size:17px; }}
  .note {{ color:var(--sub); margin:0 0 14px; }}
  .banner {{ background:linear-gradient(135deg,#0f2233,#1d3f5c); color:#eaf3fb; border-radius:16px; padding:22px 26px; }}
  .banner h1 {{ color:#fff; }}
  .banner p {{ margin:6px 0 0; color:#bcd3e6; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:18px 0 8px; }}
  .kpi {{ background:#fff; border-radius:13px; padding:15px 17px; box-shadow:0 2px 12px #13223812; }}
  .kpi span {{ display:block; color:var(--sub); font-size:13px; }}
  .kpi strong {{ display:block; margin-top:6px; font-size:25px; letter-spacing:.3px; }}
  .kpi em {{ display:block; margin-top:3px; font-style:normal; font-size:12px; color:var(--sub); }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px #13223812; font-size:14px; }}
  th,td {{ padding:9px 12px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ background:#e9eff5; font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  .good {{ color:#1a7f37; }} .bad {{ color:#c62828; }} .warn {{ color:#b26a00; }}
  code {{ background:#eef3f7; padding:1px 5px; border-radius:4px; font-size:13px; }}
  .sample {{ background:#fff; border-radius:14px; padding:16px; margin:16px 0; box-shadow:0 2px 14px #13223812; }}
  .sample img, .shot {{ display:block; width:100%; border-radius:9px; }}
  .mini {{ color:var(--sub); margin:0 0 10px; font-size:14px; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .callout {{ background:#fff8e6; border-left:5px solid #e0a300; border-radius:10px; padding:14px 18px; }}
  ul {{ margin:8px 0; padding-left:22px; }} li {{ margin:4px 0; }}
  footer {{ margin-top:40px; color:var(--sub); font-size:13px; border-top:1px solid var(--line); padding-top:14px; }}
  @media (max-width:1000px) {{ .cols {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>

<div class="banner">
  <h1>YOLO 五通道农田分割 · 项目进度与最新效果报告</h1>
  <p>输入通道 <code>R, G, B, NDVI, NDRE</code> · 生产版本 <b>{html.escape(production['model_name'])}</b> ·
     报告生成时间 {html.escape(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z'))}</p>
  <p>数据边界：效果图与指标全部来自 <b>train/valid</b>；封存测试集与一次性发布盲测均未被读取或复用。</p>
</div>

<section>
  <h2>一、当前生产版本</h2>
  <div class="kpis">
    <div class="kpi"><span>主模型（必备来源）</span><strong>{html.escape(production['primary']['architecture'])}</strong>
      <em>sha256 {production['primary']['sha256'][:12]}…</em></div>
    <div class="kpi"><span>辅助模型（等权）</span><strong>{html.escape(production['auxiliary']['architecture'])}</strong>
      <em>sha256 {production['auxiliary']['sha256'][:12]}…</em></div>
    <div class="kpi"><span>融合权重 / 匹配 IoU</span><strong>{fusion['model_weights'][0]:.1f} : {fusion['model_weights'][1]:.1f}</strong>
      <em>mask 匹配 IoU {fusion['match_iou']} · 投票 {fusion['mask_vote']}</em></div>
    <div class="kpi"><span>推理设置</span><strong>{production['imgsz']} px</strong>
      <em>置信度 {fusion['score_threshold']} · 未确认 {fusion['unconfirmed_score_threshold']} · min_models {fusion['min_models']}</em></div>
  </div>
  <p class="note">每个融合输出都必须包含 YOLO11 来源（<code>every_output_requires_yolo11 = true</code>）；
    回滚配置 <code>configs/deployment.pre_v27.json</code> 可用，回滚检查点与哈希均已验证。</p>
</section>

<section>
  <h2>二、最新效果指标</h2>
  <div class="kpis">
    <div class="kpi"><span>有效集 Mask F1（60 图，生产融合）</span><strong>{fmt(aggregate['fusion']['f1'])}</strong>
      <em>P {fmt(aggregate['fusion']['precision'])} · R {fmt(aggregate['fusion']['recall'])} · 匹配 IoU {fmt(aggregate['fusion']['mean_matched_iou'])}</em></div>
    <div class="kpi"><span>发布盲测 Mask F1（30 图，一次性）</span><strong>{fmt(blind_fusion.get('f1'))}</strong>
      <em>对照旧版 YOLOv8 {fmt(blind_incumbent.get('f1'))} · 提升 {fmt(blind.get('values', {}).get('fusion_f1_delta_vs_incumbent'))}</em></div>
    <div class="kpi"><span>V19 有效集 Mask mAP50-95</span><strong>0.3960</strong>
      <em>三种子平均较 V15 提升 +1.665 pp，最差种子 +0.947 pp</em></div>
    <div class="kpi"><span>单图端到端耗时（all 模式）</span><strong>{fmt(runtime_after, 2)} s</strong>
      <em>优化前 {fmt(runtime_before, 2)} s，提速 {runtime.get('improvement', {}).get('median_percent_faster', 0):.2f}%，输出逐像素等价</em></div>
  </div>
  <table>
    <thead><tr><th>视角</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>匹配 IoU</th></tr></thead>
    <tbody>
      <tr><td>生产融合（YOLO11 + YOLOv8，1:1）</td><td>{aggregate['fusion']['tp']}</td><td>{aggregate['fusion']['fp']}</td><td>{aggregate['fusion']['fn']}</td>
        <td>{fmt(aggregate['fusion']['precision'])}</td><td>{fmt(aggregate['fusion']['recall'])}</td><td><b>{fmt(aggregate['fusion']['f1'])}</b></td><td>{fmt(aggregate['fusion']['mean_matched_iou'])}</td></tr>
      <tr><td>仅主模型 YOLO11（生产检查点）</td><td>{aggregate['primary_yolo11']['tp']}</td><td>{aggregate['primary_yolo11']['fp']}</td><td>{aggregate['primary_yolo11']['fn']}</td>
        <td>{fmt(aggregate['primary_yolo11']['precision'])}</td><td>{fmt(aggregate['primary_yolo11']['recall'])}</td><td>{fmt(aggregate['primary_yolo11']['f1'])}</td><td>{fmt(aggregate['primary_yolo11']['mean_matched_iou'])}</td></tr>
      <tr><td>仅辅助模型 YOLOv8（等权对照）</td><td>{aggregate['auxiliary_yolov8']['tp']}</td><td>{aggregate['auxiliary_yolov8']['fp']}</td><td>{aggregate['auxiliary_yolov8']['fn']}</td>
        <td>{fmt(aggregate['auxiliary_yolov8']['precision'])}</td><td>{fmt(aggregate['auxiliary_yolov8']['recall'])}</td><td>{fmt(aggregate['auxiliary_yolov8']['f1'])}</td><td>{fmt(aggregate['auxiliary_yolov8']['mean_matched_iou'])}</td></tr>
    </tbody>
  </table>
  <p class="note">盲测 F1 明显低于有效集，说明有效集存在乐观偏差：盲测精确率仅 {fmt(blind_fusion.get('precision'))}，
    误检偏多、漏检较少（召回 {fmt(blind_fusion.get('recall'))}）。这是当前模型最大的现实短板。</p>
</section>

<section>
  <h2>三、报告可信度校验（与既有记录逐项对齐）</h2>
  <p class="note">本报告不是重新训练，而是用冻结的生产检查点与锁定融合配置在原始分辨率上重放验证集，
    并与仓库中已记录的 V27 基线逐项比对：</p>
  <table>
    <thead><tr><th>指标</th><th>本报告重放</th><th>仓库记录</th><th>差值</th></tr></thead>
    <tbody>
      <tr><td>融合 F1</td><td>{fmt(aggregate['fusion']['f1'])}</td><td>{fmt(verification['recorded']['f1'])}</td><td>{delta(aggregate['fusion']['f1'], verification['recorded']['f1'])}</td></tr>
      <tr><td>融合 Precision</td><td>{fmt(aggregate['fusion']['precision'])}</td><td>{fmt(verification['recorded']['precision'])}</td><td>{delta(aggregate['fusion']['precision'], verification['recorded']['precision'])}</td></tr>
      <tr><td>融合 Recall</td><td>{fmt(aggregate['fusion']['recall'])}</td><td>{fmt(verification['recorded']['recall'])}</td><td>{delta(aggregate['fusion']['recall'], verification['recorded']['recall'])}</td></tr>
      <tr><td>融合匹配 IoU</td><td>{fmt(aggregate['fusion']['mean_matched_iou'])}</td><td>{fmt(verification['recorded']['mean_matched_iou'])}</td><td>{delta(aggregate['fusion']['mean_matched_iou'], verification['recorded']['mean_matched_iou'])}</td></tr>
      <tr><td>TP / FP / FN</td><td>{aggregate['fusion']['tp']} / {aggregate['fusion']['fp']} / {aggregate['fusion']['fn']}</td>
        <td>{verification['recorded']['tp']} / {verification['recorded']['fp']} / {verification['recorded']['fn']}</td>
        <td>{'完全一致' if verification['exact'] else '存在差异'}</td></tr>
    </tbody>
  </table>
  <h3 style="margin-top:16px">重放前置校验</h3>
  <table><tbody>{checks}</tbody></table>
</section>

<section>
  <h2>四、效果标注图（生产模型真实输出）</h2>
  <p class="note">每张图四格：① 人工真值轮廓；② 生产融合预测（命中/误检着色）；③ 误差分布（TP/FP/FN）；
    ④ 最严重错误区域 2× 局部放大。选取规则：F1 最差图、F1 最好图、误检最多图、漏检最多图、中位图
    （已排除无标注背景图，原因见下）。</p>
{zero_block}
{chr(10).join(cards)}
</section>

{chr(10).join(extra_blocks)}

<section>
  <h2>五、进度主线（V1 → V29）</h2>
  <table>
    <thead><tr><th>阶段</th><th>目标</th><th>结论</th><th>证据</th></tr></thead>
    <tbody>
      <tr><td>V1–V3</td><td>五通道数据构建、标签质量审计与几何复核</td><td>完成；建立只读审计与纠正流程，测试集封存</td><td><code>reports/Data_v3_label_quality</code></td></tr>
      <tr><td>V4–V5</td><td>光谱特征因子筛选（17 通道候选、紧凑筛选）</td><td>候选 <code>RGB+RE_BSI_PROXY</code> 与 <code>RGB+MCARI</code> 均未超过部署模型，归档为研究结论</td><td><code>reports/Data_v5_final_summary.md</code></td></tr>
      <tr><td>V6–V8</td><td>地形因子、教师精炼标签、多光谱双分支融合</td><td>地形训练受限于真实 DEM/DSM；教师精炼与融合筛选形成后续数据基础</td><td><code>reports/Data_v8_yolo11_paired_multiseed</code></td></tr>
      <tr><td>V9–V18</td><td>分组交叉验证、边界掩膜、标准化、阈值稳健性</td><td>确立“按航段分组、训练重叠掩膜关闭”的评测协议</td><td><code>reports/Data_v13_overlap_precision_analysis</code></td></tr>
      <tr><td>V19（当前主模型）</td><td>YOLO11n 无重叠边界多种子训练</td><td>通过预注册门槛并冻结：seed-0 Mask mAP50-95 <b>0.39596</b>，sha256 <code>eea619ab…</code></td><td><code>reports/Data_v19_final_report/README.md</code></td></tr>
      <tr><td>V22–V27</td><td>融合矩阵、全分辨率融合、召回恢复与生产切换</td><td>锁定 1:1 严格 YOLO11 主融合；盲测 F1 <b>{fmt(blind_fusion.get('f1'))}</b> 优于旧版 {fmt(blind_incumbent.get('f1'))}，正式上线</td><td><code>logs/v27_production_promotion.summary.json</code></td></tr>
      <tr><td>V28（已拒绝）</td><td>训练集难例挖掘：4 个困难负样本裁片 + 8 个困难正样本重放</td><td>60 epoch，最佳 epoch {hard_progress.get('best_epoch')}；融合 F1 {fmt(hard_metrics.get('f1'))}
        （{delta(hard_metrics.get('f1', 0), hard_baseline.get('f1', 0))}），召回 {delta(hard_metrics.get('recall', 0), hard_baseline.get('recall', 0))} → 未过门槛</td>
        <td><code>logs/v28_yolo11_hard_examples.summary.json</code></td></tr>
      <tr><td>V29（已拒绝）</td><td>召回优先：V19 冻结权重低学习率微调 + 训练集假阴性加权采样（不加负样本）</td>
        <td>12 epoch，最佳 epoch {recall_progress.get('best_epoch')}；融合 F1 {fmt(recall_metrics.get('f1'))}
        （{delta(recall_metrics.get('f1', 0), recall_baseline.get('f1', 0))}），召回 {delta(recall_metrics.get('recall', 0), recall_baseline.get('recall', 0))} → 未过门槛</td>
        <td><code>logs/v29_yolo11_recall_finetune.summary.json</code></td></tr>
      <tr><td>当前进行中</td><td>困难样本纠正台质量检查升级（未提交）</td><td>整图质量检查、轮廓编号、顶点微调、重复实例拦截；单元测试 6/6、Playwright 冒烟通过</td><td><code>difficult_sample_correction_app.py</code></td></tr>
    </tbody>
  </table>
</section>

<section>
  <h2>六、两条失败路线的关键数字</h2>
  <div class="cols">
    <div>
      <h3>V28 · 难例挖掘</h3>
      <table><tbody>
        <tr><th>训练轮次 / 最佳 epoch</th><td>{hard_progress.get('epochs_recorded')} / {hard_progress.get('best_epoch')}</td></tr>
        <tr><th>最佳 Mask mAP50-95</th><td>{fmt(hard_progress.get('best_mask_map50_95'))}</td></tr>
        <tr><th>融合 F1（有效集）</th><td>{fmt(hard_metrics.get('f1'))}（{delta(hard_metrics.get('f1', 0), hard_baseline.get('f1', 0))}）</td></tr>
        <tr><th>精确率 / 召回变化</th><td class="good">{delta(hard_metrics.get('precision', 0), hard_baseline.get('precision', 0))}</td></tr>
        <tr><th>召回变化</th><td class="bad">{delta(hard_metrics.get('recall', 0), hard_baseline.get('recall', 0))}</td></tr>
        <tr><th>门槛</th><td>F1 ≥ +0.015、精确率不降、召回 ≥ -0.01 → <b class="bad">未通过</b></td></tr>
      </tbody></table>
      <p class="note">结论：抑制误检有效（精确率 +0.0395），但代价是召回大幅下降（-0.0690），净 F1 变差。</p>
    </div>
    <div>
      <h3>V29 · 召回优先微调</h3>
      <table><tbody>
        <tr><th>训练轮次 / 最佳 epoch</th><td>{recall_progress.get('epochs_recorded')} / {recall_progress.get('best_epoch')}</td></tr>
        <tr><th>最佳 Mask mAP50-95</th><td>{fmt(recall_progress.get('best_mask_map50_95'))}</td></tr>
        <tr><th>融合 F1（有效集）</th><td>{fmt(recall_metrics.get('f1'))}（{delta(recall_metrics.get('f1', 0), recall_baseline.get('f1', 0))}）</td></tr>
        <tr><th>召回变化</th><td class="bad">{delta(recall_metrics.get('recall', 0), recall_baseline.get('recall', 0))}</td></tr>
        <tr><th>匹配 IoU 变化</th><td class="good">{delta(recall_metrics.get('mean_matched_iou', 0), recall_baseline.get('mean_matched_iou', 0))}</td></tr>
        <tr><th>门槛</th><td>F1 ≥ +0.015、召回 ≥ +0.01、精确率不降 → <b class="bad">未通过</b></td></tr>
      </tbody></table>
      <p class="note">结论：最佳轮次出现在第 1 轮，之后持续退化；低学习率微调未能把召回拉回，反而整体变差。</p>
    </div>
  </div>
  <div class="callout">
    <b>两条路线的共同含义：</b>V27 生产版本保持不变（两次运行的部署配置哈希前后一致。
    V28 <code>production_deployment_unchanged = {str(hard.get('production_deployment_unchanged'))}</code>、
    V29 <code>production_deployment_unchanged = {str(recall.get('production_deployment_unchanged'))}</code>）。
    在 60 张验证图上反复调参已接近当前数据与协议的上限，下一步更需要的是<b>标签质量与难例定义</b>的改进，而不是继续在同一切分上做采样/权重变体。
  </div>
</section>

<section>
  <h2>七、当前状态与风险</h2>
  <div class="cols">
    <div>
      <h3>已具备</h3>
      <ul>
        <li>生产融合可复现重放，指标与记录完全一致，检查点与融合配置哈希前置校验通过。</li>
        <li>完整回归测试套件 262 项全部通过。</li>
        <li>Web 端 <code>all</code> 模式单图中位耗时由 {fmt(runtime_before, 2)} s 降至 {fmt(runtime_after, 2)} s，输出逐像素等价。</li>
        <li>困难样本纠正队列 20 张验证图全部完成（corrected 20/20），并新增整图质量检查。</li>
      </ul>
    </div>
    <div>
      <h3>风险与限制</h3>
      <ul>
        <li>盲测精确率仅 {fmt(blind_fusion.get('precision'))}，误检偏多；地图上“多报”比“漏报”更突出。</li>
        <li>验证集 F1（{fmt(aggregate['fusion']['f1'])}）明显高于盲测 F1（{fmt(blind_fusion.get('f1'))}），验证集不可外推为实际表现。</li>
        <li>杂草层是<b>作物行约束的复核辅助</b>，不是经验证的杂草分类器，也不能作为自动施药处方。</li>
        <li>标注重叠、稀疏低覆盖航段是已知退化切片（V19 报告中 FP 增加的主要来源）。</li>
      </ul>
    </div>
  </div>
</section>

<footer>
  数据来源：<code>datasets/Data_v3/valid</code>（60 图）+ 冻结生产检查点与锁定融合配置
  （<code>reports/Data_v23_yolo11_fusion_valid/predictions_*</code>，原始分辨率、imgsz 1024、conf 0.2）。
  盲测、有效集指标分别引用 <code>logs/v27_release_blind_final.summary.json</code>、
  <code>logs/v28_*</code>、<code>logs/v29_*</code>。封存测试集 <code>datasets/Data_v3/test</code> 与
  <code>datasets/Data_release_blind_v1</code> 未在本报告中读取。
</footer>

</main></body></html>
'''
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--showcase-limit", type=int, default=5)
    args = parser.parse_args()
    output = args.output.resolve()
    assets = output / "assets"
    if assets.exists():
        shutil.rmtree(assets)

    production, fusion = load_production()
    names = ("yolo11_primary", "yolov8_auxiliary")
    print("生产配置前置校验通过，开始重放验证集…")
    per_image, aggregate = evaluate_valid(fusion, names)

    recorded = ((_read_json(BASELINE_SUMMARY, {}) or {}).get("v27_seed0_baseline_fusion")) or {}
    if not recorded:
        raise RuntimeError(f"Recorded V27 baseline is missing in {BASELINE_SUMMARY}")
    keys = ("tp", "fp", "fn", "precision", "recall", "f1", "mean_matched_iou")
    deltas = {key: float(aggregate["fusion"][key]) - float(recorded[key]) for key in keys}
    exact = all(int(aggregate["fusion"][key]) == int(recorded[key]) for key in ("tp", "fp", "fn"))
    tolerance = 1e-6
    checks = {
        "生产配置已激活且声明 YOLO11 必备来源": production.get("every_output_requires_yolo11") is True,
        "主/辅检查点 sha256 与部署配置一致": True,
        "预测导出声明未使用测试集": True,
        "融合 TP/FP/FN 与仓库记录完全一致": exact,
        f"融合 F1 与记录差值 ≤ {tolerance:g}": abs(deltas["f1"]) <= tolerance,
        "封存测试集与发布盲测未被读取": True,
    }
    verification = {"recorded": {key: recorded[key] for key in keys}, "deltas": deltas,
                    "exact": exact, "checks": checks}
    print(f"重放融合：TP={aggregate['fusion']['tp']} FP={aggregate['fusion']['fp']} FN={aggregate['fusion']['fn']} "
          f"F1={aggregate['fusion']['f1']:.6f}（记录 {float(recorded['f1']):.6f}）")
    if not all(checks.values()):
        raise RuntimeError(f"报告前置校验失败：{checks}")

    showcase = select_showcase(per_image, limit=max(1, args.showcase_limit))
    empty_labels = zero_ground_truth(per_image)
    if empty_labels:
        print(f"注意：{len(empty_labels)} 张验证图标签为空，已从效果图选取中排除：" 
              + ", ".join(row["image"] for row in empty_labels))
    figures: dict[str, str] = {}
    for row in showcase:
        figures[row["stem"]] = render_figure(row, fusion, names, assets)
        print(f"效果图已生成：{figures[row['stem']]}")

    extras: dict[str, str] = {}
    assets.mkdir(parents=True, exist_ok=True)
    for key, source, name in (
        ("correction", CORRECTION_SCREENSHOT, "difficult_sample_correction_quality_v2.png"),
        ("home", DEPLOYED_HOME_SCREENSHOT, "v27_deployed_web_home.png"),
    ):
        if source.is_file():
            shutil.copy2(source, assets / name)
            extras[key] = name

    output.mkdir(parents=True, exist_ok=True)
    page = build_html(output, production, fusion, aggregate, verification, showcase, figures, extras, names,
                      zero_gt=empty_labels)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "per_image_metrics.json").write_text(
        json.dumps({"version": "project_progress_per_image_v1", "split": "valid",
                    "held_out_test_used": False, "images": per_image}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    summary = {
        "version": "project_progress_report_summary_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "production_model": production["model_name"],
        "split": "valid",
        "image_count": len(per_image),
        "held_out_test_used": False,
        "sealed_blind_used": False,
        "aggregate": aggregate,
        "recorded_baseline": verification["recorded"],
        "deltas": deltas,
        "exact_match": exact,
        "checks": checks,
        "showcase_images": [row["image"] for row in showcase],
        "empty_label_images": [
            {"image": row["image"], "predictions": row["fusion"]["predictions"], "fp": row["fusion"]["fp"]}
            for row in empty_labels
        ],
        "figures": figures,
        "extras": extras,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())