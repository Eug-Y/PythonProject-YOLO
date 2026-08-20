"""Calibrate two-model YOLO mask fusion on validation predictions only."""

from __future__ import annotations

import argparse
import html
import itertools
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from multispectral import PROJECT_ROOT, read_image
from prediction_bundle import load_prediction_bundle
from yolo_mask_ensemble import MaskPrediction, bbox_from_mask, cropped_mask_iou, fuse_mask_predictions, mask_iou


def load_yolo_ground_truth(label_path: Path, image_hw: tuple[int, int]) -> list[np.ndarray]:
    height, width = image_hw
    masks: list[np.ndarray] = []
    if not label_path.is_file():
        return masks
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        values = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        if len(values) < 6 or len(values) % 2:
            raise ValueError(f"Invalid polygon at {label_path}:{line_number}")
        points = values.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygon = np.rint(points).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        if mask.any():
            masks.append(mask.astype(bool))
    return masks


_BBOX_CACHE: dict[int, tuple[np.ndarray, tuple[int, int, int, int]]] = {}


def cached_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    key = id(mask)
    cached = _BBOX_CACHE.get(key)
    if cached is not None and cached[0] is mask:
        return cached[1]
    bbox = bbox_from_mask(mask)
    _BBOX_CACHE[key] = (mask, bbox)
    return bbox


def match_instances(
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
    threshold: float = 0.5,
    prediction_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> dict:
    matched: set[int] = set()
    true_positive = 0
    matched_ious: list[float] = []
    if prediction_bboxes is None:
        prediction_bboxes = [cached_bbox(mask) for mask in predictions]
    target_bboxes = [cached_bbox(mask) for mask in ground_truth]
    for prediction, prediction_bbox in zip(predictions, prediction_bboxes):
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
            true_positive += 1
            matched_ious.append(best_iou)
    return {
        "tp": true_positive,
        "fp": len(predictions) - true_positive,
        "fn": len(ground_truth) - true_positive,
        "matched_ious": matched_ious,
    }


def evaluate_fusion(
    ground_truth: dict[str, list[np.ndarray]],
    predictions: tuple[dict[str, list[MaskPrediction]], dict[str, list[MaskPrediction]]],
    *,
    model_weights: tuple[float, float],
    match_iou: float,
    mask_vote: float,
    score_threshold: float,
    min_models: int,
    required_model_indices: tuple[int, ...] = (),
    unconfirmed_score_threshold: float | None = None,
    evaluation_iou: float = 0.5,
) -> dict[str, float | int]:
    tp = fp = fn = 0
    matched_ious: list[float] = []
    for stem, targets in ground_truth.items():
        fused = fuse_mask_predictions(
            [predictions[0].get(stem, []), predictions[1].get(stem, [])],
            model_weights=model_weights,
            match_iou=match_iou,
            mask_vote=mask_vote,
            score_threshold=score_threshold,
            min_models=min_models,
            required_model_indices=required_model_indices,
            unconfirmed_score_threshold=unconfirmed_score_threshold,
        )
        matched = match_instances(
            [item.mask for item in fused], targets, evaluation_iou,
            prediction_bboxes=[item.bbox for item in fused],
        )
        tp += int(matched["tp"])
        fp += int(matched["fp"])
        fn += int(matched["fn"])
        matched_ious.extend(matched["matched_ious"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def calibrate(
    ground_truth: dict[str, list[np.ndarray]],
    predictions: tuple[dict[str, list[MaskPrediction]], dict[str, list[MaskPrediction]]],
    *,
    second_model_weights: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    match_ious: tuple[float, ...] = (0.3, 0.5, 0.7),
    mask_votes: tuple[float, ...] = (0.4, 0.5, 0.6),
    score_thresholds: tuple[float, ...] = (0.2, 0.3),
    min_model_options: tuple[int, ...] = (1, 2),
    required_model_indices: tuple[int, ...] = (),
    unconfirmed_score_thresholds: tuple[float | None, ...] = (None,),
) -> list[dict]:
    rows: list[dict] = []
    for second_weight, match_iou_value, mask_vote, score, min_models, unconfirmed_threshold in itertools.product(
        second_model_weights, match_ious, mask_votes, score_thresholds, min_model_options,
        unconfirmed_score_thresholds,
    ):
        metrics = evaluate_fusion(
            ground_truth,
            predictions,
            model_weights=(1.0, second_weight),
            match_iou=match_iou_value,
            mask_vote=mask_vote,
            score_threshold=score,
            min_models=min_models,
            required_model_indices=required_model_indices,
            unconfirmed_score_threshold=unconfirmed_threshold,
        )
        rows.append(
            {
                "model_weights": [1.0, second_weight],
                "match_iou": match_iou_value,
                "mask_vote": mask_vote,
                "score_threshold": score,
                "min_models": min_models,
                "required_model_indices": list(required_model_indices),
                "unconfirmed_score_threshold": unconfirmed_threshold,
                **metrics,
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["f1"], row["mean_matched_iou"], row["precision"], row["recall"]),
        reverse=True,
    )


def parse_prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Prediction input must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Prediction input must use non-empty NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def calibration_shape(image_hw: tuple[int, int], max_side: int) -> tuple[int, int]:
    height, width = image_hw
    if max_side <= 0 or max(height, width) <= max_side:
        return height, width
    scale = max_side / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def resize_masks(masks: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    if not len(masks):
        return np.zeros((0, height, width), dtype=bool)
    return np.stack(
        [cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
         for mask in masks]
    )


def _finalize_counts(counts: dict[str, float | int | list[float]]) -> dict[str, float | int]:
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    ious = counts["matched_ious"]
    assert isinstance(ious, list)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_matched_iou": float(np.mean(ious)) if ious else 0.0,
    }


def evaluate_locked_full_resolution(
    images: list[Path],
    dataset: Path,
    roots: tuple[Path, Path],
    names: tuple[str, str],
    config: dict,
) -> dict[str, dict[str, float | int]]:
    """Stream the single locked configuration at original resolution."""
    _BBOX_CACHE.clear()
    counters = {
        key: {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        for key in ("fusion", "primary_yolo11", "auxiliary_yolov8")
    }
    threshold = float(config["score_threshold"])
    for image_path in images:
        image_hw = tuple(read_image(image_path).shape[:2])
        targets = load_yolo_ground_truth(dataset / "valid" / "labels" / f"{image_path.stem}.txt", image_hw)
        per_model: list[list[MaskPrediction]] = []
        for name, root in zip(names, roots):
            bundle = load_prediction_bundle(root / "predictions" / f"{image_path.stem}.npz")
            if tuple(bundle["image_hw"]) != image_hw:
                raise ValueError(f"Prediction/image shape mismatch for {name}/{image_path.stem}")
            per_model.append([
                MaskPrediction(mask, float(score), int(class_id), name)
                for mask, score, class_id in zip(bundle["masks"], bundle["scores"], bundle["class_ids"])
            ])
        fused = fuse_mask_predictions(
            per_model,
            model_weights=tuple(config["model_weights"]),
            match_iou=float(config["match_iou"]),
            mask_vote=float(config["mask_vote"]),
            score_threshold=threshold,
            min_models=int(config["min_models"]),
            required_model_indices=tuple(config.get("required_model_indices", ())),
            unconfirmed_score_threshold=config.get("unconfirmed_score_threshold"),
        )
        accepted_primary = [item for item in per_model[0] if item.score >= threshold]
        accepted_auxiliary = [item for item in per_model[1] if item.score >= threshold]
        variants = {
            "fusion": ([item.mask for item in fused], [item.bbox for item in fused]),
            "primary_yolo11": ([item.mask for item in accepted_primary], [item.bbox for item in accepted_primary]),
            "auxiliary_yolov8": ([item.mask for item in accepted_auxiliary], [item.bbox for item in accepted_auxiliary]),
        }
        for key, (masks, bboxes) in variants.items():
            matched = match_instances(masks, targets, prediction_bboxes=bboxes)
            counters[key]["tp"] += int(matched["tp"])
            counters[key]["fp"] += int(matched["fp"])
            counters[key]["fn"] += int(matched["fn"])
            counters[key]["matched_ious"].extend(matched["matched_ious"])
        _BBOX_CACHE.clear()
    return {key: _finalize_counts(value) for key, value in counters.items()}


def evaluate_config_matrix_full_resolution(
    images: list[Path],
    dataset: Path,
    roots: tuple[Path, Path],
    names: tuple[str, str],
    configs: list[dict],
) -> list[dict]:
    """Evaluate several frozen configs in one streaming pass over full masks."""
    if not configs:
        return []
    if len(set(names)) != len(names):
        raise ValueError("Full-resolution config matrix requires unique model names")
    thresholds = {float(config["score_threshold"]) for config in configs}
    if len(thresholds) != 1:
        raise ValueError("Full-resolution config matrix requires one common score threshold")
    threshold = thresholds.pop()
    matrix_counters = [
        {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        for _ in configs
    ]
    baseline_counters = {
        key: {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        for key in ("primary_yolo11", "auxiliary_yolov8")
    }
    # Configs that differ only in the threshold for an unconfirmed primary
    # cluster share exactly the same clustering and fused masks.  Build the
    # lowest-threshold superset once and filter it for the remaining rows.
    # This matters for original-resolution masks, where clustering dominates
    # the cost of a recall-threshold matrix.
    structural_groups: dict[tuple, list[int]] = {}
    for index, config in enumerate(configs):
        key = (
            tuple(float(value) for value in config["model_weights"]),
            float(config["match_iou"]),
            float(config["mask_vote"]),
            int(config["min_models"]),
            tuple(int(value) for value in config.get("required_model_indices", ())),
        )
        structural_groups.setdefault(key, []).append(index)
    _BBOX_CACHE.clear()
    for image_path in images:
        image_hw = tuple(read_image(image_path).shape[:2])
        targets = load_yolo_ground_truth(dataset / "valid" / "labels" / f"{image_path.stem}.txt", image_hw)
        per_model: list[list[MaskPrediction]] = []
        for name, root in zip(names, roots):
            bundle = load_prediction_bundle(root / "predictions" / f"{image_path.stem}.npz")
            if tuple(bundle["image_hw"]) != image_hw:
                raise ValueError(f"Prediction/image shape mismatch for {name}/{image_path.stem}")
            per_model.append([
                MaskPrediction(mask, float(score), int(class_id), name)
                for mask, score, class_id in zip(bundle["masks"], bundle["scores"], bundle["class_ids"])
            ])
        for key, items in zip(baseline_counters, per_model):
            accepted = [item for item in items if item.score >= threshold]
            matched = match_instances(
                [item.mask for item in accepted], targets,
                prediction_bboxes=[item.bbox for item in accepted],
            )
            baseline_counters[key]["tp"] += int(matched["tp"])
            baseline_counters[key]["fp"] += int(matched["fp"])
            baseline_counters[key]["fn"] += int(matched["fn"])
            baseline_counters[key]["matched_ious"].extend(matched["matched_ious"])
        for indices in structural_groups.values():
            representative = configs[indices[0]]
            base_unconfirmed = min(
                threshold if configs[index].get("unconfirmed_score_threshold") is None
                else float(configs[index]["unconfirmed_score_threshold"])
                for index in indices
            )
            fused_superset = fuse_mask_predictions(
                per_model,
                model_weights=tuple(representative["model_weights"]),
                match_iou=float(representative["match_iou"]),
                mask_vote=float(representative["mask_vote"]),
                score_threshold=threshold,
                min_models=int(representative["min_models"]),
                required_model_indices=tuple(representative.get("required_model_indices", ())),
                unconfirmed_score_threshold=base_unconfirmed,
            )
            for index in indices:
                config = configs[index]
                unconfirmed = config.get("unconfirmed_score_threshold")
                effective_unconfirmed = threshold if unconfirmed is None else float(unconfirmed)
                fused = [
                    item for item in fused_superset
                    if len(item.sources) > 1 or item.score >= effective_unconfirmed
                ]
                matched = match_instances(
                    [item.mask for item in fused], targets,
                    prediction_bboxes=[item.bbox for item in fused],
                )
                matrix_counters[index]["tp"] += int(matched["tp"])
                matrix_counters[index]["fp"] += int(matched["fp"])
                matrix_counters[index]["fn"] += int(matched["fn"])
                matrix_counters[index]["matched_ious"].extend(matched["matched_ious"])
        _BBOX_CACHE.clear()
    baselines = {key: _finalize_counts(value) for key, value in baseline_counters.items()}
    return [
        {"config": config, "metrics": {"fusion": _finalize_counts(counts), **baselines}}
        for config, counts in zip(configs, matrix_counters)
    ]


def render_html(summary: dict) -> str:
    rows = []
    for index, row in enumerate(summary["top_candidates"], 1):
        rows.append(
            f'<tr><td>{index}</td><td>{html.escape(str(row["model_weights"]))}</td>'
            f'<td>{row["match_iou"]:.2f}</td><td>{row["mask_vote"]:.2f}</td>'
            f'<td>{row["score_threshold"]:.2f}</td><td>{row["min_models"]}</td>'
            f'<td>{row["precision"]:.4f}</td><td>{row["recall"]:.4f}</td>'
            f'<td>{row["f1"]:.4f}</td><td>{row["mean_matched_iou"]:.4f}</td></tr>'
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>YOLO 融合校准</title>
<style>body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#f3f6f6;margin:28px;color:#172126}}
main{{max-width:1100px;margin:auto;background:white;padding:24px;border:1px solid #d8e0e3}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px;border-bottom:1px solid #e3e8ea;text-align:left}}th{{background:#eef3f3}}</style></head>
<body><main><h1>YOLO 实例掩膜融合校准</h1><p>模型：{html.escape(' + '.join(summary["models"]))}</p>
<p>仅使用 valid；冻结 test 未读取。最优配置已写入 locked_fusion_config.json。</p>
<table><thead><tr><th>#</th><th>权重</th><th>匹配 IoU</th><th>掩膜投票</th><th>分数阈值</th><th>最少模型</th><th>P</th><th>R</th><th>F1</th><th>匹配 IoU</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", type=parse_prediction_spec, required=True, help="Exactly two NAME=PATH exports.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "Data_v6_yolo_ensemble_calibration")
    parser.add_argument(
        "--second-model-weights", nargs="+", type=float, default=(0.5, 0.75, 1.0, 1.25, 1.5),
        help="Candidate weights for the second model; the first model always has weight 1.0.",
    )
    parser.add_argument(
        "--require-primary", action="store_true",
        help="Require every fused output to include the first (primary) model.",
    )
    parser.add_argument(
        "--calibration-max-side", type=int, default=320,
        help="Downsample masks for grid search; the locked winner is re-evaluated at full resolution.",
    )
    parser.add_argument("--match-ious", nargs="+", type=float, default=(0.3, 0.5, 0.7))
    parser.add_argument("--mask-votes", nargs="+", type=float, default=(0.4, 0.5, 0.6))
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=(0.2, 0.3))
    parser.add_argument("--min-model-options", nargs="+", type=int, default=(1, 2))
    parser.add_argument(
        "--unconfirmed-score-thresholds", nargs="+", type=float, default=None,
        help="Optional higher score thresholds for clusters emitted by only one model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.prediction) != 2:
        raise ValueError("Exactly two --prediction NAME=PATH inputs are required")
    names = tuple(item[0] for item in args.prediction)
    roots = tuple(item[1].resolve() for item in args.prediction)
    if names[0] == names[1]:
        raise ValueError("Prediction model names must be unique")
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") != "valid" or manifest.get("held_out_test_used") is not False:
            raise RuntimeError(f"Fusion calibration only accepts valid-only prediction exports: {manifest_path}")

    images = sorted((args.dataset / args.split / "images").glob("*.JPG"))
    ground_truth: dict[str, list[np.ndarray]] = {}
    predictions: list[dict[str, list[MaskPrediction]]] = [{}, {}]
    for image_path in images:
        image_hw = tuple(read_image(image_path).shape[:2])
        search_hw = calibration_shape(image_hw, args.calibration_max_side)
        ground_truth[image_path.stem] = load_yolo_ground_truth(
            args.dataset / args.split / "labels" / f"{image_path.stem}.txt", search_hw
        )
        for model_index, (name, root) in enumerate(zip(names, roots)):
            bundle = load_prediction_bundle(root / "predictions" / f"{image_path.stem}.npz")
            if tuple(bundle["image_hw"]) != image_hw:
                raise ValueError(f"Prediction/image shape mismatch for {name}/{image_path.stem}")
            search_masks = resize_masks(bundle["masks"], search_hw)
            nonempty = search_masks.any(axis=(1, 2)) if len(search_masks) else np.zeros(0, dtype=bool)
            predictions[model_index][image_path.stem] = [
                MaskPrediction(mask, float(score), int(class_id), name)
                for mask, score, class_id in zip(
                    search_masks[nonempty], bundle["scores"][nonempty], bundle["class_ids"][nonempty]
                )
            ]

    second_weights = tuple(float(value) for value in args.second_model_weights)
    if any(value <= 0 for value in second_weights):
        raise ValueError("--second-model-weights must all be positive")
    ranking = calibrate(
        ground_truth,
        (predictions[0], predictions[1]),
        second_model_weights=second_weights,
        match_ious=tuple(args.match_ious),
        mask_votes=tuple(args.mask_votes),
        score_thresholds=tuple(args.score_thresholds),
        min_model_options=tuple(args.min_model_options),
        required_model_indices=(0,) if args.require_primary else (),
        unconfirmed_score_thresholds=(
            tuple(args.unconfirmed_score_thresholds) if args.unconfirmed_score_thresholds is not None else (None,)
        ),
    )
    if not ranking:
        raise RuntimeError("Fusion calibration produced no candidates")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    best = ranking[0]
    exact_metrics = evaluate_locked_full_resolution(images, args.dataset, roots, names, best)
    summary = {
        "version": "data_v6_yolo_ensemble_calibration_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selection_split": "valid",
        "held_out_test_used": False,
        "models": list(names),
        "prediction_roots": [str(root) for root in roots],
        "images": len(images),
        "search_candidates": len(ranking),
        "primary_model": names[0] if args.require_primary else None,
        "require_primary": args.require_primary,
        "allowed_second_model_weights": list(second_weights),
        "calibration_max_side": args.calibration_max_side,
        "best": best,
        "locked_full_resolution_metrics": exact_metrics,
        "top_candidates": ranking[:20],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    locked = {
        "version": "locked_yolo_mask_fusion_v1",
        "calibrated_on": "valid",
        "held_out_test_used": False,
        "models": list(names),
        "primary_model": names[0] if args.require_primary else None,
        "require_primary": args.require_primary,
        **{key: best[key] for key in (
            "model_weights", "match_iou", "mask_vote", "score_threshold", "min_models",
            "unconfirmed_score_threshold",
        )},
        "required_model_indices": best["required_model_indices"],
    }
    (output / "locked_fusion_config.json").write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.html").write_text(render_html(summary), encoding="utf-8")
    print(f"Report: {output / 'index.html'}")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
