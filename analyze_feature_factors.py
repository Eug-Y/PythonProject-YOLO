"""Analyze configurable multichannel feature influence using train/valid only."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import zlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

from multispectral import PROJECT_ROOT


FEATURE_GROUPS = {
    "RGB": ("R", "G", "B"),
    "raw_spectral": ("NIR", "RedEdge"),
    "current_indices": ("NDVI", "NDRE"),
    "red_nir_family": ("NDVI", "SAVI", "OSAVI", "MSAVI2", "EVI2"),
    "new_vegetation": ("GNDVI", "RECI", "ExG"),
    "soil_proxies": ("BI", "SCI", "RE_BSI_PROXY"),
    "v5_visible_robust": ("VARI", "ARVI"),
    "v5_pigment_senescence": ("SIPI", "PSRI"),
    "v5_rededge_chlorophyll": ("MCARI", "MTCI"),
    "v5_spatial": ("NDVI_GRAD", "NDVI_STD5"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v4_candidates")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v4_feature_factor_analysis",
    )
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--max-train-samples", type=int, default=150000)
    parser.add_argument("--max-valid-samples", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_binary_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for raw in label_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        values = np.asarray(raw.split()[1:], dtype=np.float64).reshape(-1, 2)
        pixels = np.rint(values * np.asarray([width, height])).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [pixels], 1)
    return mask


def sample_split(
    dataset: Path,
    feature_root: Path,
    split: str,
    channels: list[str],
    samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    image_rows: list[dict[str, object]] = []
    image_paths = sorted((dataset / split / "images").glob("*.JPG"))
    for group_id, image_path in enumerate(image_paths):
        feature_path = feature_root / split / "features" / f"{image_path.stem}.npy"
        label_path = dataset / split / "labels" / f"{image_path.stem}.txt"
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if features.dtype != np.uint8 or features.ndim != 3 or features.shape[-1] != len(channels):
            raise ValueError(f"Invalid feature array: {feature_path} {features.shape} {features.dtype}")
        height, width = features.shape[:2]
        mask = load_binary_mask(label_path, height, width)
        flat_mask = mask.reshape(-1)
        positive = np.flatnonzero(flat_mask == 1)
        negative = np.flatnonzero(flat_mask == 0)
        local_seed = seed ^ zlib.crc32(f"{split}:{image_path.stem}".encode())
        rng = np.random.default_rng(local_seed)
        pos_count = min(samples_per_class, len(positive))
        neg_count = min(samples_per_class, len(negative))
        pos_indices = rng.choice(positive, pos_count, replace=False)
        neg_indices = rng.choice(negative, neg_count, replace=False)
        indices = np.concatenate((pos_indices, neg_indices))
        y = np.concatenate((np.ones(pos_count, dtype=np.uint8), np.zeros(neg_count, dtype=np.uint8)))
        order = rng.permutation(len(indices))
        x = np.asarray(features.reshape(-1, len(channels))[indices[order]], dtype=np.float32) / 255.0
        xs.append(x)
        ys.append(y[order])
        groups.append(np.full(len(indices), group_id, dtype=np.int32))
        image_rows.append({
            "split": split,
            "stem": image_path.stem,
            "positive_pixels": int(len(positive)),
            "negative_pixels": int(len(negative)),
            "sampled_positive": int(pos_count),
            "sampled_negative": int(neg_count),
            "foreground_ratio": float(len(positive) / max(height * width, 1)),
        })
        if (group_id + 1) % 30 == 0 or group_id + 1 == len(image_paths):
            print(f"sample {split}: {group_id + 1}/{len(image_paths)}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups), image_rows


def cap_samples(x: np.ndarray, y: np.ndarray, groups: np.ndarray, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) <= maximum:
        return x, y, groups
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for class_id in (0, 1):
        candidates = np.flatnonzero(y == class_id)
        count = min(maximum // 2, len(candidates))
        chosen.append(rng.choice(candidates, count, replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return x[indices], y[indices], groups[indices]


def binary_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.uint8)
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    sorted_y = np.asarray(y, dtype=np.uint8)[order]
    positives = int(sorted_y.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)
    return float((precision * sorted_y).sum() / positives)


def discretized_mutual_information(values: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return 0.0
    bucket = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    table = np.zeros((len(edges) - 1, 2), dtype=np.float64)
    np.add.at(table, (bucket, y.astype(int)), 1)
    probability = table / max(table.sum(), 1.0)
    px = probability.sum(axis=1, keepdims=True)
    py = probability.sum(axis=0, keepdims=True)
    expected = px @ py
    valid = probability > 0
    return float(np.sum(probability[valid] * np.log(probability[valid] / expected[valid])))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 0.01) -> tuple[np.ndarray, float, dict[str, object]]:
    feature_count = x.shape[1]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, bias = parameters[:feature_count], parameters[-1]
        logits = x @ weights + bias
        probability = expit(logits)
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits) + 0.5 * l2 * np.dot(weights, weights))
        residual = probability - y
        gradient = np.concatenate((x.T @ residual / len(y) + l2 * weights, [residual.mean()]))
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(feature_count + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 120, "ftol": 1e-9},
    )
    return result.x[:feature_count], float(result.x[-1]), {
        "success": bool(result.success), "iterations": int(result.nit), "loss": float(result.fun), "message": str(result.message)
    }


def plot_correlation(path: Path, channels: list[str], correlation: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(channels)), channels, rotation=55, ha="right")
    axis.set_yticks(range(len(channels)), channels)
    axis.set_title("Train/valid sampled-pixel Spearman correlation")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    features = args.features.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((features / "manifest.json").read_text(encoding="utf-8"))
    channels = list(manifest["channels"])
    dataset_version = str(manifest.get("version", "multichannel_candidates"))

    train_x, train_y, train_groups, train_images = sample_split(
        dataset, features, "train", channels, args.samples_per_class, args.seed,
    )
    valid_x, valid_y, valid_groups, valid_images = sample_split(
        dataset, features, "valid", channels, args.samples_per_class, args.seed,
    )
    train_x, train_y, train_groups = cap_samples(train_x, train_y, train_groups, args.max_train_samples, args.seed)
    valid_x, valid_y, valid_groups = cap_samples(valid_x, valid_y, valid_groups, args.max_valid_samples, args.seed + 1)
    print(f"samples train={len(train_y)} valid={len(valid_y)} features={len(channels)}")

    combined_x = np.concatenate((train_x, valid_x))
    combined_y = np.concatenate((train_y, valid_y))
    spearman = np.asarray(spearmanr(combined_x, axis=0).statistic, dtype=np.float64)
    pearson = np.corrcoef(combined_x, rowvar=False)
    inverse_correlation = np.linalg.pinv(pearson + np.eye(len(channels)) * 1e-6)
    vif = np.diag(inverse_correlation)

    feature_rows: list[dict[str, object]] = []
    for index, name in enumerate(channels):
        values = combined_x[:, index]
        foreground = values[combined_y == 1]
        background = values[combined_y == 0]
        pooled = math.sqrt(max((foreground.var() + background.var()) * 0.5, 1e-12))
        auc = binary_auc(combined_y, values)
        feature_rows.append({
            "feature": name,
            "mean_foreground": float(foreground.mean()),
            "mean_background": float(background.mean()),
            "cohens_d": float((foreground.mean() - background.mean()) / pooled),
            "auc": auc,
            "auc_oriented": max(auc, 1.0 - auc),
            "direction": "higher_foreground" if auc >= 0.5 else "lower_foreground",
            "mutual_information": discretized_mutual_information(values, combined_y),
            "std": float(values.std()),
            "vif": float(vif[index]),
        })

    train_mean = train_x.mean(axis=0, dtype=np.float64)
    train_std = np.maximum(train_x.std(axis=0, dtype=np.float64), 1e-6)
    standardized_train = (train_x - train_mean) / train_std
    standardized_valid = (valid_x - train_mean) / train_std
    weights, bias, optimizer = fit_logistic(standardized_train.astype(np.float64), train_y.astype(np.float64))
    base_scores = standardized_valid @ weights + bias
    base_auc = binary_auc(valid_y, base_scores)
    base_ap = average_precision(valid_y, base_scores)
    rng = np.random.default_rng(args.seed + 2)

    permutation_rows: list[dict[str, object]] = []
    for index, name in enumerate(channels):
        permutation = rng.permutation(len(valid_y))
        permuted = standardized_valid.copy()
        permuted[:, index] = permuted[permutation, index]
        scores = permuted @ weights + bias
        permutation_rows.append({
            "feature": name,
            "auc_drop": float(base_auc - binary_auc(valid_y, scores)),
            "ap_drop": float(base_ap - average_precision(valid_y, scores)),
            "abs_standardized_weight": float(abs(weights[index])),
        })

    group_rows: list[dict[str, object]] = []
    for group_name, group_features in FEATURE_GROUPS.items():
        indices = [channels.index(name) for name in group_features if name in channels]
        if not indices:
            continue
        permutation = rng.permutation(len(valid_y))
        permuted = standardized_valid.copy()
        permuted[:, indices] = permuted[permutation][:, indices]
        scores = permuted @ weights + bias
        group_rows.append({
            "group": group_name,
            "features": ",".join(channels[index] for index in indices),
            "auc_drop": float(base_auc - binary_auc(valid_y, scores)),
            "ap_drop": float(base_ap - average_precision(valid_y, scores)),
        })

    redundancy_pairs = []
    for first in range(len(channels)):
        for second in range(first + 1, len(channels)):
            value = float(spearman[first, second])
            if abs(value) >= 0.95:
                redundancy_pairs.append({"first": channels[first], "second": channels[second], "spearman": value})

    permutation_map = {row["feature"]: row for row in permutation_rows}
    metric_map = {row["feature"]: row for row in feature_rows}
    redundancy_neighbors: dict[str, list[str]] = {name: [] for name in channels}
    for pair in redundancy_pairs:
        redundancy_neighbors[pair["first"]].append(pair["second"])
        redundancy_neighbors[pair["second"]].append(pair["first"])
    preliminary = []
    for name in channels:
        metric = metric_map[name]
        importance = permutation_map[name]
        flags = []
        if metric["std"] < 0.01:
            flags.append("near_zero_variance")
        if redundancy_neighbors[name]:
            flags.append("high_redundancy")
        if importance["auc_drop"] <= 0.0005:
            flags.append("low_individual_permutation_gain")
        preliminary.append({
            "feature": name,
            "flags": flags,
            "redundant_with": redundancy_neighbors[name],
            "requires_yolo_ablation": name not in {"R", "G", "B"},
        })

    correlation_rows = [
        {"feature": channels[row], **{channels[col]: float(spearman[row, col]) for col in range(len(channels))}}
        for row in range(len(channels))
    ]
    write_csv(output / "feature_metrics.csv", list(feature_rows[0]), feature_rows)
    write_csv(output / "permutation_importance.csv", list(permutation_rows[0]), permutation_rows)
    write_csv(output / "group_permutation_importance.csv", list(group_rows[0]), group_rows)
    write_csv(output / "correlation_spearman.csv", ["feature", *channels], correlation_rows)
    write_csv(output / "sampled_images.csv", list((train_images + valid_images)[0]), train_images + valid_images)
    plot_correlation(output / "correlation_heatmap.png", channels, spearman)

    summary = {
        "selection_splits": ["train", "valid"],
        "test_accessed": False,
        "dataset_version": dataset_version,
        "channels": channels,
        "samples": {"train": len(train_y), "valid": len(valid_y)},
        "images": {"train": len(train_images), "valid": len(valid_images)},
        "logistic_validation": {"auc": base_auc, "average_precision": base_ap, "optimizer": optimizer},
        "top_univariate": sorted(feature_rows, key=lambda row: row["auc_oriented"], reverse=True)[:10],
        "top_permutation": sorted(permutation_rows, key=lambda row: row["auc_drop"], reverse=True),
        "group_permutation": sorted(group_rows, key=lambda row: row["auc_drop"], reverse=True),
        "redundancy_pairs_abs_spearman_ge_0_95": redundancy_pairs,
        "preliminary_flags": preliminary,
        "warning": "Statistical flags are screening evidence, not final deletion decisions. YOLO grouped ablation is required.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows_html = "".join(
        f"<tr><td>{html.escape(str(row['feature']))}</td><td>{row['auc_oriented']:.4f}</td>"
        f"<td>{row['mutual_information']:.4f}</td><td>{row['cohens_d']:.3f}</td>"
        f"<td>{permutation_map[row['feature']]['auc_drop']:.4f}</td><td>{row['vif']:.1f}</td></tr>"
        for row in sorted(feature_rows, key=lambda item: item["auc_oriented"], reverse=True)
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(dataset_version)} 特征影响因子</title>
<style>body{{font-family:Segoe UI,'Microsoft YaHei',sans-serif;margin:32px;max-width:1200px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#f3f3f3;padding:2px 5px}}</style></head><body>
<h1>{html.escape(dataset_version)} 特征影响因子初筛</h1><p>仅使用 train/valid；test 未读取。训练样本 {len(train_y)}，验证样本 {len(valid_y)}。</p>
<p>轻量逻辑模型验证 AUC={base_auc:.4f}，AP={base_ap:.4f}。统计结果只用于初筛，最终舍弃必须通过 YOLO 消融。</p>
<table><thead><tr><th>因子</th><th>单变量 AUC</th><th>互信息</th><th>Cohen's d</th><th>置换 AUC 下降</th><th>VIF</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>Spearman 相关性</h2><img src="correlation_heatmap.png" alt="Spearman correlation">
<p><a href="summary.json">summary.json</a> · <a href="feature_metrics.csv">feature_metrics.csv</a> · <a href="permutation_importance.csv">permutation_importance.csv</a></p>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({
        "train_samples": len(train_y), "valid_samples": len(valid_y), "base_auc": base_auc,
        "base_ap": base_ap, "redundancy_pairs": len(redundancy_pairs),
        "top_permutation": summary["top_permutation"][:5],
    }, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
