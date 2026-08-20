"""Build a compact visual gallery from the final five-channel test validation plots."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE = PROJECT_ROOT / "reports" / "Data_v3_grouped_reviewed_v1_test_recheck"
OUTPUT = PROJECT_ROOT / "reports" / "Data_v3_prediction_gallery"
TEST_IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "test" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Validation report containing metrics and plots.")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Gallery output directory.")
    return parser.parse_args()


def validation_batch_index(path: Path) -> int:
    match = re.search(r"val_batch(\d+)_labels", path.stem)
    return int(match.group(1)) if match else 10**9


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def add_header(left: np.ndarray, right: np.ndarray, sample_name: str) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    pair = np.hstack((left, right))
    header = np.full((76, pair.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, sample_name, (24, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(header, "Ground truth", (24, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 90, 190), 2, cv2.LINE_AA)
    cv2.putText(
        header,
        "Model prediction",
        (left.shape[1] + 24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (35, 135, 65),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((header, pair))


def metric(metrics: dict[str, float], key: str) -> str:
    return f"{metrics[key]:.4f}"


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    test_names = sorted(path.name for path in TEST_IMAGES.glob("*.JPG"))
    label_plots = sorted(source.glob("val_batch*_labels.jpg"), key=validation_batch_index)
    image_metrics_path = source / "per_image_metrics.json"
    image_metrics = (
        json.loads(image_metrics_path.read_text(encoding="utf-8")) if image_metrics_path.is_file() else {}
    )
    comparisons: list[tuple[str, Path]] = []
    stacked: list[np.ndarray] = []

    for index, label_path in enumerate(label_plots):
        pred_path = label_path.with_name(label_path.name.replace("_labels.jpg", "_pred.jpg"))
        if not pred_path.is_file():
            continue
        sample_name = test_names[index] if index < len(test_names) else label_path.stem
        comparison = add_header(read_image(label_path), read_image(pred_path), sample_name)
        output_path = output / f"{Path(sample_name).stem}_comparison.jpg"
        write_image(output_path, comparison)
        comparisons.append((sample_name, output_path))
        stacked.append(comparison)

    if not comparisons:
        raise RuntimeError(f"No validation plot pairs found in {source}")
    comparisons.sort(
        key=lambda item: (
            image_metrics.get(item[0], {}).get("mask", {}).get("f1", 1.0),
            item[0],
        )
    )

    max_width = max(image.shape[1] for image in stacked)
    normalized: list[np.ndarray] = []
    for image in stacked:
        if image.shape[1] != max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(image, (max_width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        normalized.append(image)
        normalized.append(np.full((20, max_width, 3), 255, dtype=np.uint8))
    write_image(output / "comparison_grid.jpg", np.vstack(normalized[:-1]))

    cards = "\n".join(
        f'''<article class="sample">
          <h2>{html.escape(name)}</h2>
          <p class="note">Mask F1: {image_metrics.get(name, {}).get("mask", {}).get("f1", float("nan")):.4f}
          · TP {image_metrics.get(name, {}).get("mask", {}).get("tp", "-")}
          · FP {image_metrics.get(name, {}).get("mask", {}).get("fp", "-")}
          · FN {image_metrics.get(name, {}).get("mask", {}).get("fn", "-")}</p>
          <img src="{html.escape(path.name)}" alt="Ground truth and model prediction for {html.escape(name)}">
        </article>'''
        for name, path in comparisons
    )
    source_relative = Path(os.path.relpath(source, output)).as_posix()
    curves = "\n".join(
        f'<a class="curve" href="{source_relative}/{name}"><img src="{source_relative}/{name}" alt="{name}"></a>'
        for name in ("MaskPR_curve.png", "MaskF1_curve.png", "confusion_matrix_normalized.png")
    )
    page = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Data_v3 Test Prediction Gallery</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", sans-serif; }}
    body {{ margin: 0; background: #f3f5f7; color: #17202a; }}
    main {{ width: min(1500px, calc(100% - 32px)); margin: 28px auto 64px; }}
    h1 {{ margin-bottom: 8px; }}
    .note {{ color: #536273; margin-top: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 22px 0; }}
    .metric {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 12px #13223812; }}
    .metric span {{ display: block; color: #68788a; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 7px; font-size: 24px; }}
    .sample {{ background: white; border-radius: 14px; padding: 16px; margin: 18px 0; box-shadow: 0 2px 14px #13223812; }}
    .sample h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .sample img {{ display: block; width: 100%; border-radius: 8px; }}
    .curves {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .curve img {{ width: 100%; background: white; border-radius: 10px; }}
    @media (max-width: 800px) {{ .metrics, .curves {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body><main>
  <h1>Data_v3 独立测试集：分割效果</h1>
  <p class="note">左侧为人工真值，右侧为模型预测。以下图像来自最终 best.pt 的 5 通道、1024 像素测试验证。</p>
  <section class="metrics">
    <div class="metric"><span>Box mAP50</span><strong>{metric(metrics, 'metrics/mAP50(B)')}</strong></div>
    <div class="metric"><span>Box mAP50-95</span><strong>{metric(metrics, 'metrics/mAP50-95(B)')}</strong></div>
    <div class="metric"><span>Mask mAP50</span><strong>{metric(metrics, 'metrics/mAP50(M)')}</strong></div>
    <div class="metric"><span>Mask mAP50-95</span><strong>{metric(metrics, 'metrics/mAP50-95(M)')}</strong></div>
  </section>
  {cards}
  <h2>测试曲线</h2><section class="curves">{curves}</section>
</main></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"comparisons={len(comparisons)}")
    print(f"gallery={output / 'index.html'}")
    print(f"grid={output / 'comparison_grid.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
