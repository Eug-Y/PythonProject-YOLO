"""绘制 YOLO 训练结果的 3×2 自定义曲线图。

用法：
    python plot_results.py path/to/results.csv
    python plot_results.py path/to/results.csv --output path/to/curves.png
"""

import argparse
import os

import matplotlib

# 使用非交互后端，避免脚本在保存图片后等待 GUI 窗口关闭。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def smooth(values, window=5):
    """使用居中的滑动平均平滑曲线，并保持数据长度不变。"""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    padded = np.pad(
        values,
        (window // 2, window - 1 - window // 2),
        mode="edge",
    )
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def add_legend(ax, **kwargs):
    """仅在当前子图存在曲线时添加图例。"""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def plot_results(csv_path, out_path=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "epoch" not in df.columns:
        raise ValueError(f"训练记录缺少 epoch 列：{csv_path}")

    epochs = df["epoch"].to_numpy(dtype=int)
    exp_name = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"训练结果 — {exp_name}（共 {len(df)} epochs）",
        fontsize=15,
        fontweight="bold",
    )

    # [0, 0] 训练损失
    ax = axes[0, 0]
    train_losses = [
        ("train/box_loss", "box_loss", "#1f77b4"),
        ("train/seg_loss", "seg_loss", "#ff7f0e"),
        ("train/cls_loss", "cls_loss", "#2ca02c"),
        ("train/dfl_loss", "dfl_loss", "#d62728"),
    ]
    for column, label, color in train_losses:
        if column in df.columns:
            values = df[column].to_numpy(dtype=float)
            ax.plot(epochs, values, color=color, alpha=0.18, linewidth=0.6)
            ax.plot(epochs, smooth(values, 7), color=color, label=label, linewidth=1.5)
    ax.set_title("训练 Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    add_legend(ax, fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # [0, 1] 验证损失
    ax = axes[0, 1]
    val_losses = [
        ("val/box_loss", "box_loss", "#1f77b4"),
        ("val/seg_loss", "seg_loss", "#ff7f0e"),
        ("val/cls_loss", "cls_loss", "#2ca02c"),
        ("val/dfl_loss", "dfl_loss", "#d62728"),
    ]
    for column, label, color in val_losses:
        if column in df.columns:
            values = df[column].to_numpy(dtype=float)
            ax.plot(epochs, values, color=color, alpha=0.18, linewidth=0.6)
            ax.plot(epochs, smooth(values, 7), color=color, label=label, linewidth=1.5)
    ax.set_title("验证 Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    add_legend(ax, fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # [1, 0] Box 精确率与召回率
    ax = axes[1, 0]
    box_metrics = [
        ("metrics/precision(B)", "精确率", "#1f77b4"),
        ("metrics/recall(B)", "召回率", "#ff7f0e"),
    ]
    for column, label, color in box_metrics:
        if column in df.columns:
            values = df[column].to_numpy(dtype=float)
            best_index = int(np.nanargmax(values))
            ax.plot(epochs, smooth(values, 5), color=color, label=label, linewidth=1.5)
            ax.scatter(epochs[best_index], values[best_index], color=color, s=35, zorder=5, marker="*")
            ax.annotate(
                f"{values[-1]:.3f}",
                (epochs[-1], values[-1]),
                textcoords="offset points",
                xytext=(5, 0),
                fontsize=7,
                color=color,
            )
    ax.set_title("Box 精确率与召回率", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    add_legend(ax, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # [1, 1] Mask 精确率与召回率
    ax = axes[1, 1]
    mask_metrics = [
        ("metrics/precision(M)", "精确率", "#1f77b4"),
        ("metrics/recall(M)", "召回率", "#ff7f0e"),
    ]
    for column, label, color in mask_metrics:
        if column in df.columns:
            values = df[column].to_numpy(dtype=float)
            best_index = int(np.nanargmax(values))
            ax.plot(epochs, smooth(values, 5), color=color, label=label, linewidth=1.5)
            ax.scatter(epochs[best_index], values[best_index], color=color, s=35, zorder=5, marker="*")
            ax.annotate(
                f"{values[-1]:.3f}",
                (epochs[-1], values[-1]),
                textcoords="offset points",
                xytext=(5, 0),
                fontsize=7,
                color=color,
            )
    ax.set_title("Mask 精确率与召回率", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    add_legend(ax, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # [2, 0] Box/Mask mAP
    ax = axes[2, 0]
    map_metrics = [
        ("metrics/mAP50(B)", "mAP50(B)", "#1f77b4", "-"),
        ("metrics/mAP50-95(B)", "mAP50-95(B)", "#1f77b4", ":"),
        ("metrics/mAP50(M)", "mAP50(M)", "#ff7f0e", "--"),
        ("metrics/mAP50-95(M)", "mAP50-95(M)", "#ff7f0e", "-."),
    ]
    for column, label, color, line_style in map_metrics:
        if column in df.columns:
            values = df[column].to_numpy(dtype=float)
            best_index = int(np.nanargmax(values))
            ax.plot(
                epochs,
                smooth(values, 5),
                color=color,
                label=label,
                linewidth=1.5,
                linestyle=line_style,
            )
            ax.scatter(epochs[best_index], values[best_index], color=color, s=35, zorder=5, marker="*")
            ax.annotate(
                f"{values[-1]:.3f}",
                (epochs[-1], values[-1]),
                textcoords="offset points",
                xytext=(5, 0),
                fontsize=7,
                color=color,
            )
    ax.set_title("mAP 曲线（★ 为最佳值）", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    add_legend(ax, fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    # [2, 1] 学习率
    ax = axes[2, 1]
    learning_rates = [
        ("lr/pg0", "lr/pg0", "#1f77b4"),
        ("lr/pg1", "lr/pg1", "#ff7f0e"),
        ("lr/pg2", "lr/pg2", "#2ca02c"),
    ]
    for column, label, color in learning_rates:
        if column in df.columns:
            ax.plot(epochs, df[column], color=color, label=label, linewidth=1.2)
    ax.set_title("学习率", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    add_legend(ax, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if out_path is None:
        stem, _ = os.path.splitext(csv_path)
        out_path = f"{stem}_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存：{out_path}")
    print("\n" + "=" * 72)
    print(f"训练总结 — {exp_name}")
    print("=" * 72)
    summary_metrics = [
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
    ]
    for column in summary_metrics:
        if column in df.columns:
            values = df[column].to_numpy(dtype=float)
            best_index = int(np.nanargmax(values))
            name = column.replace("metrics/", "")
            print(
                f"{name:<20s}  Best: {values[best_index]:.4f} "
                f"@ Epoch {epochs[best_index]:3d}  |  Final: {values[-1]:.4f}"
            )
    print("=" * 72)
    return os.path.abspath(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 YOLO 训练结果的 3×2 曲线图")
    parser.add_argument("results_csv", help="Ultralytics 训练产生的 results.csv")
    parser.add_argument("--output", "-o", help="输出图片路径；默认与 CSV 同目录")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.results_csv):
        raise FileNotFoundError(f"文件不存在：{args.results_csv}")
    plot_results(args.results_csv, args.output)


if __name__ == "__main__":
    main()
