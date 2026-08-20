"""Summarize paired three-seed Data_v5 validation results."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from run_data_v5_ablations import experiment_name as stage1_name
from run_data_v5_multiseed import GROUPS, PROJECT, experiment_name
from run_feature_ablations import read_progress


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v5_yolo_multiseed_validation"
SEEDS = (0, 1, 2)
PROMOTION_DELTA = 0.003
T95_DF2 = 4.3026527297


def run_dir(group: str, seed: int) -> Path:
    return PROJECT / (stage1_name(group) if seed == 0 else experiment_name(group, seed))


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = T95_DF2 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, mean - margin, mean + margin


def main() -> None:
    groups_config = json.loads((ROOT / "configs" / "feature_ablation_groups_v5.json").read_text(encoding="utf-8"))["groups"]
    runs = []
    progress_by_key = {}
    for seed in SEEDS:
        for group in GROUPS:
            directory = run_dir(group, seed)
            progress = read_progress(directory / "results.csv")
            if not progress:
                raise SystemExit(f"Missing result for {group} seed {seed}: {directory}")
            key = (group, seed)
            progress_by_key[key] = progress
            runs.append({
                "group": group,
                "feature": "RGB" if group == "rgb" else groups_config[group][3],
                "seed": seed,
                **progress,
                "run_dir": str(directory.resolve()),
            })

    group_rows = []
    for group in GROUPS:
        values = [progress_by_key[(group, seed)]["best_mask_map50_95"] for seed in SEEDS]
        mean, low, high = mean_ci95(values)
        group_rows.append({
            "group": group,
            "feature": "RGB" if group == "rgb" else groups_config[group][3],
            "mean_mask_map50_95": mean,
            "std_mask_map50_95": statistics.stdev(values),
            "ci95_low": low,
            "ci95_high": high,
            "seed_values": values,
        })

    paired_rows = []
    for group in GROUPS:
        if group == "rgb":
            continue
        deltas = [
            progress_by_key[(group, seed)]["best_mask_map50_95"]
            - progress_by_key[("rgb", seed)]["best_mask_map50_95"]
            for seed in SEEDS
        ]
        precision_deltas = [
            progress_by_key[(group, seed)]["best_mask_precision"]
            - progress_by_key[("rgb", seed)]["best_mask_precision"]
            for seed in SEEDS
        ]
        recall_deltas = [
            progress_by_key[(group, seed)]["best_mask_recall"]
            - progress_by_key[("rgb", seed)]["best_mask_recall"]
            for seed in SEEDS
        ]
        mean, low, high = mean_ci95(deltas)
        positive = sum(delta > 0.0 for delta in deltas)
        stable = mean >= PROMOTION_DELTA and positive >= 2 and low > -0.003
        paired_rows.append({
            "group": group,
            "feature": groups_config[group][3],
            "seed_deltas": deltas,
            "mean_delta": mean,
            "std_delta": statistics.stdev(deltas),
            "ci95_low": low,
            "ci95_high": high,
            "positive_seeds": positive,
            "mean_precision_delta": statistics.mean(precision_deltas),
            "mean_recall_delta": statistics.mean(recall_deltas),
            "decision": "stable candidate" if stable else "do not promote",
        })
    paired_rows.sort(key=lambda item: item["mean_delta"], reverse=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    with (OUTPUT / "paired_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "feature", "mean_delta", "std_delta", "ci95_low", "ci95_high", "positive_seeds",
            "mean_precision_delta", "mean_recall_delta", "decision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired_rows)
    summary = {
        "version": "data_v5_yolo_multiseed_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "seeds": list(SEEDS),
        "promotion_delta": PROMOTION_DELTA,
        "group_summary": group_rows,
        "paired_vs_rgb": paired_rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    group_map = {row["group"]: row for row in group_rows}
    body = "".join(
        "<tr>"
        f"<td>{html.escape(row['feature'])}</td>"
        f"<td>{group_map[row['group']]['mean_mask_map50_95']:.4f}</td>"
        f"<td>{row['mean_delta'] * 100:+.2f} pp</td>"
        f"<td>{row['ci95_low'] * 100:+.2f} 至 {row['ci95_high'] * 100:+.2f} pp</td>"
        f"<td>{row['positive_seeds']}/3</td>"
        f"<td>{row['mean_precision_delta'] * 100:+.2f} pp</td>"
        f"<td>{row['mean_recall_delta'] * 100:+.2f} pp</td>"
        f"<td>{html.escape(row['decision'])}</td>"
        "</tr>"
        for row in paired_rows
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v5 三 seed 复验</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:1150px;margin:auto;background:#fff;padding:28px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}</style></head><body><main>
<h1>Data_v5 三随机种子复验</h1><p>seed 0/1/2 配对比较，统一 30 轮、640 px；test 未读取。晋级线为平均 +0.30 pp、至少 2/3 seed 为正且置信区间无明显退化。</p>
<table><thead><tr><th>因子</th><th>平均 mAP50-95</th><th>平均相对 RGB</th><th>95% CI</th><th>正向 seed</th><th>Precision</th><th>Recall</th><th>判断</th></tr></thead><tbody>{body}</tbody></table>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
