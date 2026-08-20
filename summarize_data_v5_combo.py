"""Compare the three-seed MCARI+NDVI_STD5 combination with prior candidates."""

from __future__ import annotations

import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from run_data_v5_ablations import PROJECT, experiment_name as stage1_name
from run_data_v5_combo import SEEDS, experiment_name as combo_name
from run_data_v5_multiseed import experiment_name as multiseed_name
from run_feature_ablations import read_progress


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v5_yolo_combo_validation"
T95_DF2 = 4.3026527297


def previous_dir(group: str, seed: int) -> Path:
    return PROJECT / (stage1_name(group) if seed == 0 else multiseed_name(group, seed))


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    margin = T95_DF2 * std / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def main() -> None:
    references = ("rgb", "add_ndvi_std5", "add_mcari")
    metrics = {}
    for seed in SEEDS:
        combo = read_progress(PROJECT / combo_name(seed) / "results.csv")
        if not combo:
            raise SystemExit(f"Missing combo result for seed {seed}")
        metrics[("combo", seed)] = combo
        for group in references:
            progress = read_progress(previous_dir(group, seed) / "results.csv")
            if not progress:
                raise SystemExit(f"Missing {group} result for seed {seed}")
            metrics[(group, seed)] = progress

    combo_values = [metrics[("combo", seed)]["best_mask_map50_95"] for seed in SEEDS]
    combo_mean, combo_low, combo_high = mean_ci95(combo_values)
    comparisons = []
    labels = {"rgb": "RGB", "add_ndvi_std5": "NDVI_STD5", "add_mcari": "MCARI"}
    for group in references:
        deltas = [
            metrics[("combo", seed)]["best_mask_map50_95"] - metrics[(group, seed)]["best_mask_map50_95"]
            for seed in SEEDS
        ]
        mean, low, high = mean_ci95(deltas)
        comparisons.append({
            "reference": labels[group],
            "seed_deltas": deltas,
            "mean_delta": mean,
            "std_delta": statistics.stdev(deltas),
            "ci95_low": low,
            "ci95_high": high,
            "positive_seeds": sum(value > 0.0 for value in deltas),
        })

    best_single_deltas = [
        metrics[("combo", seed)]["best_mask_map50_95"]
        - max(
            metrics[("add_ndvi_std5", seed)]["best_mask_map50_95"],
            metrics[("add_mcari", seed)]["best_mask_map50_95"],
        )
        for seed in SEEDS
    ]
    best_mean, best_low, best_high = mean_ci95(best_single_deltas)
    precision_delta = statistics.mean([
        metrics[("combo", seed)]["best_mask_precision"] - metrics[("rgb", seed)]["best_mask_precision"]
        for seed in SEEDS
    ])
    recall_delta = statistics.mean([
        metrics[("combo", seed)]["best_mask_recall"] - metrics[("rgb", seed)]["best_mask_recall"]
        for seed in SEEDS
    ])
    adopt_combo = best_mean >= 0.003 and sum(value > 0.0 for value in best_single_deltas) >= 2
    decision = "adopt combo" if adopt_combo else "prefer simpler best single feature"
    summary = {
        "version": "data_v5_yolo_combo_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "seeds": list(SEEDS),
        "combo": ["R", "G", "B", "MCARI", "NDVI_STD5"],
        "combo_seed_values": combo_values,
        "combo_mean_mask_map50_95": combo_mean,
        "combo_ci95": [combo_low, combo_high],
        "comparisons": comparisons,
        "vs_best_single": {
            "seed_deltas": best_single_deltas,
            "mean_delta": best_mean,
            "ci95": [best_low, best_high],
            "positive_seeds": sum(value > 0.0 for value in best_single_deltas),
        },
        "mean_precision_delta_vs_rgb": precision_delta,
        "mean_recall_delta_vs_rgb": recall_delta,
        "decision": decision,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = "".join(
        f"<tr><td>{html.escape(item['reference'])}</td><td>{item['mean_delta'] * 100:+.2f} pp</td>"
        f"<td>{item['ci95_low'] * 100:+.2f} 至 {item['ci95_high'] * 100:+.2f} pp</td>"
        f"<td>{item['positive_seeds']}/3</td></tr>"
        for item in comparisons
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v5 组合复验</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:1000px;margin:auto;background:#fff;padding:28px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}</style></head><body><main>
<h1>MCARI + NDVI_STD5 三 seed 组合复验</h1><p>组合平均 Mask mAP50-95：{combo_mean:.4f}。test 未读取。</p>
<table><thead><tr><th>参考方案</th><th>组合平均差值</th><th>95% CI</th><th>正向 seed</th></tr></thead><tbody>{rows}</tbody></table>
<p>相对每个 seed 的最佳单因子：{best_mean * 100:+.2f} pp；Precision 相对 RGB：{precision_delta * 100:+.2f} pp；Recall：{recall_delta * 100:+.2f} pp。</p><p><strong>{html.escape(decision)}</strong></p>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
