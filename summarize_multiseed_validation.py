"""Summarize paired three-seed RGB versus RGB+RE_BSI_PROXY validation."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from run_feature_ablations import PROJECT, read_progress
from run_multiseed_validation import run_name


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v4_yolo_multiseed_validation"
GROUPS = ("rgb", "add_re_bsi_proxy")


def path_for(group: str, seed: int) -> Path:
    if seed == 0:
        name = "v4_ablate_rgb_n_seed0_v1" if group == "rgb" else "v4_ablate_add_re_bsi_proxy_n_seed0_v1"
    else:
        name = run_name(group, seed)
    return PROJECT / name


def summarize(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    # Student-t 95% critical value for df=2 (three seeds).
    half_width = 4.3026527299 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"values": values, "mean": mean, "std": std, "ci95": [mean - half_width, mean + half_width]}


def main() -> None:
    records = []
    by_group: dict[str, list[float]] = {group: [] for group in GROUPS}
    by_seed: dict[int, dict[str, float]] = {}
    for seed in (0, 1, 2):
        by_seed[seed] = {}
        for group in GROUPS:
            run_dir = path_for(group, seed)
            progress = read_progress(run_dir / "results.csv")
            if not progress:
                raise SystemExit(f"Missing results: {run_dir}")
            score = progress["best_mask_map50_95"]
            by_group[group].append(score)
            by_seed[seed][group] = score
            records.append(
                {
                    "group": group,
                    "seed": seed,
                    "best_epoch": progress["best_epoch"],
                    "mask_map50_95": score,
                    "mask_map50": progress["best_mask_map50"],
                    "precision": progress["best_mask_precision"],
                    "recall": progress["best_mask_recall"],
                    "run_dir": str(run_dir.resolve()),
                }
            )
    group_stats = {group: summarize(values) for group, values in by_group.items()}
    paired_deltas = [by_seed[seed]["add_re_bsi_proxy"] - by_seed[seed]["rgb"] for seed in (0, 1, 2)]
    delta_stats = summarize(paired_deltas)
    positive_seeds = sum(delta > 0.0 for delta in paired_deltas)
    stable_gain = delta_stats["mean"] > 0.002 and positive_seeds >= 2
    decision = "retain RE_BSI_PROXY for final high-resolution training" if stable_gain else "discard RE_BSI_PROXY; retain RGB"

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "per_seed.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "version": "data_v4_yolo_multiseed_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "metric": "best validation Mask mAP50-95",
        "groups": group_stats,
        "paired_proxy_minus_rgb": {**delta_stats, "positive_seed_count": positive_seeds},
        "stable_gain_rule": "paired mean > +0.002 and positive in at least 2 of 3 seeds",
        "stable_gain": stable_gain,
        "decision": decision,
        "soil_proxy_disclaimer": "RE_BSI_PROXY is experimental and is not standard SWIR-based BSI.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for seed in (0, 1, 2):
        rgb = by_seed[seed]["rgb"]
        proxy = by_seed[seed]["add_re_bsi_proxy"]
        rows.append(
            f"<tr><td>{seed}</td><td>{rgb:.4f}</td><td>{proxy:.4f}</td><td>{(proxy-rgb)*100:+.2f} pp</td></tr>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v4 多 seed 复验</title><style>body{{font-family:system-ui;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:900px;margin:auto;background:#fff;padding:28px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}.note{{padding:14px;background:#fff7ed;border-left:4px solid #f97316}}</style></head>
<body><main><h1>RGB 与土壤代理的三 seed 复验</h1><p>仅使用 train/valid；test 未读取。</p>
<table><thead><tr><th>Seed</th><th>RGB</th><th>RGB + RE_BSI_PROXY</th><th>配对差值</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>RGB 均值：{group_stats['rgb']['mean']:.4f}；代理均值：{group_stats['add_re_bsi_proxy']['mean']:.4f}；平均配对差值：{delta_stats['mean']*100:+.2f} pp。</p>
<p><strong>结论：</strong>{html.escape(decision)}</p><p class="note">RE_BSI_PROXY 是实验性 RedEdge 土壤代理，不是标准 SWIR-BSI。</p>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()

