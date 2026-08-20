"""Summarize the screened Data_v5 stage-1 YOLO ablations."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path

from run_data_v5_ablations import GROUP_ORDER, PROJECT, experiment_name
from run_feature_ablations import read_progress


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v5_yolo_stage1_ablation"
PROMOTION_DELTA = 0.003


def main() -> None:
    groups = json.loads((ROOT / "configs" / "feature_ablation_groups_v5.json").read_text(encoding="utf-8"))["groups"]
    rows = []
    for group in GROUP_ORDER:
        run_dir = PROJECT / experiment_name(group)
        progress = read_progress(run_dir / "results.csv")
        if not progress:
            continue
        rows.append({
            "group": group,
            "added_feature": "none" if group == "rgb" else groups[group][3],
            "channels": ", ".join(groups[group]),
            **progress,
            "run_dir": str(run_dir.resolve()),
        })
    baseline = next((row for row in rows if row["group"] == "rgb"), None)
    if baseline is None:
        raise SystemExit("Missing Data_v5 RGB baseline")
    for row in rows:
        row["delta_vs_rgb"] = row["best_mask_map50_95"] - baseline["best_mask_map50_95"]
        if row["group"] == "rgb":
            row["decision"] = "baseline"
        elif row["delta_vs_rgb"] >= PROMOTION_DELTA:
            row["decision"] = "promote to multi-seed validation"
        elif row["delta_vs_rgb"] >= -0.002:
            row["decision"] = "within training noise; do not promote"
        else:
            row["decision"] = "discard"
    rows.sort(key=lambda item: item["best_mask_map50_95"], reverse=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "added_feature", "best_epoch", "best_mask_map50_95", "best_mask_map50",
        "best_mask_precision", "best_mask_recall", "delta_vs_rgb", "decision", "run_dir",
    ]
    with (OUTPUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "version": "data_v5_yolo_stage1_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "rgb_baseline_mask_map50_95": baseline["best_mask_map50_95"],
        "promotion_delta": PROMOTION_DELTA,
        "rows": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    body = "".join(
        "<tr>"
        f"<td>{html.escape(row['added_feature'])}</td><td>{row['best_epoch']}</td>"
        f"<td>{row['best_mask_map50_95']:.4f}</td><td>{row['best_mask_map50']:.4f}</td>"
        f"<td>{row['delta_vs_rgb'] * 100:+.2f} pp</td><td>{html.escape(row['decision'])}</td>"
        "</tr>"
        for row in rows
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v5 第一阶段消融</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:1050px;margin:auto;background:#fff;padding:28px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}</style></head><body><main>
<h1>Data_v5 YOLO 第一阶段消融</h1><p>统一 train/valid、30 轮、640 px、seed 0；独立 test 未读取。晋级线为相对 RGB +0.30 pp。</p>
<table><thead><tr><th>新增因子</th><th>最佳轮次</th><th>Mask mAP50-95</th><th>Mask mAP50</th><th>相对 RGB</th><th>判断</th></tr></thead><tbody>{body}</tbody></table>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
