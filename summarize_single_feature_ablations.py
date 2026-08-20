"""Summarize RGB plus seven one-feature-at-a-time Data_v4 ablations."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path

from run_feature_ablations import PROJECT, SINGLE_GROUP_ORDER, experiment_name, read_progress


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v4_yolo_single_feature_ablation"


def main() -> None:
    groups = json.loads(
        (ROOT / "configs" / "feature_ablation_groups.json").read_text(encoding="utf-8")
    )["groups"]
    ordered = ("rgb",) + SINGLE_GROUP_ORDER
    rows = []
    for group in ordered:
        run_dir = PROJECT / experiment_name(group)
        progress = read_progress(run_dir / "results.csv")
        if not progress:
            continue
        rows.append(
            {
                "group": group,
                "added_feature": "none" if group == "rgb" else groups[group][3],
                "channels": ", ".join(groups[group]),
                **progress,
                "run_dir": str(run_dir.resolve()),
            }
        )
    baseline = next((row for row in rows if row["group"] == "rgb"), None)
    if baseline is None:
        raise SystemExit("Missing RGB baseline")
    for row in rows:
        row["delta_vs_rgb"] = row["best_mask_map50_95"] - baseline["best_mask_map50_95"]
        if row["group"] == "rgb":
            row["preliminary_decision"] = "baseline"
        elif row["delta_vs_rgb"] > 0.002:
            row["preliminary_decision"] = "multi-seed candidate"
        elif row["delta_vs_rgb"] >= -0.002:
            row["preliminary_decision"] = "within noise; multi-seed only if scientifically required"
        else:
            row["preliminary_decision"] = "discard from expensive follow-up"
    rows.sort(key=lambda row: row["best_mask_map50_95"], reverse=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "group",
        "added_feature",
        "best_epoch",
        "best_mask_map50_95",
        "best_mask_map50",
        "best_mask_precision",
        "best_mask_recall",
        "delta_vs_rgb",
        "preliminary_decision",
        "run_dir",
    ]
    with (OUTPUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "version": "data_v4_yolo_single_feature_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "rgb_baseline_mask_map50_95": baseline["best_mask_map50_95"],
        "rows": rows,
        "decision_threshold": "+0.2 percentage points versus RGB for multi-seed candidacy",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row['added_feature'])}</td>"
            f"<td>{row['best_epoch']}</td>"
            f"<td>{row['best_mask_map50_95']:.4f}</td>"
            f"<td>{row['best_mask_map50']:.4f}</td>"
            f"<td>{row['delta_vs_rgb'] * 100:+.2f} pp</td>"
            f"<td>{html.escape(row['preliminary_decision'])}</td>"
            "</tr>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v4 单因子消融</title><style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:1100px;margin:auto;background:white;padding:28px;border-radius:14px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}.note{{background:#fff7ed;border-left:4px solid #f97316;padding:14px}}
</style></head><body><main><h1>Data_v4 YOLO 单因子消融</h1>
<p>每次只在 RGB 后加入一个因子，统一 train/valid、40 轮、640 px、seed 0。test 未读取。</p>
<table><thead><tr><th>新增因子</th><th>最佳轮次</th><th>Mask mAP50-95</th><th>Mask mAP50</th><th>相对 RGB</th><th>初步判断</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table>
<p class="note"><code>RE_BSI_PROXY</code> 是项目实验性 RedEdge 土壤代理，不是标准 BSI。最终保留必须经过多 seed 复验。</p>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()

