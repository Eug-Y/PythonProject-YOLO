"""Create a validation-only comparison report for Data_v4 group ablations."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path

from run_feature_ablations import GROUP_ORDER, PROJECT, experiment_name, read_progress


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v4_yolo_group_ablation"


def main() -> None:
    group_config = json.loads(
        (ROOT / "configs" / "feature_ablation_groups.json").read_text(encoding="utf-8")
    )["groups"]
    rows = []
    for group in GROUP_ORDER:
        run_dir = PROJECT / experiment_name(group)
        progress = read_progress(run_dir / "results.csv")
        if not progress:
            continue
        rows.append(
            {
                "group": group,
                "channels": ", ".join(group_config[group]),
                **progress,
                "run_dir": str(run_dir.resolve()),
            }
        )
    if not rows:
        raise SystemExit("No completed Data_v4 ablation results found")
    rows.sort(key=lambda row: row["best_mask_map50_95"], reverse=True)
    leader = rows[0]
    for row in rows:
        row["delta_vs_leader"] = row["best_mask_map50_95"] - leader["best_mask_map50_95"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "channels",
        "best_epoch",
        "best_mask_map50_95",
        "best_mask_map50",
        "best_mask_precision",
        "best_mask_recall",
        "delta_vs_leader",
        "last_epoch",
        "run_dir",
    ]
    with (OUTPUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "version": "data_v4_yolo_group_ablation_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "leader": leader["group"],
        "rows": rows,
        "decision_status": "preliminary; do not deploy before follow-up ablation and multi-seed validation",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['group'])}</td>"
            f"<td>{html.escape(row['channels'])}</td>"
            f"<td>{row['best_epoch']}</td>"
            f"<td>{row['best_mask_map50_95']:.4f}</td>"
            f"<td>{row['best_mask_map50']:.4f}</td>"
            f"<td>{row['best_mask_precision']:.4f}</td>"
            f"<td>{row['best_mask_recall']:.4f}</td>"
            f"<td>{row['delta_vs_leader'] * 100:+.2f} pp</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v4 YOLO 分组消融</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:32px;color:#1f2937;background:#f8fafc}}
main{{max-width:1180px;margin:auto;background:white;padding:28px;border-radius:14px;box-shadow:0 8px 30px #0f172a12}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}
th{{background:#f1f5f9}}.note{{padding:14px;background:#fff7ed;border-left:4px solid #f97316}}
code{{background:#f1f5f9;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>Data_v4 YOLO 通道分组消融</h1>
<p>仅使用 train/valid，独立 test 未读取。统一 YOLO11n-seg、CoordAtt+ECA、640 px、seed 0 和训练预算。</p>
<table><thead><tr><th>组</th><th>通道</th><th>最佳轮次</th><th>Mask mAP50-95</th><th>Mask mAP50</th><th>Precision</th><th>Recall</th><th>相对第一名</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p class="note">当前第一名为 <code>{html.escape(leader['group'])}</code>。这是首轮筛选结果，仍需后续因子消融和多随机种子复验，不能据此直接更新网站模型。</p>
<p><strong>土壤指标说明：</strong><code>RE_BSI_PROXY</code> 是 RedEdge 代理指标，不是依赖 SWIR 的标准 BSI。</p>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(document, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()

