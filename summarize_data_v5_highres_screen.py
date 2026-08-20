"""Summarize the fixed-budget Data_v5 1024px screen."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from run_data_v5_highres_screen import GROUPS, PROJECT, experiment_name
from run_feature_ablations import read_progress


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "Data_v5_yolo_highres_screen"


def main() -> None:
    config = json.loads((ROOT / "configs" / "feature_ablation_groups_v5.json").read_text(encoding="utf-8"))["groups"]
    rows = []
    for group in GROUPS:
        directory = PROJECT / experiment_name(group)
        progress = read_progress(directory / "results.csv")
        if not progress:
            raise SystemExit(f"Missing high-resolution result: {group}")
        rows.append({
            "group": group,
            "feature": "RGB" if group == "rgb" else config[group][3],
            **progress,
            "run_dir": str(directory.resolve()),
        })
    baseline = next(row for row in rows if row["group"] == "rgb")
    for row in rows:
        row["delta_vs_rgb"] = row["best_mask_map50_95"] - baseline["best_mask_map50_95"]
    candidates = [row for row in rows if row["group"] != "rgb" and row["delta_vs_rgb"] >= 0.003]
    winner = max(candidates, key=lambda row: row["best_mask_map50_95"]) if candidates else None
    rows.sort(key=lambda row: row["best_mask_map50_95"], reverse=True)
    summary = {
        "version": "data_v5_yolo_highres_screen_summary_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "imgsz": 1024,
        "epochs": 60,
        "rows": rows,
        "formal_training_winner": winner["group"] if winner else None,
        "decision": "promote winner to formal training" if winner else "stop Data_v5 model promotion",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    body = "".join(
        f"<tr><td>{html.escape(row['feature'])}</td><td>{row['best_epoch']}</td>"
        f"<td>{row['best_mask_map50_95']:.4f}</td><td>{row['best_mask_map50']:.4f}</td>"
        f"<td>{row['delta_vs_rgb'] * 100:+.2f} pp</td></tr>"
        for row in rows
    )
    winner_text = winner["feature"] if winner else "无"
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v5 1024 筛选</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:1000px;margin:auto;background:#fff;padding:28px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}</style></head><body><main>
<h1>Data_v5 1024 px 固定预算筛选</h1><p>统一 60 轮、seed 0、batch 2；test 未读取。</p><table><thead><tr><th>方案</th><th>最佳轮次</th><th>Mask mAP50-95</th><th>Mask mAP50</th><th>相对 RGB</th></tr></thead><tbody>{body}</tbody></table><p>正式训练候选：<strong>{html.escape(winner_text)}</strong></p>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
