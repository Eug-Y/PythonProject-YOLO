"""Summarize paired Data_v6 terrain-factor screens without touching the test split."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from multispectral import PROJECT_ROOT
from run_feature_ablations import read_progress
from train_ca_5ch import SUPPORTED_SEGMENTATION_MODELS


DEFAULT_MODELS = ("yolo11n-seg.pt", "yolov8n-seg.pt", "yolo12n-seg.pt", "yolo26n-seg.pt")
DEFAULT_CANDIDATE_GROUPS = ("slope_6ch",)


def experiment_name(group: str, model: str, seed: int, attention: str = "none") -> str:
    return f"v6_{group}_{Path(model).stem}_{attention}_seed{seed}_v1"


def summarize_pairs(
    project: Path,
    *,
    models: tuple[str, ...],
    candidate_groups: tuple[str, ...],
    seed: int,
    promotion_delta: float,
    attention: str = "none",
) -> dict:
    rows: list[dict[str, object]] = []
    for model in models:
        control_name = experiment_name("current_5ch", model, seed, attention)
        control = read_progress(project / control_name / "results.csv")
        for group in candidate_groups:
            candidate_name = experiment_name(group, model, seed, attention)
            candidate = read_progress(project / candidate_name / "results.csv")
            row: dict[str, object] = {
                "model": model,
                "seed": seed,
                "control_group": "current_5ch",
                "candidate_group": group,
                "control_run": control_name,
                "candidate_run": candidate_name,
            }
            if control is None or candidate is None:
                row.update(
                    status="missing_control" if control is None else "missing_candidate",
                    promoted=False,
                )
            else:
                control_score = float(control["best_mask_map50_95"])
                candidate_score = float(candidate["best_mask_map50_95"])
                delta = candidate_score - control_score
                row.update(
                    status="complete",
                    control_best_epoch=int(control["best_epoch"]),
                    candidate_best_epoch=int(candidate["best_epoch"]),
                    control_mask_map50_95=control_score,
                    candidate_mask_map50_95=candidate_score,
                    delta=delta,
                    delta_pp=delta * 100.0,
                    promoted=delta >= promotion_delta,
                )
            rows.append(row)

    complete = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(complete, key=lambda row: float(row["delta"]), reverse=True)
    promoted = [row for row in ranking if row["promoted"]]
    return {
        "version": "data_v6_terrain_model_screen_summary_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "seed": seed,
        "attention": attention,
        "promotion_delta_mask_map50_95": promotion_delta,
        "promotion_delta_pp": promotion_delta * 100.0,
        "models": list(models),
        "candidate_groups": list(candidate_groups),
        "rows": rows,
        "ranking": ranking,
        "promoted": promoted,
        "recommended_next": promoted[0] if promoted else None,
        "complete_pairs": len(complete),
        "expected_pairs": len(rows),
    }


def render_html(summary: dict) -> str:
    body_rows = []
    for row in summary["rows"]:
        if row["status"] == "complete":
            control = f'{float(row["control_mask_map50_95"]):.4f}'
            candidate = f'{float(row["candidate_mask_map50_95"]):.4f}'
            delta = f'{float(row["delta_pp"]):+.2f}'
            decision = "晋级" if row["promoted"] else "不晋级"
            css = "pass" if row["promoted"] else "fail"
        else:
            control = candidate = delta = "—"
            decision = str(row["status"])
            css = "missing"
        body_rows.append(
            "<tr>"
            f'<td>{html.escape(str(row["model"]))}</td>'
            f'<td>{html.escape(str(row["candidate_group"]))}</td>'
            f"<td>{control}</td><td>{candidate}</td><td>{delta} pp</td>"
            f'<td class="{css}">{html.escape(decision)}</td>'
            "</tr>"
        )
    next_item = summary["recommended_next"]
    recommendation = (
        f'下一阶段：{html.escape(str(next_item["model"]))} + '
        f'{html.escape(str(next_item["candidate_group"]))}，先做三随机种子验证。'
        if next_item
        else "当前没有组合达到晋级阈值；保持现有 5 通道方案。"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v6 地形模型筛选</title><style>
body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:28px;color:#172126;background:#f3f6f6}}
main{{max-width:980px;margin:auto;background:white;padding:24px;border:1px solid #d8e0e3;border-radius:8px}}
table{{width:100%;border-collapse:collapse;margin-top:18px}}th,td{{padding:10px;border-bottom:1px solid #e3e8ea;text-align:left}}
th{{background:#eef3f3}}.pass{{color:#087a62;font-weight:700}}.fail{{color:#a34135}}.missing{{color:#7b858a}}
.note{{padding:12px;background:#edf7f4;border-left:4px solid #087a62}}small{{color:#607078}}
</style></head><body><main><h1>Data_v6 地形模型筛选</h1>
<p class="note">{recommendation}</p>
<p>晋级阈值：Mask mAP50-95 提升至少 {float(summary["promotion_delta_pp"]):.2f} pp。仅使用 train/valid，test 未读取。</p>
<table><thead><tr><th>模型</th><th>地形组合</th><th>5ch</th><th>候选</th><th>增益</th><th>结论</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
<p><small>完成配对：{summary["complete_pairs"]}/{summary["expected_pairs"]}</small></p>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "segment" / "land_seg")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_SEGMENTATION_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--candidate-groups", nargs="+", default=DEFAULT_CANDIDATE_GROUPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", choices=("none", "ca_eca"), default="none")
    parser.add_argument("--promotion-delta", type=float, default=0.003)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "Data_v6_terrain_model_screen")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.promotion_delta < 0:
        raise ValueError("--promotion-delta must be non-negative")
    summary = summarize_pairs(
        args.project.resolve(),
        models=tuple(args.models),
        candidate_groups=tuple(args.candidate_groups),
        seed=args.seed,
        promotion_delta=args.promotion_delta,
        attention=args.attention,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "summary.json")
    (output / "index.html").write_text(render_html(summary), encoding="utf-8")
    print(f"Report: {output / 'index.html'}")
    print(f"Complete pairs: {summary['complete_pairs']}/{summary['expected_pairs']}")
    return 0 if summary["complete_pairs"] == summary["expected_pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
