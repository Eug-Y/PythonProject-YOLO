"""Summarize paired three-seed terrain validation and enforce stability rules."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from multispectral import PROJECT_ROOT
from run_feature_ablations import read_progress
from summarize_terrain_model_screen import experiment_name
from train_ca_5ch import SUPPORTED_SEGMENTATION_MODELS


def summarize_multiseed(
    project: Path,
    *,
    model: str,
    candidate_group: str,
    seeds: tuple[int, ...],
    attention: str,
    promotion_delta: float = 0.003,
    max_seed_regression: float = 0.003,
) -> dict:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        control_name = experiment_name("current_5ch", model, seed, attention)
        candidate_name = experiment_name(candidate_group, model, seed, attention)
        control = read_progress(project / control_name / "results.csv")
        candidate = read_progress(project / candidate_name / "results.csv")
        row: dict[str, object] = {
            "seed": seed,
            "control_run": control_name,
            "candidate_run": candidate_name,
        }
        if control is None or candidate is None:
            row.update(
                status="missing_control" if control is None else "missing_candidate",
                positive=False,
            )
        else:
            control_score = float(control["best_mask_map50_95"])
            candidate_score = float(candidate["best_mask_map50_95"])
            delta = candidate_score - control_score
            row.update(
                status="complete",
                control_mask_map50_95=control_score,
                candidate_mask_map50_95=candidate_score,
                delta=delta,
                delta_pp=delta * 100.0,
                positive=delta > 0.0,
            )
        rows.append(row)

    complete = [row for row in rows if row["status"] == "complete"]
    deltas = np.asarray([float(row["delta"]) for row in complete], dtype=np.float64)
    required_positive = max(1, int(np.ceil(len(seeds) * 2.0 / 3.0)))
    positive_count = int(sum(bool(row["positive"]) for row in complete))
    all_complete = len(complete) == len(seeds)
    mean_delta = float(deltas.mean()) if len(deltas) else None
    worst_delta = float(deltas.min()) if len(deltas) else None
    stable = bool(
        all_complete
        and mean_delta is not None
        and mean_delta >= promotion_delta
        and positive_count >= required_positive
        and worst_delta is not None
        and worst_delta >= -max_seed_regression
    )
    return {
        "version": "data_v6_terrain_multiseed_summary_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "model": model,
        "candidate_group": candidate_group,
        "attention": attention,
        "seeds": list(seeds),
        "rows": rows,
        "complete_seeds": len(complete),
        "required_positive_seeds": required_positive,
        "positive_seeds": positive_count,
        "mean_delta": mean_delta,
        "mean_delta_pp": mean_delta * 100.0 if mean_delta is not None else None,
        "worst_delta": worst_delta,
        "worst_delta_pp": worst_delta * 100.0 if worst_delta is not None else None,
        "promotion_delta": promotion_delta,
        "promotion_delta_pp": promotion_delta * 100.0,
        "max_seed_regression": max_seed_regression,
        "max_seed_regression_pp": max_seed_regression * 100.0,
        "stable_for_fusion": stable,
        "decision": "promote_to_fusion" if stable else ("incomplete" if not all_complete else "reject_unstable"),
    }


def render_html(summary: dict) -> str:
    rows = []
    for row in summary["rows"]:
        if row["status"] == "complete":
            delta = f'{float(row["delta_pp"]):+.2f} pp'
            result = "正增益" if row["positive"] else "回退"
            css = "pass" if row["positive"] else "fail"
            control = f'{float(row["control_mask_map50_95"]):.4f}'
            candidate = f'{float(row["candidate_mask_map50_95"]):.4f}'
        else:
            delta = control = candidate = "—"
            result = str(row["status"])
            css = "missing"
        rows.append(
            f'<tr><td>{row["seed"]}</td><td>{control}</td><td>{candidate}</td>'
            f'<td>{delta}</td><td class="{css}">{html.escape(result)}</td></tr>'
        )
    stable = summary["stable_for_fusion"]
    decision = "通过稳定性验证，可进入模型融合" if stable else "未通过稳定性验证，不进入模型融合"
    mean = "—" if summary["mean_delta_pp"] is None else f'{float(summary["mean_delta_pp"]):+.2f} pp'
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v6 三随机种子验证</title>
<style>body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#f3f6f6;color:#172126;margin:28px}}
main{{max-width:880px;margin:auto;background:white;padding:24px;border:1px solid #d8e0e3;border-radius:8px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e3e8ea;text-align:left}}th{{background:#eef3f3}}
.pass{{color:#087a62;font-weight:700}}.fail{{color:#a34135}}.missing{{color:#7b858a}}</style></head>
<body><main><h1>Data_v6 三随机种子验证</h1><h2>{html.escape(summary["model"])} · {html.escape(summary["candidate_group"])}</h2>
<p><strong>{decision}</strong></p><p>平均增益：{mean}；正增益种子：{summary["positive_seeds"]}/{len(summary["seeds"])}。</p>
<table><thead><tr><th>Seed</th><th>5ch</th><th>候选</th><th>增益</th><th>结果</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>仅使用 train/valid；冻结 test 未读取。</p></main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_SEGMENTATION_MODELS, required=True)
    parser.add_argument("--candidate-group", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--attention", choices=("none", "ca_eca"), default="none")
    parser.add_argument("--promotion-delta", type=float, default=0.003)
    parser.add_argument("--max-seed-regression", type=float, default=0.003)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "segment" / "land_seg")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "Data_v6_terrain_multiseed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    if args.promotion_delta < 0 or args.max_seed_regression < 0:
        raise ValueError("Promotion and regression thresholds must be non-negative")
    summary = summarize_multiseed(
        args.project.resolve(),
        model=args.model,
        candidate_group=args.candidate_group,
        seeds=seeds,
        attention=args.attention,
        promotion_delta=args.promotion_delta,
        max_seed_regression=args.max_seed_regression,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "summary.json")
    (output / "index.html").write_text(render_html(summary), encoding="utf-8")
    print(f"Report: {output / 'index.html'}")
    print(f"Decision: {summary['decision']}")
    return 0 if summary["complete_seeds"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())

