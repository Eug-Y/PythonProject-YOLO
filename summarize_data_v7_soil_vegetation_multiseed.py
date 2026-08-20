"""Summarize three-seed Data_v7 confirmation without reading the frozen test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from multispectral import PROJECT_ROOT
from run_feature_ablations import read_progress
from run_data_v7_soil_vegetation_screen import DEPLOYMENT, DEPLOYMENT_SHA256, GROUPS, experiment_name


PROJECT = PROJECT_ROOT / "runs" / "segment" / "land_seg"
OUTPUT = PROJECT_ROOT / "reports" / "Data_v7_soil_vegetation_multiseed"
BASELINE_RUNS = {
    seed: f"v6_final_reviewed_teacher_refined_yolov8n_long_seed{seed}_v2" for seed in (0, 1, 2)
}


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_multiseed(
    project: Path,
    candidate_group: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    *,
    promotion_delta: float = 0.003,
    max_seed_regression: float = 0.003,
    max_recall_regression: float = 0.01,
) -> dict:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        baseline_name = BASELINE_RUNS.get(seed)
        candidate_name = experiment_name(candidate_group, seed)
        baseline = read_progress(project / baseline_name / "results.csv") if baseline_name else None
        candidate = read_progress(project / candidate_name / "results.csv")
        candidate_checkpoint = project / candidate_name / "weights" / "best.pt"
        row: dict[str, object] = {
            "seed": seed,
            "baseline_run": baseline_name,
            "candidate_run": candidate_name,
        }
        if baseline is None or candidate is None or not candidate_checkpoint.is_file():
            row.update(
                status="missing_baseline" if baseline is None else "missing_candidate",
                positive=False,
            )
        else:
            delta = float(candidate["best_mask_map50_95"]) - float(baseline["best_mask_map50_95"])
            precision_delta = float(candidate["best_mask_precision"]) - float(baseline["best_mask_precision"])
            recall_delta = float(candidate["best_mask_recall"]) - float(baseline["best_mask_recall"])
            row.update(
                status="complete",
                baseline_mask_map50_95=float(baseline["best_mask_map50_95"]),
                candidate_mask_map50_95=float(candidate["best_mask_map50_95"]),
                baseline_precision=float(baseline["best_mask_precision"]),
                candidate_precision=float(candidate["best_mask_precision"]),
                baseline_recall=float(baseline["best_mask_recall"]),
                candidate_recall=float(candidate["best_mask_recall"]),
                delta=delta,
                delta_pp=delta * 100.0,
                precision_delta=precision_delta,
                recall_delta=recall_delta,
                positive=delta > 0.0,
                candidate_checkpoint=str(candidate_checkpoint.resolve()),
            )
        rows.append(row)

    complete = [row for row in rows if row["status"] == "complete"]
    deltas = [float(row["delta"]) for row in complete]
    candidate_scores = [float(row["candidate_mask_map50_95"]) for row in complete]
    baseline_scores = [float(row["baseline_mask_map50_95"]) for row in complete]
    recall_deltas = [float(row["recall_delta"]) for row in complete]
    required_positive = 2 if len(seeds) == 3 else max(1, (len(seeds) + 1) // 2)
    positive_count = sum(bool(row["positive"]) for row in complete)
    all_complete = len(complete) == len(seeds)
    mean_delta = statistics.mean(deltas) if deltas else None
    worst_delta = min(deltas) if deltas else None
    mean_recall_delta = statistics.mean(recall_deltas) if recall_deltas else None
    stable = bool(
        all_complete
        and mean_delta is not None
        and mean_delta >= promotion_delta
        and positive_count >= required_positive
        and worst_delta is not None
        and worst_delta >= -max_seed_regression
        and mean_recall_delta is not None
        and mean_recall_delta >= -max_recall_regression
    )
    deployment_hash = file_sha256(DEPLOYMENT)
    return {
        "version": "data_v7_soil_vegetation_multiseed_summary_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "slope_included": False,
        "teacher_student_pipeline": "YOLO11 teacher-refined labels -> YOLOv8n student",
        "candidate_group": candidate_group,
        "seeds": list(seeds),
        "rows": rows,
        "complete_seeds": len(complete),
        "candidate_mean_mask_map50_95": statistics.mean(candidate_scores) if candidate_scores else None,
        "candidate_std_mask_map50_95": statistics.stdev(candidate_scores) if len(candidate_scores) > 1 else 0.0 if candidate_scores else None,
        "baseline_mean_mask_map50_95": statistics.mean(baseline_scores) if baseline_scores else None,
        "baseline_std_mask_map50_95": statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.0 if baseline_scores else None,
        "mean_delta": mean_delta,
        "mean_delta_pp": mean_delta * 100.0 if mean_delta is not None else None,
        "worst_delta": worst_delta,
        "worst_delta_pp": worst_delta * 100.0 if worst_delta is not None else None,
        "mean_recall_delta": mean_recall_delta,
        "mean_recall_delta_pp": mean_recall_delta * 100.0 if mean_recall_delta is not None else None,
        "required_positive_seeds": required_positive,
        "positive_seeds": positive_count,
        "promotion_delta": promotion_delta,
        "max_seed_regression": max_seed_regression,
        "max_recall_regression": max_recall_regression,
        "stable_candidate": stable,
        "decision": "promote_to_frozen_candidate_review" if stable else ("incomplete" if not all_complete else "reject_unstable"),
        "deployment_eligible": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
    }


def render_html(summary: dict) -> str:
    rows = []
    for row in summary["rows"]:
        if row["status"] != "complete":
            rows.append(f'<tr><td>{row["seed"]}</td><td colspan="7">{html.escape(str(row["status"]))}</td></tr>')
            continue
        result = "正增益" if row["positive"] else "回退"
        css = "pass" if row["positive"] else "fail"
        rows.append(
            f'<tr><td>{row["seed"]}</td>'
            f'<td>{float(row["baseline_mask_map50_95"]):.5f}</td>'
            f'<td>{float(row["candidate_mask_map50_95"]):.5f}</td>'
            f'<td>{float(row["delta_pp"]):+.2f} pp</td>'
            f'<td>{float(row["candidate_precision"]):.4f}</td>'
            f'<td>{float(row["candidate_recall"]):.4f}</td>'
            f'<td>{float(row["recall_delta"])*100:+.2f} pp</td>'
            f'<td class="{css}">{result}</td></tr>'
        )
    stable = summary["stable_candidate"]
    decision = "通过三 seed 稳定性门控，进入冻结候选人工复核" if stable else "未通过稳定性门控，保留现有五通道方案"
    mean_delta = "—" if summary["mean_delta_pp"] is None else f'{float(summary["mean_delta_pp"]):+.2f} pp'
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v7 三 seed 确认</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f4f7f5;color:#17231e;margin:28px}}main{{max-width:1050px;margin:auto;background:#fff;padding:26px;border:1px solid #dce6df;border-radius:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e5ebe7;text-align:left}}th{{background:#edf4ef}}.pass{{color:#08765a;font-weight:700}}.fail{{color:#a04a38}}</style></head><body><main>
<h1>Data_v7 三 seed 确认</h1><p><strong>{decision}</strong></p>
<p>候选：{html.escape(str(summary['candidate_group']))}；平均 mAP50-95 增益 {mean_delta}；正增益 seed {summary['positive_seeds']}/{len(summary['seeds'])}。</p>
<table><thead><tr><th>Seed</th><th>5ch</th><th>候选</th><th>mAP 增益</th><th>Precision</th><th>Recall</th><th>Recall 变化</th><th>结论</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>YOLO11 教师细化标签 → YOLOv8n 学生训练。仅使用 train/valid；冻结 test 未读取；坡度未加入；部署权重未改写。</p>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-group", choices=GROUPS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--promotion-delta", type=float, default=0.003)
    parser.add_argument("--max-seed-regression", type=float, default=0.003)
    parser.add_argument("--max-recall-regression", type=float, default=0.01)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    if min(args.promotion_delta, args.max_seed_regression, args.max_recall_regression) < 0:
        raise ValueError("Promotion/regression thresholds must be non-negative")
    summary = summarize_multiseed(
        args.project.resolve(),
        args.candidate_group,
        seeds,
        promotion_delta=args.promotion_delta,
        max_seed_regression=args.max_seed_regression,
        max_recall_regression=args.max_recall_regression,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "summary.json")
    (output / "index.html").write_text(render_html(summary), encoding="utf-8")
    complete = [row for row in summary["rows"] if row["status"] == "complete"]
    if complete:
        with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(complete[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(complete)
    print(f"Report: {output / 'index.html'}")
    print(f"Decision: {summary['decision']}")
    return 0 if summary["complete_seeds"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
