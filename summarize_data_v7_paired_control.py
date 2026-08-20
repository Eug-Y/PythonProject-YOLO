"""Compare Data_v7 candidates with the exact 30-epoch, 1024px five-channel control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from multispectral import PROJECT_ROOT
from run_feature_ablations import read_progress
from run_data_v7_soil_vegetation_screen import DEPLOYMENT, DEPLOYMENT_SHA256, GROUPS, experiment_name


PROJECT = PROJECT_ROOT / "runs" / "segment" / "land_seg"
CONTROL_NAME = "v7_paired_current5_yolov8n_none_1024_seed0_control_v1"
OUTPUT = PROJECT_ROOT / "reports" / "Data_v7_paired_control"


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_paired(
    project: Path,
    groups: tuple[str, ...] = GROUPS,
    *,
    promotion_delta: float = 0.003,
    max_recall_regression: float = 0.01,
) -> dict:
    control = read_progress(project / CONTROL_NAME / "results.csv")
    control_checkpoint = project / CONTROL_NAME / "weights" / "best.pt"
    control_complete = control is not None and control_checkpoint.is_file()
    rows: list[dict[str, object]] = []
    for group in groups:
        name = experiment_name(group, 0)
        candidate = read_progress(project / name / "results.csv")
        checkpoint = project / name / "weights" / "best.pt"
        row: dict[str, object] = {"group": group, "control_run": CONTROL_NAME, "candidate_run": name}
        if not control_complete or candidate is None or not checkpoint.is_file():
            row.update(
                status="missing_control" if not control_complete else "missing_candidate",
                promoted=False,
            )
        else:
            delta = float(candidate["best_mask_map50_95"]) - float(control["best_mask_map50_95"])
            precision_delta = float(candidate["best_mask_precision"]) - float(control["best_mask_precision"])
            recall_delta = float(candidate["best_mask_recall"]) - float(control["best_mask_recall"])
            row.update(
                status="complete",
                control_mask_map50_95=float(control["best_mask_map50_95"]),
                candidate_mask_map50_95=float(candidate["best_mask_map50_95"]),
                control_precision=float(control["best_mask_precision"]),
                candidate_precision=float(candidate["best_mask_precision"]),
                control_recall=float(control["best_mask_recall"]),
                candidate_recall=float(candidate["best_mask_recall"]),
                delta=delta,
                delta_pp=delta * 100.0,
                precision_delta=precision_delta,
                recall_delta=recall_delta,
                promoted=delta >= promotion_delta and recall_delta >= -max_recall_regression,
                checkpoint=str(checkpoint.resolve()),
            )
        rows.append(row)
    complete = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(complete, key=lambda row: float(row["delta"]), reverse=True)
    promoted = [row for row in ranking if row["promoted"]]
    deployment_hash = file_sha256(DEPLOYMENT)
    return {
        "version": "data_v7_exact_paired_control_summary_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "control_contract": {
            "run": CONTROL_NAME,
            "channels": ["R", "G", "B", "NDVI", "NDRE"],
            "model": "yolov8n-seg.pt",
            "attention": "none",
            "seed": 0,
            "epochs": 30,
            "imgsz": 1024,
            "batch": 2,
            "teacher_refined_train": True,
        },
        "control_complete": control_complete,
        "control_metrics": control,
        "promotion_delta": promotion_delta,
        "max_recall_regression": max_recall_regression,
        "rows": rows,
        "ranking": ranking,
        "promoted": promoted,
        "recommended_next": promoted[0] if promoted else None,
        "decision": "promote_best_to_three_seed" if promoted else ("retain_current_5ch" if control_complete else "incomplete"),
        "deployment_eligible": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
    }


def render_html(summary: dict) -> str:
    body = []
    for row in summary["rows"]:
        if row["status"] != "complete":
            body.append(f'<tr><td>{html.escape(str(row["group"]))}</td><td colspan="6">{row["status"]}</td></tr>')
            continue
        decision = "晋级" if row["promoted"] else "不晋级"
        body.append(
            f'<tr><td>{html.escape(str(row["group"]))}</td>'
            f'<td>{float(row["control_mask_map50_95"]):.5f}</td>'
            f'<td>{float(row["candidate_mask_map50_95"]):.5f}</td>'
            f'<td>{float(row["delta_pp"]):+.2f} pp</td>'
            f'<td>{float(row["precision_delta"])*100:+.2f} pp</td>'
            f'<td>{float(row["recall_delta"])*100:+.2f} pp</td><td>{decision}</td></tr>'
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v7 严格配对对照</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f4f7f5;margin:28px}}main{{max-width:1000px;margin:auto;background:white;padding:26px;border-radius:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e1e8e3;text-align:left}}th{{background:#edf4ef}}</style></head><body><main>
<h1>Data_v7 严格配对对照</h1><p>所有组合统一为 YOLOv8n、seed 0、30 轮、1024、batch 2、无注意力、同一初始化和教师精炼 train。冻结 test 未读取。</p>
<table><thead><tr><th>候选</th><th>配对 5ch</th><th>候选</th><th>mAP 差</th><th>Precision 差</th><th>Recall 差</th><th>门控</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<p>结论：{html.escape(str(summary['decision']))}。初筛晋级不等于部署资格。</p></main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--promotion-delta", type=float, default=0.003)
    parser.add_argument("--max-recall-regression", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.promotion_delta < 0 or args.max_recall_regression < 0:
        raise ValueError("Thresholds must be non-negative")
    summary = summarize_paired(
        args.project.resolve(),
        promotion_delta=args.promotion_delta,
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
        with (output / "paired.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(complete[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(complete)
    print(f"Report: {output / 'index.html'}")
    print(f"Decision: {summary['decision']}")
    return 0 if summary["control_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
