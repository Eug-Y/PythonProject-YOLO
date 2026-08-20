"""Summarize Data_v7 soil/vegetation screens against the sealed five-channel baseline."""

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


PROJECT = PROJECT_ROOT / "runs" / "segment" / "land_seg"
OUTPUT = PROJECT_ROOT / "reports" / "Data_v7_soil_vegetation_screen"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
GROUPS = ("add_re_bsi_proxy_6ch", "add_mcari_6ch", "soil_vegetation_7ch")
BASELINE = {
    "run": "v6_final_reviewed_teacher_refined_yolov8n_long_seed0_v2",
    "best_epoch": 45,
    "mask_precision": 0.68283,
    "mask_recall": 0.56966,
    "mask_map50": 0.58089,
    "mask_map50_95": 0.36968,
}


def experiment_name(group: str, seed: int = 0) -> str:
    return f"v7_{group}_yolov8n_none_1024_seed{seed}_screen_v1"


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(
    project: Path,
    groups: tuple[str, ...],
    *,
    seed: int = 0,
    promotion_delta: float = 0.003,
) -> dict:
    rows: list[dict[str, object]] = []
    for group in groups:
        name = experiment_name(group, seed)
        progress = read_progress(project / name / "results.csv")
        checkpoint = project / name / "weights" / "best.pt"
        if progress is None or not checkpoint.is_file():
            rows.append({"group": group, "seed": seed, "run": name, "status": "missing", "promoted": False})
            continue
        score = float(progress["best_mask_map50_95"])
        precision = float(progress["best_mask_precision"])
        recall = float(progress["best_mask_recall"])
        delta = score - float(BASELINE["mask_map50_95"])
        rows.append(
            {
                "group": group,
                "seed": seed,
                "run": name,
                "status": "complete",
                **progress,
                "checkpoint": str(checkpoint.resolve()),
                "delta_vs_5ch": delta,
                "delta_vs_5ch_pp": delta * 100.0,
                "precision_delta_vs_5ch": precision - float(BASELINE["mask_precision"]),
                "recall_delta_vs_5ch": recall - float(BASELINE["mask_recall"]),
                "promoted": delta >= promotion_delta,
            }
        )
    complete = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(complete, key=lambda row: float(row["best_mask_map50_95"]), reverse=True)
    promoted = [row for row in ranking if row["promoted"]]
    deployment_hash = file_sha256(DEPLOYMENT)
    return {
        "version": "data_v7_soil_vegetation_screen_summary_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "slope_included": False,
        "soil_index": "RE_BSI_PROXY (experimental RedEdge bare-soil proxy; not standard SWIR-BSI)",
        "vegetation_index": "MCARI (DJI P4M band-adapted)",
        "seed": seed,
        "promotion_delta_mask_map50_95": promotion_delta,
        "promotion_delta_pp": promotion_delta * 100.0,
        "baseline": BASELINE,
        "groups": list(groups),
        "rows": rows,
        "ranking": ranking,
        "promoted": promoted,
        "recommended_next": promoted[0] if promoted else None,
        "decision": "promote_best_to_three_seed" if promoted else "retain_current_5ch",
        "complete_groups": len(complete),
        "expected_groups": len(groups),
        "deployment_eligible": False,
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
    }


def render_html(summary: dict) -> str:
    body_rows = []
    for row in summary["rows"]:
        if row["status"] != "complete":
            body_rows.append(
                f'<tr><td>{html.escape(str(row["group"]))}</td><td colspan="6">缺少结果</td></tr>'
            )
            continue
        decision = "进入三 seed" if row["promoted"] else "不晋级"
        css = "pass" if row["promoted"] else "hold"
        body_rows.append(
            "<tr>"
            f'<td>{html.escape(str(row["group"]))}</td>'
            f'<td>{int(row["best_epoch"])}</td>'
            f'<td>{float(row["best_mask_precision"]):.4f}</td>'
            f'<td>{float(row["best_mask_recall"]):.4f}</td>'
            f'<td>{float(row["best_mask_map50"]):.4f}</td>'
            f'<td>{float(row["best_mask_map50_95"]):.4f} ({float(row["delta_vs_5ch_pp"]):+.2f} pp)</td>'
            f'<td class="{css}">{decision}</td>'
            "</tr>"
        )
    recommended = summary["recommended_next"]
    recommendation = (
        f'建议下一阶段：{html.escape(str(recommended["group"]))} 做 seed 0/1/2 确认。'
        if recommended
        else "当前没有组合达到 +0.30 pp 初筛门槛，继续保留五通道模型。"
    )
    baseline = summary["baseline"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v7 土壤与植被指数初筛</title><style>
body{{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f4f7f5;color:#17231e;margin:28px}}main{{max-width:1050px;margin:auto;background:#fff;padding:26px;border:1px solid #dce6df;border-radius:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e5ebe7;text-align:left}}th{{background:#edf4ef}}.pass{{color:#08765a;font-weight:700}}.hold{{color:#a04a38}}.note{{padding:12px;background:#edf7f2;border-left:4px solid #08765a}}
</style></head><body><main><h1>Data_v7 土壤与植被指数初筛</h1>
<p class="note">{recommendation}</p>
<p>五通道 seed-0 基线 Mask mAP50-95：{float(baseline['mask_map50_95']):.5f}。土壤通道为非标准 RE_BSI_PROXY；植被通道为 P4M 适配 MCARI。本轮不含坡度。</p>
<table><thead><tr><th>组合</th><th>最佳轮</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95 / 增益</th><th>结论</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table>
<p>仅使用 train/valid；冻结 test 未读取。部署权重未改写：{html.escape(str(summary['deployment_unchanged']))}。初筛通过也不具备直接部署资格。</p>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--promotion-delta", type=float, default=0.003)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.promotion_delta < 0:
        raise ValueError("--promotion-delta must be non-negative")
    summary = summarize(
        args.project.resolve(),
        tuple(args.groups),
        seed=args.seed,
        promotion_delta=args.promotion_delta,
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
    return 0 if summary["complete_groups"] == summary["expected_groups"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
