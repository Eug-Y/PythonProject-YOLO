"""Summarize the Data_v8 YOLO11 C0/C1 paired confirmation and apply its gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_feature_ablations import read_progress


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "runs" / "segment" / "land_seg"
OUTPUT = ROOT / "reports" / "Data_v8_yolo11_paired_multiseed"
DEPLOYMENT = PROJECT / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
GROUPS = ("c0_current_indices_early", "c1_raw_bands_early")
LABELS = {
    "c0_current_indices_early": "C0：RGB + NDVI + NDRE",
    "c1_raw_bands_early": "C1：RGB + NIR + RedEdge",
}
T95 = {2: 4.3026527297}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(group: str, seed: int) -> str:
    return f"v8_yolo11_{group}_ca_eca_1024_seed{seed}_confirm_v1"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_std(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def paired_ci95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    center = statistics.fmean(values)
    if len(values) == 1:
        return center, center
    critical = T95.get(len(values) - 1)
    if critical is None:
        return None, None
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return center - margin, center + margin


def summarize(seeds: tuple[int, ...]) -> dict:
    runs: list[dict[str, object]] = []
    by_seed: dict[int, dict[str, dict[str, object]]] = {seed: {} for seed in seeds}
    for seed in seeds:
        for group in GROUPS:
            name = run_name(group, seed)
            run_dir = PROJECT / name
            progress = read_progress(run_dir / "results.csv")
            checkpoint = run_dir / "weights" / "best.pt"
            if not progress or not checkpoint.is_file():
                row: dict[str, object] = {
                    "seed": seed,
                    "group": group,
                    "label": LABELS[group],
                    "run": name,
                    "status": "missing",
                }
            else:
                row = {
                    "seed": seed,
                    "group": group,
                    "label": LABELS[group],
                    "run": name,
                    "status": "complete",
                    **progress,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            runs.append(row)
            by_seed[seed][group] = row

    paired: list[dict[str, object]] = []
    for seed in seeds:
        c0 = by_seed[seed][GROUPS[0]]
        c1 = by_seed[seed][GROUPS[1]]
        if c0["status"] != "complete" or c1["status"] != "complete":
            paired.append({"seed": seed, "status": "missing_pair"})
            continue
        paired.append(
            {
                "seed": seed,
                "status": "complete",
                "c0_mask_map50_95": float(c0["best_mask_map50_95"]),
                "c1_mask_map50_95": float(c1["best_mask_map50_95"]),
                "delta_mask_map50_95": float(c1["best_mask_map50_95"]) - float(c0["best_mask_map50_95"]),
                "delta_mask_precision": float(c1["best_mask_precision"]) - float(c0["best_mask_precision"]),
                "delta_mask_recall": float(c1["best_mask_recall"]) - float(c0["best_mask_recall"]),
            }
        )

    complete_pairs = [row for row in paired if row["status"] == "complete"]
    map_deltas = [float(row["delta_mask_map50_95"]) for row in complete_pairs]
    precision_deltas = [float(row["delta_mask_precision"]) for row in complete_pairs]
    recall_deltas = [float(row["delta_mask_recall"]) for row in complete_pairs]
    complete = len(complete_pairs) == len(seeds)
    mean_map_delta = mean(map_deltas)
    mean_precision_delta = mean(precision_deltas)
    mean_recall_delta = mean(recall_deltas)
    ci_low, ci_high = paired_ci95(map_deltas)
    gate_checks = {
        "all_three_pairs_complete": complete and len(seeds) == 3,
        "mean_map_delta_at_least_0_003": bool(complete and mean_map_delta is not None and mean_map_delta >= 0.003),
        "at_least_two_positive_seeds": bool(complete and sum(value > 0 for value in map_deltas) >= 2),
        "worst_seed_not_below_minus_0_003": bool(complete and map_deltas and min(map_deltas) >= -0.003),
        "mean_precision_regression_within_0_01": bool(
            complete and mean_precision_delta is not None and mean_precision_delta >= -0.01
        ),
        "mean_recall_regression_within_0_01": bool(
            complete and mean_recall_delta is not None and mean_recall_delta >= -0.01
        ),
    }
    gate_passed = all(gate_checks.values())
    c1_complete = [
        row for row in runs if row["group"] == GROUPS[1] and row["status"] == "complete"
    ]
    best_c1 = max(c1_complete, key=lambda row: float(row["best_mask_map50_95"])) if c1_complete else None
    deployment_hash = sha256(DEPLOYMENT)
    if not complete:
        decision = "incomplete"
    elif gate_passed:
        decision = "freeze_c1_candidate_for_review_no_auto_deployment"
    else:
        decision = "reject_c1_retain_current_deployment"

    return {
        "version": "data_v8_yolo11_paired_multiseed_summary_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "experiment": {
            "architecture": "YOLO11n-seg",
            "attention": "3x CoordAtt + 3x ECA",
            "control": LABELS[GROUPS[0]],
            "candidate": LABELS[GROUPS[1]],
            "paired_initialization": True,
            "seeds": list(seeds),
            "epochs": 60,
            "patience": 20,
            "imgsz": 1024,
            "batch": 2,
        },
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "slope_included": False,
        "runs": runs,
        "paired_results": paired,
        "complete_pairs": len(complete_pairs),
        "mean_delta_mask_map50_95": mean_map_delta,
        "std_delta_mask_map50_95": sample_std(map_deltas),
        "paired_delta_ci95_low": ci_low,
        "paired_delta_ci95_high": ci_high,
        "positive_seeds": sum(value > 0 for value in map_deltas),
        "worst_seed_delta": min(map_deltas) if map_deltas else None,
        "mean_delta_mask_precision": mean_precision_delta,
        "mean_delta_mask_recall": mean_recall_delta,
        "gate_thresholds": {
            "mean_delta_mask_map50_95_min": 0.003,
            "positive_seed_count_min": 2,
            "worst_seed_delta_min": -0.003,
            "mean_precision_delta_min": -0.01,
            "mean_recall_delta_min": -0.01,
        },
        "gate_checks": gate_checks,
        "gate_passed": gate_passed,
        "decision": decision,
        "candidate_checkpoint": str(best_c1["checkpoint"]) if gate_passed and best_c1 else None,
        "candidate_selection_rule": "highest C1 validation Mask mAP50-95 after group-level paired gate",
        "deployment_eligible": False,
        "deployment_write_allowed": False,
        "deployment_checkpoint": str(DEPLOYMENT.resolve()),
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
        "prior_screen_failure_context": {
            "architecture": "YOLOv8n-seg screening only; not an inference ensemble",
            "f1_p4_gain_vs_c1_pp": 0.004,
            "f2_multiscale_gated_delta_vs_c1_pp": -0.211,
            "f3_stopped": True,
            "c1_known_weak_slice": "road/facility edges",
            "c1_known_regressions": [
                {"image": "DJI_1780.JPG", "f1": "0.621 -> 0.550", "fp_delta": 7, "fn_delta": 3},
                {"image": "DJI_0800.JPG", "f1": "0.561 -> 0.500", "fp_delta": 8, "fn_delta": 2},
            ],
            "slope": "not evaluated because no strictly aligned DEM/DSM/DTM is available",
        },
    }


def fmt(value: object, digits: int = 5) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_html(summary: dict) -> str:
    pair_rows = []
    for row in summary["paired_results"]:
        if row["status"] != "complete":
            pair_rows.append(f"<tr><td>{row['seed']}</td><td colspan='6'>未完成</td></tr>")
            continue
        delta = float(row["delta_mask_map50_95"])
        css = "gain" if delta > 0 else "loss" if delta < 0 else ""
        pair_rows.append(
            "<tr>"
            f"<td>{row['seed']}</td>"
            f"<td>{float(row['c0_mask_map50_95']):.5f}</td>"
            f"<td>{float(row['c1_mask_map50_95']):.5f}</td>"
            f"<td class='{css}'>{delta * 100:+.3f} pp</td>"
            f"<td>{float(row['delta_mask_precision']) * 100:+.3f} pp</td>"
            f"<td>{float(row['delta_mask_recall']) * 100:+.3f} pp</td>"
            "</tr>"
        )
    checks = []
    check_labels = {
        "all_three_pairs_complete": "三组配对均完成",
        "mean_map_delta_at_least_0_003": "平均 Mask mAP50-95 ≥ +0.300 pp",
        "at_least_two_positive_seeds": "至少 2/3 seed 为正",
        "worst_seed_not_below_minus_0_003": "最差 seed 不低于 -0.300 pp",
        "mean_precision_regression_within_0_01": "平均 Precision 回退不超过 1 pp",
        "mean_recall_regression_within_0_01": "平均 Recall 回退不超过 1 pp",
    }
    for key, passed in summary["gate_checks"].items():
        checks.append(
            f"<li class='{'ok' if passed else 'bad'}'>{'通过' if passed else '未通过'}："
            f"{html.escape(check_labels[key])}</li>"
        )
    decision_text = {
        "incomplete": "训练尚未完成，暂不作模型结论。",
        "freeze_c1_candidate_for_review_no_auto_deployment": "C1 通过配对门控，仅冻结为候选；不自动部署。",
        "reject_c1_retain_current_deployment": "C1 未通过配对门控，保留现有部署模型。",
    }[summary["decision"]]
    prior = summary["prior_screen_failure_context"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v8 YOLO11 三 Seed 配对确认</title>
<style>
:root{{--ink:#17231e;--muted:#5d6b64;--line:#dce6df;--paper:#fff;--bg:#f2f6f3;--green:#247a4a;--red:#b53b32;--gold:#a46b00}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.6}}
main{{max-width:1120px;margin:28px auto;padding:0 18px 60px}}header,section{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:24px;margin-bottom:18px}}
h1,h2{{line-height:1.25}}.decision{{border-left:5px solid var(--gold);background:#fff7df;padding:14px 16px}}.facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}
.fact{{background:#f7faf8;border:1px solid var(--line);border-radius:10px;padding:14px}}.fact strong{{display:block;font-size:1.35rem}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}th{{background:#f2f7f4}}.gain,.ok{{color:var(--green);font-weight:700}}.loss,.bad{{color:var(--red);font-weight:700}}
.warning{{border-left:5px solid var(--red);background:#fff0ed;padding:14px 16px}}code{{overflow-wrap:anywhere}}@media(max-width:760px){{.facts{{grid-template-columns:1fr 1fr}}table{{font-size:.88rem}}}}
</style></head><body><main>
<header><p>Data_v8 · train/valid only · test 冻结</p><h1>YOLO11 C0/C1 三 Seed 配对确认</h1>
<p class="decision"><strong>{html.escape(decision_text)}</strong><br>正式部署权重未修改：{summary['deployment_unchanged']}。</p>
<div class="facts"><div class="fact"><span>平均 Δ mAP50-95</span><strong>{fmt(summary['mean_delta_mask_map50_95'])}</strong></div>
<div class="fact"><span>正向 seed</span><strong>{summary['positive_seeds']}/3</strong></div>
<div class="fact"><span>最差 seed</span><strong>{fmt(summary['worst_seed_delta'])}</strong></div>
<div class="fact"><span>配对 95% CI</span><strong>{fmt(summary['paired_delta_ci95_low'],4)}～{fmt(summary['paired_delta_ci95_high'],4)}</strong></div></div></header>
<section><h2>模型与输入</h2><p>两组均为 <strong>YOLO11n-seg + 3×CoordAtt + 3×ECA</strong>，1024、batch 2、最多 60 轮、patience 20。C0 输入 RGB + NDVI + NDRE；C1 输入 RGB + NIR + RedEdge。每个 seed 的两组从同一预训练权重和相同确定性随机种子初始化。</p>
<p>坡度未加入：缺少严格空间对齐的 DEM/DSM/DTM，不能评价其好坏。</p></section>
<section><h2>逐 Seed 结果</h2><table><thead><tr><th>Seed</th><th>C0 mAP50-95</th><th>C1 mAP50-95</th><th>C1−C0</th><th>Δ Precision</th><th>Δ Recall</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></section>
<section><h2>部署门控</h2><ul>{''.join(checks)}</ul><p>只有全部条件同时通过，C1 才能冻结为待评审候选；本脚本无部署写权限。</p></section>
<section><h2>已知效果不佳位置</h2><div class="warning"><strong>这些是前置 YOLOv8 结构筛选证据，不能冒充本轮 YOLO11 最终逐图结论。</strong></div>
<p>道路/设施边缘仍是 C1 的主要风险：DJI_1780.JPG 的 F1 由 0.621 降至 0.550（FP +7、FN +3）；DJI_0800.JPG 由 0.561 降至 0.500（FP +8、FN +2）。</p>
<p>双分支机制没有证明收益：P4 双分支仅比 C1 高 {prior['f1_p4_gain_vs_c1_pp']:+.3f} pp；多尺度门控比 C1 低 {prior['f2_multiscale_gated_delta_vs_c1_pp']:+.3f} pp；F3 已按门控停止。</p></section>
<section><h2>审计</h2><p>冻结 test 未读取。部署 SHA256：<code>{html.escape(str(summary['deployment_sha256_current']))}</code>。机器可读数据：<a href="summary.json">summary.json</a> · <a href="paired_results.csv">paired_results.csv</a>。</p></section>
</main></body></html>"""


def write_outputs(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "seed",
            "status",
            "c0_mask_map50_95",
            "c1_mask_map50_95",
            "delta_mask_map50_95",
            "delta_mask_precision",
            "delta_mask_recall",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["paired_results"])
    (output / "index.html").write_text(render_html(summary), encoding="utf-8")
    readme = (
        "# Data_v8 YOLO11 三 Seed 配对确认\n\n"
        f"- 决策：{summary['decision']}\n"
        f"- 完成配对：{summary['complete_pairs']}/3\n"
        f"- 平均 Mask mAP50-95 变化：{summary['mean_delta_mask_map50_95']}\n"
        f"- 正向 seed：{summary['positive_seeds']}/3\n"
        f"- 门控通过：{summary['gate_passed']}\n"
        "- test：冻结、未读取\n"
        f"- 部署权重未修改：{summary['deployment_unchanged']}\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    seeds = tuple(cli.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    summary = summarize(seeds)
    write_outputs(summary, cli.output.resolve())
    print(json.dumps({
        "complete_pairs": summary["complete_pairs"],
        "gate_passed": summary["gate_passed"],
        "decision": summary["decision"],
        "deployment_unchanged": summary["deployment_unchanged"],
    }, ensure_ascii=False, indent=2))
    print(f"report={cli.output.resolve() / 'index.html'}")
    return 0 if summary["complete_pairs"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
