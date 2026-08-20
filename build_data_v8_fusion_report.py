"""Build the validation-only Data_v8 fusion decision and failure webpage."""

from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path

from build_sparse_mixed_difficult_review import scene_features
from multispectral import PROJECT_ROOT


ANALYSIS = PROJECT_ROOT / "reports" / "Data_v8_fusion_failure_analysis"
SCREEN_SUMMARY = PROJECT_ROOT / "reports" / "Data_v8_multispectral_fusion_screen" / "summary.json"
OUTPUT = PROJECT_ROOT / "reports" / "Data_v8_multispectral_fusion_report"
IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images"
LABELS = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "labels"
QUEUE = PROJECT_ROOT / "reports" / "Data_v6_sparse_mixed_difficult_review" / "review_queue.csv"
CONTROL = "c0_current_indices_early"
GROUPS = ("c1_raw_bands_early", "f1_p4_concat", "f2_multiscale_gated")
LABELS_BY_GROUP = {
    CONTROL: "C0：RGB + NDVI + NDRE",
    "c1_raw_bands_early": "C1：RGB + NIR + RedEdge 早期融合",
    "f1_p4_concat": "F1：P4 单点双分支融合",
    "f2_multiscale_gated": "F2：P3/P4/P5 门控融合",
}
SLICE_ORDER = ("稀疏/低覆盖", "道路/设施边缘", "明暗混合", "裸土/混凝土混合")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def scene_types(image_name: str, queue_reasons: dict[str, str]) -> tuple[list[str], dict[str, float | int]]:
    image = IMAGES / image_name
    label = LABELS / f"{Path(image_name).stem}.txt"
    values = scene_features(image, label)
    reasons = queue_reasons.get(image_name, "")
    result: list[str] = []
    if float(values["foreground_coverage"]) < 0.15 or "稀疏" in reasons:
        result.append("稀疏/低覆盖")
    if float(values["background_edge_density"]) >= 0.15 or any(word in reasons for word in ("道路", "设施")):
        result.append("道路/设施边缘")
    if float(values["background_luma_std"]) >= 0.18 or "明暗" in reasons:
        result.append("明暗混合")
    if any(word in reasons for word in ("裸土", "混凝土", "混合背景")):
        result.append("裸土/混凝土混合")
    if int(values["polygon_count"]) == 0:
        result.append("空标签/纯背景")
    return list(dict.fromkeys(result)), values


def compare(
    reference_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    group: str,
    queue_reasons: dict[str, str],
) -> list[dict[str, object]]:
    reference = {row["image"]: row for row in reference_rows}
    candidate = {row["image"]: row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise RuntimeError(f"Image mismatch for {group}")
    order = {name: index for index, name in enumerate(sorted(reference))}
    records = []
    for image_name in sorted(reference):
        base = reference[image_name]
        trial = candidate[image_name]
        slices, features = scene_types(image_name, queue_reasons)
        fp_delta = int(round(number(trial, "mask_fp") - number(base, "mask_fp")))
        fn_delta = int(round(number(trial, "mask_fn") - number(base, "mask_fn")))
        reasons = list(slices) or ["一般场景"]
        if fp_delta > 0:
            reasons.append("误检 FP 增加")
        if fn_delta > 0:
            reasons.append("漏检 FN 增加")
        records.append(
            {
                "group": group,
                "image": image_name,
                "batch_index": order[image_name],
                "control_f1": number(base, "mask_f1"),
                "candidate_f1": number(trial, "mask_f1"),
                "f1_delta": number(trial, "mask_f1") - number(base, "mask_f1"),
                "precision_delta": number(trial, "mask_precision") - number(base, "mask_precision"),
                "recall_delta": number(trial, "mask_recall") - number(base, "mask_recall"),
                "fp_delta": fp_delta,
                "fn_delta": fn_delta,
                "polygon_count": int(features["polygon_count"]),
                "foreground_coverage": float(features["foreground_coverage"]),
                "scene_types": "|".join(slices),
                "reasons": "、".join(dict.fromkeys(reasons)),
            }
        )
    return sorted(records, key=lambda row: (float(row["f1_delta"]), -int(row["fp_delta"]), str(row["image"])))


def summarize_slices(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for name in SLICE_ORDER:
        selected = [row for row in records if name in str(row["scene_types"]).split("|")]
        deltas = [float(row["f1_delta"]) for row in selected]
        output.append(
            {
                "slice": name,
                "images": len(selected),
                "mean_f1_delta": statistics.fmean(deltas) if deltas else None,
                "median_f1_delta": statistics.median(deltas) if deltas else None,
                "degraded_images": sum(delta < 0 for delta in deltas),
                "improved_images": sum(delta > 0 for delta in deltas),
            }
        )
    return output


def metric_rows(screen: dict) -> dict[str, dict]:
    rows = screen["controls"] + screen["candidates"]
    return {row["group"]: row for row in rows}


def render(summary: dict) -> str:
    metrics = summary["metrics"]
    metric_table = []
    for group in (CONTROL, *GROUPS):
        row = metrics[group]
        metric_table.append(
            f"<tr><td>{html.escape(LABELS_BY_GROUP[group])}</td>"
            f"<td>{float(row['mask_map50_95']):.5f}</td>"
            f"<td>{float(row['mask_precision']):.5f}</td>"
            f"<td>{float(row['mask_recall']):.5f}</td>"
            f"<td>{int(row['total_parameters']):,}</td></tr>"
        )

    sections = []
    for group in GROUPS:
        row = metrics[group]
        cases = summary["worst_cases"][group]
        cards = []
        for case in cases:
            index = int(case["batch_index"])
            cards.append(
                f'''<article class="case"><h3>{html.escape(str(case["image"]))}</h3>
<p><strong>逐图F1 {float(case["control_f1"]):.3f} → {float(case["candidate_f1"]):.3f}
（{float(case["f1_delta"]):+.3f}）</strong><br>{html.escape(str(case["reasons"]))}<br>
FP {int(case["fp_delta"]):+d}，FN {int(case["fn_delta"]):+d}，前景覆盖 {float(case["foreground_coverage"])*100:.1f}%</p>
<div class="triplet">
<figure><img src="../Data_v8_fusion_failure_analysis/{CONTROL}/val_batch{index}_labels.jpg"><figcaption>人工标签 GT</figcaption></figure>
<figure><img src="../Data_v8_fusion_failure_analysis/{CONTROL}/val_batch{index}_pred.jpg"><figcaption>{html.escape(LABELS_BY_GROUP[CONTROL])}</figcaption></figure>
<figure><img src="../Data_v8_fusion_failure_analysis/{group}/val_batch{index}_pred.jpg"><figcaption>{html.escape(LABELS_BY_GROUP[group])}</figcaption></figure>
</div></article>'''
            )
        slices = []
        for item in summary["slice_summary_vs_c0"][group]:
            median = item["median_f1_delta"]
            median_text = "无样本" if median is None else f"{float(median):+.4f}"
            slices.append(
                f"<tr><td>{html.escape(str(item['slice']))}</td><td>{int(item['images'])}</td>"
                f"<td>{median_text}</td><td>{int(item['improved_images'])}/{int(item['degraded_images'])}</td></tr>"
            )
        delta_c0 = float(row["mask_map50_95"]) - float(metrics[CONTROL]["mask_map50_95"])
        delta_c1 = float(row["mask_map50_95"]) - float(metrics["c1_raw_bands_early"]["mask_map50_95"])
        sections.append(
            f'''<section><h2>{html.escape(LABELS_BY_GROUP[group])}</h2>
<p>Mask mAP50-95 相对C0：{delta_c0*100:+.3f} pp；相对C1：{delta_c1*100:+.3f} pp。</p>
<table><thead><tr><th>困难切片</th><th>图像数</th><th>F1中位变化（vs C0）</th><th>改善/退化</th></tr></thead><tbody>{''.join(slices)}</tbody></table>
{''.join(cards)}</section>'''
        )
    decision = summary["decision"]
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data_v8 多光谱融合结果与失败定位</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f2f6f4;color:#17231e;margin:0}}main{{max-width:1280px;margin:24px auto;padding:0 18px 60px}}header,section{{background:white;border:1px solid #dce5df;border-radius:12px;padding:24px;margin-bottom:18px}}.notice{{background:#fff5da;border-left:5px solid #d49a18;padding:12px}}.good{{background:#eaf7ee;border-left:5px solid #2f855a;padding:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e3e9e5;text-align:left}}.case{{border-top:1px solid #e3e9e5;padding:18px 0}}.triplet{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;border-radius:7px}}figcaption{{padding:6px;color:#52625a}}@media(max-width:850px){{.triplet{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>Data_v8 多光谱融合：结果与失败位置</h1>
<p>只使用train/valid；冻结test未读取。所有模型统一YOLOv8n-seg、seed 0、30轮、1024、batch 2、attention=none。</p>
<p class="notice"><strong>双分支结论：</strong>F1仅比C1高0.004 pp，F2比C1低0.211 pp，均未证明双分支融合有效；F3按门控停止。</p>
<p class="good"><strong>C1结论：</strong>C1相对C0提高0.318 pp，且没有增加参数。困难切片门控：{html.escape(str(decision['c1_difficult_gate']))}。</p>
<p><strong>下一步：</strong>{html.escape(str(decision['next_step']))}</p>
<p>坡度仍因缺少对齐DEM/DSM/DTM而不可评估；部署权重保持不变。</p></header>
<section><h2>总体指标</h2><table><thead><tr><th>组</th><th>Mask mAP50-95</th><th>Precision</th><th>Recall</th><th>参数量</th></tr></thead><tbody>{''.join(metric_table)}</tbody></table></section>
{''.join(sections)}
<section><h2>完整数据</h2><p><a href="per_image_degradation.csv">逐图退化CSV</a> · <a href="summary.json">机器可读总结</a></p></section>
</main></body></html>'''


def main() -> None:
    screen = json.loads(SCREEN_SUMMARY.read_text(encoding="utf-8"))
    if screen.get("held_out_test_used") is not False:
        raise RuntimeError("Screen summary is not valid-only")
    queue_reasons = {}
    if QUEUE.is_file():
        queue_reasons = {row["image"]: row.get("reasons", "") for row in read_csv(QUEUE)}
    base_rows = read_csv(ANALYSIS / CONTROL / "per_image_metrics.csv")
    metrics = metric_rows(screen)
    comparisons = {
        group: compare(
            base_rows,
            read_csv(ANALYSIS / group / "per_image_metrics.csv"),
            group,
            queue_reasons,
        )
        for group in GROUPS
    }
    slice_summaries = {group: summarize_slices(records) for group, records in comparisons.items()}
    c1_slices = slice_summaries["c1_raw_bands_early"]
    positive_slices = sum(
        item["median_f1_delta"] is not None and float(item["median_f1_delta"]) > 0
        for item in c1_slices
    )
    empty_rows = [row for row in comparisons["c1_raw_bands_early"] if int(row["polygon_count"]) == 0]
    empty_fp_delta = sum(int(row["fp_delta"]) for row in empty_rows)
    c1 = metrics["c1_raw_bands_early"]
    c0 = metrics[CONTROL]
    aggregate_gate = bool(
        float(c1["mask_map50_95"]) - float(c0["mask_map50_95"]) >= 0.003
        and float(c1["mask_precision"]) - float(c0["mask_precision"]) >= -0.01
        and float(c1["mask_recall"]) - float(c0["mask_recall"]) >= -0.01
    )
    difficult_gate = bool(positive_slices >= 3 and empty_fp_delta <= 0)
    promote_c1 = aggregate_gate and difficult_gate
    summary = {
        "version": "data_v8_multispectral_fusion_failure_report_v1",
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "metrics": metrics,
        "slice_summary_vs_c0": slice_summaries,
        "worst_cases": {group: records[:5] for group, records in comparisons.items()},
        "decision": {
            "dual_branch_numeric_gate_passed": False,
            "f3_allowed": False,
            "c1_aggregate_gate": aggregate_gate,
            "c1_positive_difficult_slices": positive_slices,
            "c1_empty_background_fp_delta": empty_fp_delta,
            "c1_difficult_gate": difficult_gate,
            "promote_c1_to_yolo11_three_seed": promote_c1,
            "next_step": (
                "将C1原始5波段早期融合与C0当前指数输入移植到YOLO11+CoordAtt/ECA，进行三seed配对确认。"
                if promote_c1
                else "停止本轮融合路线，保留部署模型并转向困难负样本与边界优化。"
            ),
        },
        "slope": {
            "status": "not_evaluated_missing_aligned_elevation_source",
            "effect_claimed_bad": False,
        },
        "deployment_sha256": "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rows = [row for group in GROUPS for row in comparisons[group]]
    with (OUTPUT / "per_image_degradation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUTPUT / "index.html").write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))
    print(f"report={OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
