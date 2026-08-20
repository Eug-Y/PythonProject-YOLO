"""Build a validation-only webpage that localizes where Data_v7 feature candidates regress."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from build_sparse_mixed_difficult_review import scene_features
from multispectral import PROJECT_ROOT
from run_data_v7_soil_vegetation_screen import GROUPS


ANALYSIS_ROOT = PROJECT_ROOT / "reports" / "Data_v7_failure_analysis"
PAIRED_SUMMARY = PROJECT_ROOT / "reports" / "Data_v7_paired_control" / "summary.json"
OUTPUT = PROJECT_ROOT / "reports" / "Data_v7_feature_failure_report"
IMAGES = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images"
LABELS = PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "labels"
DIFFICULT_QUEUE = PROJECT_ROOT / "reports" / "Data_v6_sparse_mixed_difficult_review" / "review_queue.csv"
CONTROL = "current_5ch"
GROUP_LABELS = {
    "add_re_bsi_proxy_6ch": "土壤代理：RE_BSI_PROXY 6ch",
    "add_mcari_6ch": "植被/叶绿素：MCARI 6ch",
    "soil_vegetation_7ch": "土壤代理 + 植被指数 7ch",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def scene_tags(
    image_name: str,
    image_dir: Path,
    label_dir: Path,
    queue_reasons: dict[str, str],
) -> tuple[list[str], dict[str, float | int]]:
    stem = Path(image_name).stem
    label = label_dir / f"{stem}.txt"
    features = scene_features(image_dir / image_name, label)
    tags: list[str] = []
    if int(features["polygon_count"]) == 0:
        tags.append("空标签/纯背景")
    if float(features["foreground_coverage"]) < 0.15:
        tags.append("稀疏或低覆盖")
    if float(features["background_edge_density"]) >= 0.15:
        tags.append("道路/设施等高边缘背景")
    if float(features["background_luma_std"]) >= 0.18:
        tags.append("明暗混合背景")
    if image_name in queue_reasons:
        tags.extend(part.strip() for part in queue_reasons[image_name].split(",") if part.strip())
    return list(dict.fromkeys(tags)) or ["一般场景"], features


def rank_degradation(
    control_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    group: str,
    image_dir: Path,
    label_dir: Path,
    queue_reasons: dict[str, str],
) -> list[dict[str, object]]:
    control = {row["image"]: row for row in control_rows}
    candidate = {row["image"]: row for row in candidate_rows}
    if set(control) != set(candidate):
        raise ValueError(f"Control/candidate image sets differ for {group}")
    image_order = {name: index for index, name in enumerate(sorted(control))}
    records: list[dict[str, object]] = []
    for image_name in sorted(control):
        base = control[image_name]
        trial = candidate[image_name]
        tags, features = scene_tags(image_name, image_dir, label_dir, queue_reasons)
        f1_delta = number(trial, "mask_f1") - number(base, "mask_f1")
        precision_delta = number(trial, "mask_precision") - number(base, "mask_precision")
        recall_delta = number(trial, "mask_recall") - number(base, "mask_recall")
        fp_delta = int(round(number(trial, "mask_fp") - number(base, "mask_fp")))
        fn_delta = int(round(number(trial, "mask_fn") - number(base, "mask_fn")))
        issue_tags = list(tags)
        if fp_delta > 0:
            issue_tags.append("误检 FP 增加")
        if fn_delta > 0:
            issue_tags.append("漏检 FN 增加")
        records.append({
            "group": group,
            "image": image_name,
            "batch_index": image_order[image_name],
            "control_f1": number(base, "mask_f1"),
            "candidate_f1": number(trial, "mask_f1"),
            "f1_delta": f1_delta,
            "precision_delta": precision_delta,
            "recall_delta": recall_delta,
            "fp_delta": fp_delta,
            "fn_delta": fn_delta,
            "polygon_count": int(features["polygon_count"]),
            "foreground_coverage": float(features["foreground_coverage"]),
            "reasons": "、".join(dict.fromkeys(issue_tags)),
        })
    return sorted(records, key=lambda row: (float(row["f1_delta"]), -int(row["fp_delta"]), str(row["image"])))


def render_html(summary: dict) -> str:
    paired = summary["paired_summary"]
    paired_by_group = {row["group"]: row for row in paired["rows"]}
    sections = []
    for group in GROUPS:
        result = paired_by_group[group]
        cases = summary["worst_cases"][group]
        cards = []
        for case in cases:
            index = int(case["batch_index"])
            cards.append(
                f'''<article class="case"><h3>{html.escape(str(case["image"]))}</h3>
<p><strong>F1 {float(case["control_f1"]):.3f} → {float(case["candidate_f1"]):.3f}
（{float(case["f1_delta"]):+.3f}）</strong><br>{html.escape(str(case["reasons"]))}<br>
FP 变化 {int(case["fp_delta"]):+d}，FN 变化 {int(case["fn_delta"]):+d}，前景覆盖 {float(case["foreground_coverage"])*100:.1f}%</p>
<div class="triplet">
<figure><img src="../Data_v7_failure_analysis/{CONTROL}/val_batch{index}_labels.jpg"><figcaption>人工标签 GT</figcaption></figure>
<figure><img src="../Data_v7_failure_analysis/{CONTROL}/val_batch{index}_pred.jpg"><figcaption>严格配对 5ch</figcaption></figure>
<figure><img src="../Data_v7_failure_analysis/{group}/val_batch{index}_pred.jpg"><figcaption>{html.escape(GROUP_LABELS[group])}</figcaption></figure>
</div></article>'''
            )
        sections.append(
            f'''<section><h2>{html.escape(GROUP_LABELS[group])}</h2>
<p>配对 Mask mAP50-95：5ch {float(result["control_mask_map50_95"]):.5f}，候选 {float(result["candidate_mask_map50_95"]):.5f}，变化 {float(result["delta_pp"]):+.2f} pp；
Precision {float(result["precision_delta"])*100:+.2f} pp；Recall {float(result["recall_delta"])*100:+.2f} pp。</p>
{''.join(cards)}</section>'''
        )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>土壤、植被与坡度特征失败定位</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f6f4;color:#17231e;margin:0}}main{{max-width:1280px;margin:24px auto;padding:0 18px 60px}}header,section{{background:white;border:1px solid #dce5df;border-radius:12px;padding:24px;margin-bottom:18px}}.notice{{background:#fff6df;border-left:5px solid #d49a18;padding:12px}}.case{{border-top:1px solid #e3e9e5;padding:18px 0}}.triplet{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;border-radius:7px}}figcaption{{padding:6px;color:#52625a}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e3e9e5;text-align:left}}@media(max-width:850px){{.triplet{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>土壤、植被与坡度特征：效果与失败位置</h1>
<p>所有判断仅使用 train/valid；冻结 test 未读取。网页按严格同预算 5ch 控制定位候选退化，而不是只展示总体平均值。</p>
<p class="notice"><strong>坡度：</strong>当前没有与航拍图逐像素严格对齐的 DEM/DSM/DTM，因此坡度模型未训练、效果不可评估。这里不会把“缺数据”误写为“效果差”。</p>
<p><strong>总体决策：</strong>{html.escape(str(paired["decision"]))}。部署模型保持不变。</p></header>
{''.join(sections)}
<section><h2>如何阅读失败位置</h2><p>每个样本依次展示人工标签、严格配对 5ch、候选模型。重点查看蓝色掩膜是否在道路/混凝土上扩张（FP），或是否漏掉稀疏作物行和边界（FN）。完整逐图数据见 <a href="per_image_degradation.csv">CSV</a>。</p></section>
</main></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--paired-summary", type=Path, default=PAIRED_SUMMARY)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--difficult-queue", type=Path, default=DIFFICULT_QUEUE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cases-per-group", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cases_per_group < 1:
        raise ValueError("--cases-per-group must be positive")
    paired = json.loads(args.paired_summary.read_text(encoding="utf-8"))
    if not paired.get("control_complete"):
        raise RuntimeError("Exact paired control summary is incomplete")
    queue_reasons = {}
    if args.difficult_queue.is_file():
        queue_reasons = {row["image"]: row.get("reasons", "") for row in read_csv(args.difficult_queue)}
    control_rows = read_csv(args.analysis_root / CONTROL / "per_image_metrics.csv")
    all_records: list[dict[str, object]] = []
    worst_cases: dict[str, list[dict[str, object]]] = {}
    for group in GROUPS:
        records = rank_degradation(
            control_rows,
            read_csv(args.analysis_root / group / "per_image_metrics.csv"),
            group=group,
            image_dir=args.images,
            label_dir=args.labels,
            queue_reasons=queue_reasons,
        )
        all_records.extend(records)
        worst_cases[group] = records[: args.cases_per_group]
    summary = {
        "version": "data_v7_feature_failure_localization_v1",
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "paired_summary": paired,
        "slope": {
            "status": "not_evaluated_missing_aligned_elevation_source",
            "effect_claimed_bad": False,
            "reason": "No pixel-aligned DEM/DSM/DTM is available; RGB/multispectral reflectance cannot substitute for elevation.",
        },
        "worst_cases": worst_cases,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_records:
        with (output / "per_image_degradation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_records[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records)
    (output / "index.html").write_text(render_html(summary), encoding="utf-8")
    print(f"Report: {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
