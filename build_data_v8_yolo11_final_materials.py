"""Build the final Data_v8 YOLO11 decision webpage and valid-only visual evidence."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports" / "Data_v8_yolo11_final_effects"
PAIRED_SUMMARY = ROOT / "reports" / "Data_v8_yolo11_paired_multiseed" / "summary.json"
SOIL_SUMMARY = ROOT / "reports" / "Data_v7_soil_vegetation_screen" / "summary.json"
FUSION_SUMMARY = ROOT / "reports" / "Data_v8_multispectral_fusion_screen" / "summary.json"
OUTPUT = ROOT / "reports" / "Data_v8_yolo11_final_report"
VALID_IMAGES = ROOT / "datasets" / "Data_v3" / "valid" / "images"
DEPLOYMENT_SHA256 = "589A1D5600E58B1B5932A472137BDDC057821DAAC4C8A5F98BB1E3A94715649B"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def resize_width(image: np.ndarray, width: int = 560) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_header(images: list[np.ndarray], lines: list[str]) -> np.ndarray:
    height = min(image.shape[0] for image in images)
    normalized = [
        cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    body = np.hstack(normalized)
    header_height = 92
    header = np.full((header_height, body.shape[1], 3), (248, 250, 249), dtype=np.uint8)
    cv2.putText(header, lines[0], (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (24, 48, 39), 2, cv2.LINE_AA)
    cv2.putText(header, lines[1], (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 103, 82), 2, cv2.LINE_AA)
    return np.vstack((header, body))


def stack_grid(panels: list[np.ndarray], columns: int, gap: int = 18) -> np.ndarray:
    width = max(panel.shape[1] for panel in panels)
    height = max(panel.shape[0] for panel in panels)
    normalized: list[np.ndarray] = []
    for panel in panels:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        normalized.append(canvas)
    while len(normalized) % columns:
        normalized.append(np.full((height, width, 3), 255, dtype=np.uint8))
    rows = []
    spacer_v = np.full((gap, columns * width + (columns - 1) * gap, 3), 255, dtype=np.uint8)
    for offset in range(0, len(normalized), columns):
        row_parts = []
        for index, panel in enumerate(normalized[offset : offset + columns]):
            if index:
                row_parts.append(np.full((height, gap, 3), 255, dtype=np.uint8))
            row_parts.append(panel)
        rows.append(np.hstack(row_parts))
        rows.append(spacer_v)
    return np.vstack(rows[:-1])


def metric_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["image"]: row for row in rows}


def n(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def batch_paths(source: Path, image_name: str, names: list[str]) -> tuple[Path, Path]:
    index = names.index(image_name)
    return source / f"val_batch{index}_labels.jpg", source / f"val_batch{index}_pred.jpg"


def deployment_panel(image_name: str, row: dict[str, str], names: list[str]) -> np.ndarray:
    label, pred = batch_paths(SOURCE / "deployment", image_name, names)
    panel = add_header(
        [resize_width(read_image(label)), resize_width(read_image(pred))],
        [
            f"{image_name} | deployment",
            f"GT / prediction | F1 {n(row, 'mask_f1'):.3f} | FP {int(n(row, 'mask_fp'))} | FN {int(n(row, 'mask_fn'))}",
        ],
    )
    return panel


def comparison_panel(
    image_name: str,
    c0: dict[str, str],
    c1: dict[str, str],
    names: list[str],
) -> np.ndarray:
    label, c0_pred = batch_paths(SOURCE / "seed1_c0", image_name, names)
    _, c1_pred = batch_paths(SOURCE / "seed1_c1", image_name, names)
    delta = n(c1, "mask_f1") - n(c0, "mask_f1")
    panel = add_header(
        [
            resize_width(read_image(label), 500),
            resize_width(read_image(c0_pred), 500),
            resize_width(read_image(c1_pred), 500),
        ],
        [
            f"{image_name} | GT / C0 indices / C1 raw bands",
            f"F1 {n(c0, 'mask_f1'):.3f} -> {n(c1, 'mask_f1'):.3f} | delta {delta:+.3f}",
        ],
    )
    return panel


def html_deployment_cards(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f'''<article class="case">
          <div><span class="tag {html.escape(str(row['class']))}">{html.escape(str(row['diagnosis']))}</span>
          <h3>{html.escape(str(row['image']))}</h3>
          <p>F1 {float(row['f1']):.3f} · Precision {float(row['precision']):.3f} · Recall {float(row['recall']):.3f} · FP {int(row['fp'])} · FN {int(row['fn'])}</p></div>
          <a href="assets/{html.escape(str(row['asset']))}" target="_blank"><img src="assets/{html.escape(str(row['asset']))}" alt="{html.escape(str(row['image']))} ground truth and deployment prediction"></a>
        </article>'''
        for row in rows
    )


def html_comparison_cards(rows: list[dict[str, object]]) -> str:
    return "\n".join(
        f'''<article class="case">
          <div><span class="tag {html.escape(str(row['class']))}">{html.escape(str(row['diagnosis']))}</span>
          <h3>{html.escape(str(row['image']))}</h3>
          <p>C0 F1 {float(row['c0_f1']):.3f} → C1 F1 {float(row['c1_f1']):.3f}（{float(row['delta']):+.3f}）；FP {int(row['fp_delta']):+d}，FN {int(row['fn_delta']):+d}</p></div>
          <a href="assets/{html.escape(str(row['asset']))}" target="_blank"><img src="assets/{html.escape(str(row['asset']))}" alt="{html.escape(str(row['image']))} GT C0 C1 comparison"></a>
        </article>'''
        for row in rows
    )


def render_html(summary: dict) -> str:
    pairs = summary["paired_results"]
    pair_rows = "".join(
        f"<tr><td>{row['seed']}</td><td>{float(row['c0_mask_map50_95']):.5f}</td>"
        f"<td>{float(row['c1_mask_map50_95']):.5f}</td>"
        f"<td class='{'good' if float(row['delta_mask_map50_95']) > 0 else 'bad'}'>"
        f"{float(row['delta_mask_map50_95']) * 100:+.3f} pp</td></tr>"
        for row in pairs
    )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>项目最终进度与模型效果总结</title><meta name="description" content="YOLO11多光谱地块实例分割项目最终阶段总结、特征消融、门控结论和最新valid效果图">
<style>
:root{{--ink:#173129;--muted:#5d7168;--navy:#133f34;--teal:#23765d;--green:#237a4d;--red:#b23a32;--gold:#9b6a00;--bg:#f3f7f5;--paper:#fff;--line:#d8e4de;--soft:#e9f3ee;--warn:#fff4d9}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.65}}
.shell{{width:min(1180px,calc(100% - 36px));margin:auto}}.hero{{padding:62px 0 42px;color:white;background:linear-gradient(125deg,#102f28,#185343 58%,#24725c)}}
.eyebrow{{letter-spacing:.15em;font-weight:800;color:#9ed8c5;font-size:.78rem}}h1{{font-size:clamp(2rem,5vw,4rem);line-height:1.1;margin:10px 0}}.lede{{max-width:820px;color:#d9eee6}}
.nav{{position:sticky;top:0;z-index:10;background:#f9fbfaee;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}}.nav div{{display:flex;gap:18px;padding:10px 0;overflow:auto;white-space:nowrap}}.nav a{{color:var(--muted);text-decoration:none;font-weight:700;font-size:.88rem}}
main{{padding:26px 0 70px}}section{{margin-top:24px;scroll-margin-top:60px}}.panel,.metric,.case{{background:var(--paper);border:1px solid var(--line);border-radius:15px;box-shadow:0 7px 24px #15382c0c}}
.decision{{padding:24px;border-left:6px solid var(--gold)}}.decision strong{{font-size:1.28rem}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:15px}}.metric{{padding:17px}}.metric span{{color:var(--muted);font-size:.8rem}}.metric b{{display:block;font-size:1.65rem;color:var(--navy)}}
h2{{color:var(--navy);border-bottom:2px solid var(--teal);padding-bottom:7px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:720px}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left}}th{{background:var(--navy);color:white}}.good{{color:var(--green);font-weight:800}}.bad{{color:var(--red);font-weight:800}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}.card{{padding:20px}}.card h3{{margin-top:0}}.note{{background:var(--warn);border-left:5px solid var(--gold);padding:14px 16px}}.case{{overflow:hidden;margin:15px 0}}.case>div{{padding:15px 18px 7px}}.case h3{{margin:8px 0 2px}}.case p{{margin:0 0 7px;color:var(--muted)}}.case img{{display:block;width:100%}}
.tag{{display:inline-block;padding:3px 9px;border-radius:999px;font-size:.74rem;font-weight:800;background:var(--soft);color:var(--teal)}}.tag.risk{{background:#ffebe8;color:var(--red)}}.tag.gain{{background:#e7f5eb;color:var(--green)}}
.figure{{padding:14px}}.figure img{{width:100%;display:block;border-radius:10px}}.figure figcaption{{padding:10px 3px 2px;color:var(--muted)}}code{{overflow-wrap:anywhere}}footer{{text-align:center;padding:24px;color:var(--muted)}}
@media(max-width:800px){{.metrics,.split{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.metrics,.split{{grid-template-columns:1fr}}}}
</style></head><body>
<header class="hero"><div class="shell"><div class="eyebrow">FINAL MODEL REVIEW / 2026-08-17</div><h1>项目最终进度与模型效果总结</h1><p class="lede">多光谱无人机地块实例分割：当前模型机制、植被/土壤特征、三 seed 配对证据、失败位置和最新 valid 标注效果。</p></div></header>
<nav class="nav"><div class="shell"><a href="#decision">结论</a><a href="#model">模型机制</a><a href="#paired">三 Seed</a><a href="#features">特征结论</a><a href="#effects">部署效果</a><a href="#failure">C1失败位置</a><a href="#next">下一步</a></div></nav>
<main class="shell"><section id="decision"><div class="panel decision"><strong>结论：保留当前部署模型，拒绝 C1 原始波段替换。</strong><p>C1（RGB+NIR+RedEdge）三 seed 平均 Mask mAP50-95 变化为 {float(summary['mean_delta'])*100:+.3f} pp；Seed 1 退化 {float(summary['worst_delta'])*100:+.3f} pp，未满足平均提升和最差 seed 两项门控。部署权重未修改。</p></div>
<div class="metrics"><div class="metric"><span>现部署结构</span><b>YOLO11n-seg</b><small>3×CoordAtt + 3×ECA</small></div><div class="metric"><span>部署输入</span><b>5 通道</b><small>RGB + NDVI + NDRE</small></div><div class="metric"><span>C1 平均变化</span><b>{float(summary['mean_delta'])*100:+.3f} pp</b><small>未达 +0.300 pp 门槛</small></div><div class="metric"><span>部署哈希</span><b>未改变</b><small>589A…649B</small></div></div></section>
<section id="model"><h2>当前模型、特征与机制</h2><div class="split"><article class="panel card"><h3>正式部署模型</h3><p><strong>YOLO11n-seg</strong>，并不是 YOLO11 与 YOLOv8 的推理混合。YOLOv8 仅用于低成本结构和特征筛选。</p><p>输入为 R、G、B、NDVI、NDRE。NDVI 和 NDRE 都是植被指数，当前模型已经包含植被信息。</p></article><article class="panel card"><h3>注意力与训练机制</h3><p>在三个骨干尺度各插入一组 CoordAtt 与 ECA，共 <strong>3×CoordAtt + 3×ECA</strong>。CoordAtt 强化方向与位置关系，ECA 做轻量通道重标定。</p><p>同时使用分组划分、教师精炼、三 seed 配对、困难样本逐图诊断和冻结 test 门控。</p></article></div></section>
<section id="paired"><h2>YOLO11 三 Seed 配对结果</h2><div class="panel table-wrap"><table><thead><tr><th>Seed</th><th>C0：RGB+NDVI+NDRE</th><th>C1：RGB+NIR+RedEdge</th><th>C1−C0</th></tr></thead><tbody>{pair_rows}</tbody></table></div><p class="note">2/3 seed 为正只通过了方向计数；平均变化接近零，且最差 seed 为 -1.336 pp，说明 C1 对随机初始化敏感，不能稳定替代植被指数。</p></section>
<section id="features"><h2>土壤、植被与融合机制结论</h2><div class="panel table-wrap"><table><thead><tr><th>方案</th><th>证据</th><th>结论</th></tr></thead><tbody>
<tr><td>NDVI + NDRE</td><td>当前部署模型持续使用</td><td class="good">保留；是已验证的植被指数输入</td></tr>
<tr><td>RE_BSI_PROXY 土壤代理</td><td>Seed-0 筛选较五通道参考 -5.832 pp</td><td class="bad">未晋级；P4M 无 SWIR，不能称为标准 BSI</td></tr>
<tr><td>MCARI 植被指数</td><td>Seed-0 筛选较五通道参考 -5.723 pp</td><td class="bad">未晋级；不加入部署</td></tr>
<tr><td>土壤代理 + MCARI</td><td>Seed-0 筛选较五通道参考 -5.923 pp</td><td class="bad">组合未证明互补收益</td></tr>
<tr><td>C1 原始 NIR + RedEdge</td><td>YOLO11 三 seed 平均 -0.006 pp，最差 -1.336 pp</td><td class="bad">拒绝替换 NDVI + NDRE</td></tr>
<tr><td>P4 双分支 / 多尺度门控</td><td>F1 比 C1 仅 +0.004 pp；F2 比 C1 -0.211 pp</td><td class="bad">未通过融合门控，F3 已停止</td></tr>
<tr><td>坡度</td><td>没有严格对齐 DEM/DSM/DTM</td><td>未训练、不可评价，不写成“效果差”</td></tr></tbody></table></div></section>
<section id="effects"><h2>当前部署模型最新 valid 标注效果</h2><p>以下为 confidence 0.25 的逐图诊断：既展示强样本，也展示混合背景、低覆盖和空标注风险；不使用冻结 test。</p><figure class="panel figure"><a href="assets/deployment_overview.jpg" target="_blank"><img src="assets/deployment_overview.jpg" alt="Deployment model valid effect overview"></a><figcaption>人工标注（GT）与当前部署模型预测对比概览。</figcaption></figure>{html_deployment_cards(summary['deployment_cases'])}</section>
<section id="failure"><h2>C1 原始波段方案：效果好与效果差的位置</h2><p>Seed 1 是配对门控中的主要失败 seed。GT/C0/C1 三列显示：C1 对部分低覆盖困难图减少误检，但在多张密集或边缘场景中增加漏检和误检，整体不稳定。</p><figure class="panel figure"><a href="assets/c1_gate_overview.jpg" target="_blank"><img src="assets/c1_gate_overview.jpg" alt="C1 gate representative comparison"></a><figcaption>代表性退化与改善各一例；完整六例见下方。</figcaption></figure>{html_comparison_cards(summary['comparison_cases'])}</section>
<section id="next"><h2>下一轮计划</h2><div class="split"><article class="panel card"><h3>P0：边缘与密集场景</h3><p>优先复核 DJI_3230、DJI_0030、DJI_1790 等退化图的道路/设施边缘、相邻实例分离和漏检标签；只在人工确认后修订训练数据。</p></article><article class="panel card"><h3>P1：低风险机制优化</h3><p>保留 RGB+NDVI+NDRE 和现有注意力，转向边界损失、困难负样本采样或标注质量提升；不再扩大原始指数通道和未证实的双分支结构。</p></article></div><p class="note">冻结 test 未读取；部署模型禁止自动写入。只有后续候选通过预注册 valid 门控后，才允许进入最终独立评估。</p></section>
</main><footer><div class="shell">生成于 2026-08-17 · train/valid-only 研究证据 · 部署 SHA256：<code>{DEPLOYMENT_SHA256}</code></div></footer></body></html>'''


def main() -> int:
    for path in (SOURCE, PAIRED_SUMMARY, SOIL_SUMMARY, FUSION_SUMMARY, VALID_IMAGES):
        if not path.exists():
            raise FileNotFoundError(path)
    paired = json.loads(PAIRED_SUMMARY.read_text(encoding="utf-8"))
    if paired.get("held_out_test_used") is not False or paired.get("gate_passed") is not False:
        raise RuntimeError("Expected a completed, failed, valid-only paired gate")
    names = [path.name for path in sorted(VALID_IMAGES.glob("*.JPG"))]
    if len(names) != 60:
        raise RuntimeError(f"Expected 60 valid images, found {len(names)}")

    deployment = metric_map(read_csv(SOURCE / "deployment" / "per_image_metrics.csv"))
    c0 = metric_map(read_csv(SOURCE / "seed1_c0" / "per_image_metrics.csv"))
    c1 = metric_map(read_csv(SOURCE / "seed1_c1" / "per_image_metrics.csv"))
    if set(deployment) != set(c0) or set(c0) != set(c1) or set(c0) != set(names):
        raise RuntimeError("Per-image validation sets do not match")

    deployment_specs = [
        ("DJI_2350.JPG", "strong", "强结果"),
        ("DJI_1790.JPG", "strong", "密集场景较好"),
        ("DJI_1500.JPG", "risk", "混合背景"),
        ("DJI_2260.JPG", "risk", "低覆盖"),
        ("DJI_2330.JPG", "risk", "空标注误检审计"),
        ("DJI_2490.JPG", "strong", "较高召回"),
    ]
    comparison_specs = [
        ("DJI_3230.JPG", "risk", "最大退化"),
        ("DJI_0030.JPG", "risk", "误检与漏检增加"),
        ("DJI_1790.JPG", "risk", "密集场景退化"),
        ("DJI_1500.JPG", "gain", "困难图改善"),
        ("DJI_2260.JPG", "gain", "低覆盖改善"),
        ("DJI_2340.JPG", "gain", "边界场景改善"),
    ]

    assets = OUTPUT / "assets"
    deployment_cases: list[dict[str, object]] = []
    deployment_panels: list[np.ndarray] = []
    for image_name, css, diagnosis in deployment_specs:
        row = deployment[image_name]
        panel = deployment_panel(image_name, row, names)
        asset = f"deployment_{Path(image_name).stem}.jpg"
        write_image(assets / asset, panel)
        deployment_panels.append(panel)
        deployment_cases.append(
            {
                "image": image_name,
                "class": css,
                "diagnosis": diagnosis,
                "f1": n(row, "mask_f1"),
                "precision": n(row, "mask_precision"),
                "recall": n(row, "mask_recall"),
                "fp": int(n(row, "mask_fp")),
                "fn": int(n(row, "mask_fn")),
                "asset": asset,
            }
        )
    write_image(assets / "deployment_overview.jpg", stack_grid(deployment_panels[:4], 2))

    comparison_cases: list[dict[str, object]] = []
    comparison_panels: list[np.ndarray] = []
    for image_name, css, diagnosis in comparison_specs:
        c0_row, c1_row = c0[image_name], c1[image_name]
        panel = comparison_panel(image_name, c0_row, c1_row, names)
        asset = f"c1_compare_{Path(image_name).stem}.jpg"
        write_image(assets / asset, panel)
        comparison_panels.append(panel)
        comparison_cases.append(
            {
                "image": image_name,
                "class": css,
                "diagnosis": diagnosis,
                "c0_f1": n(c0_row, "mask_f1"),
                "c1_f1": n(c1_row, "mask_f1"),
                "delta": n(c1_row, "mask_f1") - n(c0_row, "mask_f1"),
                "fp_delta": int(n(c1_row, "mask_fp") - n(c0_row, "mask_fp")),
                "fn_delta": int(n(c1_row, "mask_fn") - n(c0_row, "mask_fn")),
                "asset": asset,
            }
        )
    write_image(assets / "c1_gate_overview.jpg", stack_grid([comparison_panels[0], comparison_panels[3]], 1))

    deployment_metrics = json.loads((SOURCE / "deployment" / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "version": "data_v8_yolo11_final_materials_v1",
        "created_at": "2026-08-17",
        "decision": paired["decision"],
        "gate_passed": paired["gate_passed"],
        "mean_delta": paired["mean_delta_mask_map50_95"],
        "worst_delta": paired["worst_seed_delta"],
        "paired_results": paired["paired_results"],
        "gate_checks": paired["gate_checks"],
        "deployment_valid_diagnostic": deployment_metrics,
        "deployment_cases": deployment_cases,
        "comparison_cases": comparison_cases,
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "deployment_sha256": DEPLOYMENT_SHA256,
        "deployment_unchanged": True,
        "slope": "not_evaluated_missing_aligned_dem_dsm_dtm",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "index.html").write_text(render_html(summary), encoding="utf-8")
    print(f"report={OUTPUT / 'index.html'}")
    print(f"deployment_overview={assets / 'deployment_overview.jpg'}")
    print(f"c1_overview={assets / 'c1_gate_overview.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
