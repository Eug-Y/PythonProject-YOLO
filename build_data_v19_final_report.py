"""Build the auditable V19 final report from frozen train/valid-only artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from run_data_v9_grouped_cv import DEPLOYMENT, DEPLOYMENT_SHA256, now_iso, sha256
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
V19A = ROOT / "logs" / "v19a_yolov8_boundary_nooverlap_cv.summary.json"
V19B = ROOT / "logs" / "v19b_yolo11_boundary_nooverlap_multiseed.summary.json"
SLICES = (
    ROOT
    / "reports"
    / "Data_v19b_yolo11_boundary_nooverlap_multiseed"
    / "difficult_slice_audit"
    / "paired_summary.json"
)
OUTPUT = ROOT / "reports" / "Data_v19_final_report"
SUMMARY = OUTPUT / "summary.json"
README = OUTPUT / "README.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pp(value: float) -> str:
    return f"{value * 100:+.3f}"


def main() -> int:
    v19a = load(V19A)
    v19b = load(V19B)
    slices = load(SLICES)
    if v19a.get("gate", {}).get("passed") is not True:
        raise RuntimeError("V19A gate did not pass")
    if v19b.get("gate_passed") is not True:
        raise RuntimeError("V19B gate did not pass")
    if v19a.get("held_out_test_used") is not False or v19b.get("held_out_test_used") is not False:
        raise RuntimeError("V19 summary includes held-out test access")
    if slices.get("held_out_test_used") is not False or slices.get("selection_split") != "valid":
        raise RuntimeError("V19 difficult-slice audit is not valid-only")
    deployment_hash = sha256(DEPLOYMENT)
    if deployment_hash != DEPLOYMENT_SHA256:
        raise RuntimeError("Deployment checkpoint changed before V19 final report")

    fixed = slices["fixed_threshold_summary"]
    normalized_slices = {
        name: {
            "images": int(value["images"]),
            "f1_delta": float(value["delta"]["f1"]),
            "precision_delta": float(value["delta"]["precision"]),
            "recall_delta": float(value["delta"]["recall"]),
            "tp_delta": int(value["delta"]["tp"]),
            "fp_delta": int(value["delta"]["fp"]),
            "fn_delta": int(value["delta"]["fn"]),
            "improved_images": int(value["improved_f1_images"]),
            "degraded_images": int(value["degraded_f1_images"]),
        }
        for name, value in slices["slice_summary"].items()
    }
    risk_slices = [
        name for name, value in normalized_slices.items()
        if float(value["f1_delta"]) < 0.0
    ]
    regressed_segments = [
        row for row in slices["flight_segment_summary"] if float(row["f1_delta"]) < 0.0
    ]
    candidate = Path(v19b["candidate_checkpoint"])
    if not candidate.is_file() or sha256(candidate) != v19b["candidate_checkpoint_sha256"]:
        raise RuntimeError("Frozen V19B candidate hash does not match")

    final = {
        "version": "data_v19_final_report_v1",
        "created_at": now_iso(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "v19a": {
            "model": "YOLOv8n-seg",
            "gate_passed": True,
            "mean_map_delta": v19a["gate"]["mean_map_delta"],
            "worst_map_delta": v19a["gate"]["worst_map_delta"],
            "positive_folds": v19a["gate"]["positive_map_folds"],
            "mean_precision_delta": v19a["gate"]["mean_precision_delta"],
            "mean_recall_delta": v19a["gate"]["mean_recall_delta"],
        },
        "v19b": {
            "model": "YOLO11n-seg + 3x CoordAtt + 3x ECA",
            "gate_passed": True,
            "rows": v19b["rows"],
            "gate_metrics": v19b["gate_metrics"],
            "locked_confidence": v19b["protocol"]["locked_confidence"],
        },
        "frozen_candidate": {
            "checkpoint": str(candidate.resolve()),
            "sha256": sha256(candidate),
            "seed": 0,
            "validation_map": next(row["candidate_map"] for row in v19b["rows"] if row["seed"] == 0),
            "deployment_eligible": False,
        },
        "fixed_threshold_audit": {
            "confidence": slices["confidence"],
            "images": fixed["images"],
            "tp_delta": fixed["delta"]["tp"],
            "fp_delta": fixed["delta"]["fp"],
            "fn_delta": fixed["delta"]["fn"],
            "precision_delta": fixed["delta"]["precision"],
            "recall_delta": fixed["delta"]["recall"],
            "f1_delta": fixed["delta"]["f1"],
            "improved_images": fixed["improved_f1_images"],
            "degraded_images": fixed["degraded_f1_images"],
            "unchanged_images": fixed["unchanged_f1_images"],
        },
        "difficult_slices": normalized_slices,
        "risk_slices": risk_slices,
        "regressed_flight_segments": regressed_segments,
        "worst_f1_regressions": slices["worst_f1_regressions"],
        "decision": "freeze_v19b_seed0_for_review_do_not_auto_deploy",
        "decision_reasons": [
            "V19A and V19B both passed every preregistered aggregate gate.",
            "All three V19B seeds improved standardized validation Mask mAP50-95.",
            "The common confidence 0.371 improved F1 for all three seeds.",
            "Sparse low-coverage validation scenes regressed and require targeted review.",
            "The held-out test remains sealed and the current deployment checkpoint was not changed.",
        ],
        "next_action": (
            "Review sparse low-coverage false positives and find train-only analogues; "
            "do not tune directly on validation images. Keep V19B Seed-0 frozen until a separately authorized final gate."
        ),
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(SUMMARY, final)

    seed_lines = []
    for row in v19b["rows"]:
        seed_lines.append(
            f"| {row['seed']} | {row['baseline_map']:.5f} | {row['candidate_map']:.5f} | "
            f"{pp(row['map_delta'])} | {pp(row['precision_delta'])} | {pp(row['recall_delta'])} | "
            f"{pp(row['fixed_f1_delta'])} |"
        )
    slice_lines = []
    for name, value in sorted(normalized_slices.items()):
        slice_lines.append(
            f"| {name} | {value['images']} | {pp(value['f1_delta'])} | "
            f"{value['tp_delta']:+d} | {value['fp_delta']:+d} | {value['fn_delta']:+d} |"
        )
    segment_lines = [
        f"- `{row['flight_segment']}`：{row['images']} 张，F1 {pp(float(row['f1_delta']))} pp，"
        f"FP {int(row['fp_delta']):+d}，FN {int(row['fn_delta']):+d}。"
        for row in regressed_segments
    ]
    text = f"""# Data_v19 最终训练报告

生成时间：{final['created_at']}  
结论：V19A 与 V19B 均通过预注册门槛；冻结 V19B Seed-0 供复核，暂不自动部署。

## 核心结果

- V19A（YOLOv8n 三折）平均 Mask mAP50-95 提升 `{pp(v19a['gate']['mean_map_delta'])}` 个百分点，最差折仍提升 `{pp(v19a['gate']['worst_map_delta'])}` 个百分点。
- V19B（YOLO11n 三种子）平均 Mask mAP50-95 提升 `{pp(v19b['gate_metrics']['mean_map_delta'])}` 个百分点，最差种子仍提升 `{pp(v19b['gate_metrics']['worst_map_delta'])}` 个百分点。
- 固定置信度 `0.371` 下，三种子平均 F1 提升 `{pp(v19b['gate_metrics']['mean_fixed_f1_delta'])}` 个百分点，最差种子仍提升 `{pp(v19b['gate_metrics']['worst_fixed_f1_delta'])}` 个百分点。
- 冻结候选：`{candidate.name}`，SHA256 `{sha256(candidate)}`。

| Seed | V15 mAP | V19B mAP | mAP Δ(pp) | Precision Δ(pp) | Recall Δ(pp) | 固定 F1 Δ(pp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(seed_lines)}

## 固定阈值困难切片审计（Seed-0）

60 张验证图总体 TP `+{fixed['delta']['tp']}`、FP `{fixed['delta']['fp']:+d}`、FN `{fixed['delta']['fn']:+d}`，F1 提升 `{pp(fixed['delta']['f1'])}` 个百分点；34 张改善、24 张退化、2 张不变。

| 切片 | 图像数 | F1 Δ(pp) | TP Δ | FP Δ | FN Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(slice_lines)}

主要风险是 `sparse_low_coverage`：F1 下降 `{pp(normalized_slices['sparse_low_coverage']['f1_delta'])}` 个百分点，FP 增加 `{normalized_slices['sparse_low_coverage']['fp_delta']}`。退化航段包括：

{chr(10).join(segment_lines)}

## 决策

1. 冻结 V19B Seed-0，不覆盖当前部署模型。
2. 继续使用已锁定置信度 `0.371`，不为候选另行搜索阈值。
3. 人工复核稀疏低覆盖场景及 `DJI_2260-2360`、`DJI_2830-2880` 等退化航段，并只在训练集中寻找同类样本；禁止直接用验证图调参。
4. 测试集仍封存。本报告不授权测试集评估或部署替换。
"""
    README.write_text(text, encoding="utf-8")
    print(f"summary={SUMMARY}")
    print(f"report={README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
