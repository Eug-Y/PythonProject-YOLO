"""Compare the once-evaluated Data_v5 MCARI candidate with deployment baseline."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

from multispectral import PROJECT_ROOT


DEFAULT_BASELINE = PROJECT_ROOT / "reports" / "Data_v3_grouped_reviewed_v1_test_after_error_review" / "metrics.json"
DEFAULT_CANDIDATE = PROJECT_ROOT / "reports" / "Data_v5_final_candidate_test" / "metrics.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "Data_v5_final_model_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def is_upgrade_eligible(deltas: dict[str, float]) -> bool:
    return (
        deltas["metrics/mAP50-95(M)"] > 0.002
        and deltas["metrics/mAP50(M)"] >= -0.002
        and deltas["metrics/recall(M)"] >= -0.01
    )


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    keys = (
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)", "fitness",
    )
    missing = [key for key in keys if key not in baseline or key not in candidate]
    if missing:
        raise KeyError(f"Missing comparison metrics: {missing}")
    deltas = {key: float(candidate[key]) - float(baseline[key]) for key in keys}
    eligible = is_upgrade_eligible(deltas)
    decision = "candidate eligible for website integration" if eligible else "keep existing website model"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "data_v5_final_model_comparison_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "test_evaluation_count_for_candidate": 1,
        "baseline_model": "exp_ca_5ch_grouped_reviewed_v1",
        "baseline_channels": ["R", "G", "B", "NDVI", "NDRE"],
        "candidate_model": "v5_final_rgb_mcari_1024_seed0_v1",
        "candidate_channels": ["R", "G", "B", "MCARI"],
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas,
        "upgrade_rule": "Mask mAP50-95 > +0.2 pp, Mask mAP50 >= -0.2 pp, Mask recall >= -1.0 pp",
        "upgrade_eligible": eligible,
        "decision": decision,
        "mcari_note": "MCARI is mapped to DJI P4M Green/Red/RedEdge bands and was empirically validated before this frozen test.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = {
        "metrics/precision(M)": "Mask Precision",
        "metrics/recall(M)": "Mask Recall",
        "metrics/mAP50(M)": "Mask mAP50",
        "metrics/mAP50-95(M)": "Mask mAP50-95",
        "metrics/mAP50-95(B)": "Box mAP50-95",
    }
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{baseline[key]:.4f}</td><td>{candidate[key]:.4f}</td><td>{deltas[key] * 100:+.2f} pp</td></tr>"
        for key, label in labels.items()
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v5 最终模型对比</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:920px;margin:auto;background:#fff;padding:28px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}.note{{padding:14px;background:#eff6ff;border-left:4px solid #2563eb}}</style></head><body><main><h1>Data_v5 独立测试集最终对比</h1><table><thead><tr><th>指标</th><th>网站现模型</th><th>RGB+MCARI</th><th>变化</th></tr></thead><tbody>{rows}</tbody></table><p><strong>结论：</strong>{html.escape(decision)}</p><p class="note">候选仅评估一次独立 test；结果用于部署决策，不再用于特征调参。</p></main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"upgrade_eligible": eligible, "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()
