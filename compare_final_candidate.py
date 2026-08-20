"""Compare the frozen Data_v4 final candidate with the deployment baseline."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

from multispectral import PROJECT_ROOT


DEFAULT_BASELINE = (
    PROJECT_ROOT / "reports" / "Data_v3_grouped_reviewed_v1_test_after_error_review" / "metrics.json"
)
DEFAULT_CANDIDATE = PROJECT_ROOT / "reports" / "Data_v4_final_candidate_test" / "metrics.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "Data_v4_final_model_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    keys = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
        "fitness",
    )
    deltas = {key: float(candidate[key]) - float(baseline[key]) for key in keys}
    upgrade_eligible = (
        deltas["metrics/mAP50-95(M)"] > 0.002
        and deltas["metrics/mAP50(M)"] >= -0.002
        and deltas["metrics/recall(M)"] >= -0.01
    )
    decision = (
        "candidate eligible for website integration review"
        if upgrade_eligible
        else "keep existing website model"
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "data_v4_final_model_comparison_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "test_evaluation_count_for_candidate": 1,
        "baseline_model": "exp_ca_5ch_grouped_reviewed_v1",
        "candidate_model": "v4_final_rgb_re_bsi_proxy_1024_seed0_v1",
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas,
        "upgrade_rule": "Mask mAP50-95 > +0.2 pp, Mask mAP50 >= -0.2 pp, Mask recall >= -1.0 pp",
        "upgrade_eligible": upgrade_eligible,
        "decision": decision,
        "soil_proxy_disclaimer": "RE_BSI_PROXY is experimental and is not standard SWIR-based BSI.",
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
        f"<tr><td>{html.escape(label)}</td><td>{baseline[key]:.4f}</td><td>{candidate[key]:.4f}</td><td>{deltas[key]*100:+.2f} pp</td></tr>"
        for key, label in labels.items()
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v4 最终模型对比</title><style>body{{font-family:system-ui;margin:32px;background:#f8fafc;color:#1f2937}}main{{max-width:900px;margin:auto;background:#fff;padding:28px;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}.note{{padding:14px;background:#fff7ed;border-left:4px solid #f97316}}</style></head><body><main><h1>独立测试集最终对比</h1><table><thead><tr><th>指标</th><th>网站现模型</th><th>Data_v4 候选</th><th>变化</th></tr></thead><tbody>{rows}</tbody></table><p><strong>结论：</strong>{html.escape(decision)}</p><p class="note">候选仅评估一次独立 test。RE_BSI_PROXY 是实验代理，并非标准 SWIR-BSI。</p></main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"upgrade_eligible": upgrade_eligible, "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()

