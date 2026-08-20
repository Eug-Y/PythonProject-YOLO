"""Summarize extended final-reviewed teacher-refined runs without opening the frozen test split."""

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

from multispectral import PROJECT_ROOT
from run_feature_ablations import read_progress


PROJECT = PROJECT_ROOT / "runs" / "segment" / "land_seg"
OUTPUT = PROJECT_ROOT / "reports" / "Data_v6_final_reviewed_teacher_refined_multiseed"
ORDINARY_YOLOV8_VALID = 0.29256770547723165
SHORT_REFINED_VALID = 0.32811
DEPLOYED_LONGRUN_VALID = 0.36905
T95_DF2 = 4.3026527297
DEPLOYMENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
DIFFICULT_REVIEW = PROJECT_ROOT / "reports" / "Data_v6_sparse_mixed_difficult_review"


def experiment_name(seed: int) -> str:
    return f"v6_final_reviewed_teacher_refined_yolov8n_long_seed{seed}_v2"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(project: Path, seeds: tuple[int, ...]) -> dict:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        name = experiment_name(seed)
        progress = read_progress(project / name / "results.csv")
        checkpoint = project / name / "weights" / "best.pt"
        if progress is None or not checkpoint.is_file():
            rows.append({"seed": seed, "run": name, "status": "missing"})
            continue
        score = float(progress["best_mask_map50_95"])
        rows.append(
            {
                "seed": seed,
                "run": name,
                "status": "complete",
                **progress,
                "checkpoint": str(checkpoint.resolve()),
                "delta_vs_ordinary_yolov8n": score - ORDINARY_YOLOV8_VALID,
                "delta_vs_short_refined": score - SHORT_REFINED_VALID,
                "delta_vs_deployed_longrun": score - DEPLOYED_LONGRUN_VALID,
            }
        )
    complete = [row for row in rows if row["status"] == "complete"]
    values = [float(row["best_mask_map50_95"]) for row in complete]
    mean = statistics.mean(values) if values else None
    std = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
    margin = T95_DF2 * float(std) / math.sqrt(len(values)) if len(values) > 1 else 0.0 if values else None
    best = max(complete, key=lambda row: float(row["best_mask_map50_95"])) if complete else None
    all_complete = len(complete) == len(seeds)
    beats_deployed = bool(all_complete and mean is not None and mean > DEPLOYED_LONGRUN_VALID)
    decision = "retain_deployment"
    if beats_deployed:
        decision = "candidate_review_only_keep_deployment_until_frozen_test_gate"
    deployment_checkpoint = project / "exp_ca_5ch_grouped_reviewed_v1" / "weights" / "best.pt"
    deployment_hash = sha256(deployment_checkpoint)
    difficult_review: dict[str, object] = {"status": "missing"}
    queue_summary_path = DIFFICULT_REVIEW / "summary.json"
    queue_path = DIFFICULT_REVIEW / "review_queue.csv"
    if queue_summary_path.is_file() and queue_path.is_file():
        queue_summary = json.loads(queue_summary_path.read_text(encoding="utf-8"))
        queue_checkpoint = Path(str(queue_summary.get("checkpoint", ""))).resolve()
        best_checkpoint = Path(str(best["checkpoint"])).resolve() if best else None
        matches_best = bool(best_checkpoint and queue_checkpoint == best_checkpoint)
        difficult_review = {
            "status": "complete" if matches_best else "stale_checkpoint",
            "source_split": queue_summary.get("source_split"),
            "held_out_test_used": queue_summary.get("held_out_test_used"),
            "validation_images": queue_summary.get("validation_images"),
            "selected_images": queue_summary.get("selected_images"),
            "checkpoint_matches_best_seed": matches_best,
            "queue": str(queue_path.resolve()),
            "summary": str(queue_summary_path.resolve()),
        }
    return {
        "version": "data_v6_final_reviewed_teacher_refined_multiseed_summary_v2",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selection_splits": ["train", "valid"],
        "held_out_test_used": False,
        "external_upload_used": False,
        "dataloader_seeded_per_run": True,
        "seeds": list(seeds),
        "rows": rows,
        "complete_seeds": len(complete),
        "mean_mask_map50_95": mean,
        "std_mask_map50_95": std,
        "ci95_low": mean - margin if mean is not None and margin is not None else None,
        "ci95_high": mean + margin if mean is not None and margin is not None else None,
        "ordinary_yolov8n_seed0_valid": ORDINARY_YOLOV8_VALID,
        "short_refined_seed0_valid": SHORT_REFINED_VALID,
        "deployed_longrun_valid": DEPLOYED_LONGRUN_VALID,
        "mean_delta_vs_ordinary_yolov8n": mean - ORDINARY_YOLOV8_VALID if mean is not None else None,
        "mean_delta_vs_short_refined": mean - SHORT_REFINED_VALID if mean is not None else None,
        "mean_delta_vs_deployed_longrun": mean - DEPLOYED_LONGRUN_VALID if mean is not None else None,
        "positive_vs_short_refined_seeds": sum(value > SHORT_REFINED_VALID for value in values),
        "positive_vs_deployed_seeds": sum(value > DEPLOYED_LONGRUN_VALID for value in values),
        "best_seed": int(best["seed"]) if best else None,
        "best_checkpoint": str(best["checkpoint"]) if best else None,
        "deployment_eligible": False,
        "decision": decision if all_complete else "incomplete",
        "deployment_checkpoint": str(deployment_checkpoint.resolve()),
        "deployment_sha256_expected": DEPLOYMENT_SHA256,
        "deployment_sha256_current": deployment_hash,
        "deployment_unchanged": deployment_hash == DEPLOYMENT_SHA256,
        "difficult_review": difficult_review,
        "invalidated_seed_audit": {
            "runs": [
                f"v6_final_reviewed_teacher_refined_yolov8n_long_seed{seed}_v1" for seed in seeds
            ],
            "included_in_statistics": False,
            "reason": (
                "Ultralytics 8.4.51 used a constant DataLoader generator, so the requested seeds produced "
                "identical batches, worker augmentations, and model tensors. Version v2 injects each run seed "
                "into shuffle and worker RNG state."
            ),
        },
        "comparison_warning": "Ordinary and deployed reference values are seed-0 references, not paired three-seed controls.",
    }


def render_html(summary: dict) -> str:
    rows = []
    for row in summary["rows"]:
        if row["status"] != "complete":
            rows.append(f'<tr><td>{row["seed"]}</td><td colspan="6">缺失</td></tr>')
            continue
        rows.append(
            "<tr>"
            f'<td>{row["seed"]}</td>'
            f'<td>{row["best_epoch"]}</td>'
            f'<td>{float(row["best_mask_precision"]):.4f}</td>'
            f'<td>{float(row["best_mask_recall"]):.4f}</td>'
            f'<td>{float(row["best_mask_map50"]):.4f}</td>'
            f'<td>{float(row["best_mask_map50_95"]):.4f}</td>'
            f'<td>{float(row["delta_vs_short_refined"]) * 100:+.2f} pp</td>'
            "</tr>"
        )
    mean = summary["mean_mask_map50_95"]
    mean_text = "—" if mean is None else f"{float(mean):.4f}"
    delta = summary["mean_delta_vs_deployed_longrun"]
    delta_text = "—" if delta is None else f"{float(delta) * 100:+.2f} pp"
    std = summary["std_mask_map50_95"]
    std_text = "—" if std is None else f"{float(std):.5f}"
    ci_text = (
        "—"
        if summary["ci95_low"] is None
        else f"[{float(summary['ci95_low']):.4f}, {float(summary['ci95_high']):.4f}]"
    )
    decision = "继续保持现有部署模型" if summary["decision"] == "retain_deployment" else "候选仅进入冻结评估门控，部署仍保持不变"
    difficult = summary["difficult_review"]
    if difficult["status"] == "complete":
        queue_html = (
            f'<p>困难样本队列：从 {int(difficult["validation_images"])} 张 valid 图像中选择 '
            f'{int(difficult["selected_images"])} 张；'
            '<a href="../Data_v6_sparse_mixed_difficult_review/review_queue.csv">打开 CSV</a>。</p>'
        )
    else:
        queue_html = f'<p>困难样本队列状态：{html.escape(str(difficult["status"]))}</p>'
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>最终审核版三 seed 延长训练</title><style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f4f7f7;color:#172126;margin:28px}}main{{max-width:980px;margin:auto;background:#fff;padding:26px;border:1px solid #dce5e5;border-radius:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e5eaea;text-align:left}}th{{background:#eef4f4}}</style></head><body><main>
<h1>最终审核版三随机种子延长训练</h1><p><strong>{html.escape(decision)}</strong></p><p>三 seed 平均 Mask mAP50-95：{mean_text}，标准差 {std_text}，95% CI {ci_text}；相对长期部署验证参考：{delta_text}。</p>
<table><thead><tr><th>Seed</th><th>最佳轮</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th><th>相对15轮</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{queue_html}<p>仅使用 train/valid；冻结 test 未读取。参考基线为历史 seed-0 数值，不等同于三 seed 配对控制。失效的 v1 重复轨迹未纳入统计；部署 SHA256 保持不变：{html.escape(str(summary['deployment_unchanged']))}。</p></main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    summary = summarize(args.project.resolve(), seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.html").write_text(render_html(summary), encoding="utf-8")
    complete = [row for row in summary["rows"] if row["status"] == "complete"]
    if complete:
        fields = list(complete[0])
        with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(complete)
    readme = (
        "# 最终审核版三随机种子延长训练\n\n"
        f"- 完成 seed：{summary['complete_seeds']}/{len(seeds)}\n"
        f"- 平均 Mask mAP50-95：{summary['mean_mask_map50_95']}\n"
        f"- 相对 15 轮 seed-0：{summary['mean_delta_vs_short_refined']}\n"
        f"- 相对长期部署验证参考：{summary['mean_delta_vs_deployed_longrun']}\n"
        f"- 决策：{summary['decision']}\n"
        f"- 困难样本队列：{summary['difficult_review']}\n"
        f"- 失效 v1 纳入统计：{summary['invalidated_seed_audit']['included_in_statistics']}\n"
        "- 冻结 test：未读取\n"
        f"- 部署模型未修改：{summary['deployment_unchanged']}\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Report: {output / 'index.html'}")
    print(f"Decision: {summary['decision']}")
    return 0 if summary["complete_seeds"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
