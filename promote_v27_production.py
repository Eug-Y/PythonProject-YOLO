"""Atomically promote a passed one-shot V27 blind release to production."""

from __future__ import annotations

import json
from pathlib import Path

from deployment_fusion import load_shadow_config, sha256
from release_blind_annotation import now_iso
from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
DEPLOYMENT = ROOT / "configs" / "deployment.json"
BACKUP = ROOT / "configs" / "deployment.pre_v27.json"
BLIND_SUMMARY = ROOT / "logs" / "v27_release_blind_final.summary.json"
BLIND_GUARD = ROOT / "logs" / "v27_release_blind_evaluation_guard.json"
PROMOTION_SUMMARY = ROOT / "logs" / "v27_production_promotion.summary.json"


def build_production_config(current: dict, shadow: dict, blind: dict) -> dict:
    fusion = shadow["fusion"]
    return {
        "version": "deployment_config_v2_yolo11_primary_fusion",
        "state": "active",
        "model_name": shadow["model_name"],
        "deployment_type": "strict_yolo11_primary_fusion",
        "checkpoint": shadow["primary"]["checkpoint"],
        "sha256": shadow["primary"]["sha256"],
        "primary": shadow["primary"],
        "auxiliary": shadow["auxiliary"],
        "fusion_config": shadow["fusion_config"],
        "fusion_config_sha256": shadow["fusion_config_sha256"],
        "confidence": fusion["score_threshold"],
        "unconfirmed_score_threshold": fusion["unconfirmed_score_threshold"],
        "imgsz": shadow["imgsz"],
        "channels": shadow["channels"],
        "every_output_requires_yolo11": True,
        "release_version": shadow["release_candidate"],
        "release_blind_summary": str(BLIND_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        "release_blind_summary_sha256": sha256(BLIND_SUMMARY),
        "promoted_at": now_iso(),
        "rollback_model": current,
    }


def main() -> int:
    if PROMOTION_SUMMARY.exists():
        raise RuntimeError("V27 production promotion has already been recorded")
    blind = json.loads(BLIND_SUMMARY.read_text(encoding="utf-8"))
    guard = json.loads(BLIND_GUARD.read_text(encoding="utf-8"))
    if (
        blind.get("decision") != "promote_v27_to_production"
        or blind.get("production_switch_allowed") is not True
        or guard.get("state") != "complete"
        or guard.get("evaluation_count") != 1
        or guard.get("summary_sha256") != sha256(BLIND_SUMMARY)
    ):
        raise RuntimeError("Passed, single-use V27 blind authorization is required")
    shadow = load_shadow_config()
    current = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    rollback = shadow["rollback"]
    if current.get("model_name") != rollback["model_name"] or current.get("sha256") != rollback["sha256"]:
        raise RuntimeError("Current production no longer matches the frozen V27 rollback target")
    if BACKUP.exists():
        if json.loads(BACKUP.read_text(encoding="utf-8")) != current:
            raise RuntimeError("Existing pre-V27 deployment backup differs from current production")
    else:
        write_json_atomic(BACKUP, current)
    promoted = build_production_config(current, shadow, blind)
    write_json_atomic(DEPLOYMENT, promoted)
    if json.loads(DEPLOYMENT.read_text(encoding="utf-8")) != promoted:
        raise RuntimeError("Atomic production deployment verification failed")
    summary = {
        "version": "v27_production_promotion_summary_v1",
        "created_at": now_iso(),
        "state": "promoted",
        "model_name": promoted["model_name"],
        "primary": promoted["primary"],
        "auxiliary": promoted["auxiliary"],
        "fusion_config_sha256": promoted["fusion_config_sha256"],
        "blind_summary_sha256": promoted["release_blind_summary_sha256"],
        "rollback_config": str(BACKUP.relative_to(ROOT)).replace("\\", "/"),
        "rollback_ready": True,
    }
    write_json_atomic(PROMOTION_SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
