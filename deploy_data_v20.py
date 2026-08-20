"""Publish an eligible sparse-context checkpoint as a reversible web release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from run_feature_ablations import write_json_atomic


ROOT = Path(__file__).resolve().parent
DECISION = ROOT / "runs" / "final_evaluation" / "data_v20_locked_test" / "decision.json"
DEPLOYMENT_CONFIG = ROOT / "configs" / "deployment.json"
MODEL_ROOT = ROOT / "runs" / "segment" / "land_seg"
RELEASE_LOG = ROOT / "logs" / "data_v20_deployment.json"
EXPECTED_INCUMBENT_SHA256 = "589a1d5600e58b1b5932a472137bddc057821daac4c8a5f98bb1e3a94715649b"
EXPERIMENT_LABEL = "V20"
RELEASE_SERIES = "v20"
RELEASE_FLAVOR = "sparsectx"
RELEASE_VERSION = "v20_sparse_context_v1"
ARTIFACT_PREFIX = "data_v20"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_name(seed: int) -> str:
    return f"prod_{RELEASE_SERIES}_5ch_{RELEASE_FLAVOR}_seed{seed}_v1"


def load_release_plan() -> dict[str, object]:
    if not DECISION.is_file() or not DEPLOYMENT_CONFIG.is_file():
        raise FileNotFoundError(DECISION if not DECISION.is_file() else DEPLOYMENT_CONFIG)
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if decision.get("upgrade_eligible") is not True or decision.get("decision") != "authorized_for_deployment":
        raise RuntimeError(f"{EXPERIMENT_LABEL} final-test decision does not authorize deployment")
    candidate = Path(decision["candidate"])
    if not candidate.is_file() or sha256(candidate) != decision["candidate_sha256"]:
        raise RuntimeError("Authorized candidate hash mismatch")
    current = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    current_checkpoint = ROOT / current["checkpoint"]
    if not current_checkpoint.is_file() or sha256(current_checkpoint) != current["sha256"]:
        raise RuntimeError("Current deployment config/checkpoint hash mismatch")
    if current.get("release_version") == "incumbent_pre_v20" and current["sha256"] != EXPECTED_INCUMBENT_SHA256:
        raise RuntimeError(f"Pre-{EXPERIMENT_LABEL} deployment is not the expected incumbent")
    seed = int(decision["candidate_seed"])
    name = release_name(seed)
    release_dir = MODEL_ROOT / name
    return {
        "decision": decision,
        "candidate": candidate,
        "candidate_hash": decision["candidate_sha256"],
        "candidate_contract": candidate.parent.parent / "feature_contract.json",
        "candidate_args": candidate.parent.parent / "args.yaml",
        "current": current,
        "current_checkpoint": current_checkpoint,
        "seed": seed,
        "release_name": name,
        "release_dir": release_dir,
        "release_checkpoint": release_dir / "weights" / "best.pt",
    }


def apply_release(plan: dict[str, object]) -> dict[str, object]:
    release_dir = Path(plan["release_dir"])
    release_checkpoint = Path(plan["release_checkpoint"])
    candidate = Path(plan["candidate"])
    contract = Path(plan["candidate_contract"])
    args_path = Path(plan["candidate_args"])
    if not contract.is_file():
        raise FileNotFoundError(contract)
    if release_dir.exists():
        manifest_path = release_dir / "deployment_manifest.json"
        if not manifest_path.is_file() or not release_checkpoint.is_file():
            raise RuntimeError(f"Incomplete release directory already exists: {release_dir}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("sha256") != plan["candidate_hash"] or sha256(release_checkpoint) != plan["candidate_hash"]:
            raise RuntimeError(f"Existing release hash mismatch: {release_dir}")
    else:
        (release_dir / "weights").mkdir(parents=True, exist_ok=False)
        shutil.copy2(candidate, release_checkpoint)
        shutil.copy2(contract, release_dir / "feature_contract.json")
        if args_path.is_file():
            shutil.copy2(args_path, release_dir / "args.yaml")
        if sha256(release_checkpoint) != plan["candidate_hash"]:
            raise RuntimeError("Copied release checkpoint hash mismatch")
        release_manifest = {
            "version": f"{ARTIFACT_PREFIX}_deployment_release_v1",
            "created_at": now_iso(),
            "release_name": plan["release_name"],
            "source_checkpoint": str(candidate.resolve()),
            "checkpoint": str(release_checkpoint.resolve()),
            "sha256": plan["candidate_hash"],
            "seed": plan["seed"],
            "channels": ["R", "G", "B", "NDVI", "NDRE"],
            "confidence": 0.371,
            "imgsz": 1024,
            "test_decision": str(DECISION.resolve()),
            "rollback_model": plan["current"]["model_name"],
        }
        (release_dir / "deployment_manifest.json").write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    deployed_at = now_iso()
    relative_checkpoint = release_checkpoint.resolve().relative_to(ROOT.resolve()).as_posix()
    deployment = {
        "version": "deployment_config_v1",
        "state": "active",
        "model_name": plan["release_name"],
        "checkpoint": relative_checkpoint,
        "sha256": plan["candidate_hash"],
        "confidence": 0.371,
        "imgsz": 1024,
        "channels": ["R", "G", "B", "NDVI", "NDRE"],
        "release_version": RELEASE_VERSION,
        "deployed_at": deployed_at,
        "source_candidate": str(candidate.resolve()),
        "test_decision": str(DECISION.resolve()),
        "rollback_model": plan["current"]["model_name"],
        "rollback_config": plan["current"],
    }
    write_json_atomic(DEPLOYMENT_CONFIG, deployment)
    audit = {
        "version": f"{ARTIFACT_PREFIX}_deployment_audit_v1",
        "deployed_at": deployed_at,
        "authorized_by_user_request": True,
        "release": deployment,
        "release_checkpoint_hash_verified": sha256(release_checkpoint) == plan["candidate_hash"],
        "previous_deployment": plan["current"],
        "previous_checkpoint_unchanged": sha256(Path(plan["current_checkpoint"])) == plan["current"]["sha256"],
        "rollback_available": True,
        "web_restart_required": True,
        "inference_regression_completed": False,
    }
    write_json_atomic(RELEASE_LOG, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the authorized release; default is a read-only plan.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_release_plan()
    printable = {
        "release_name": plan["release_name"],
        "candidate": str(Path(plan["candidate"]).resolve()),
        "candidate_sha256": plan["candidate_hash"],
        "release_checkpoint": str(Path(plan["release_checkpoint"]).resolve()),
        "previous_model": plan["current"]["model_name"],
        "confidence": 0.371,
    }
    if not args.apply:
        print(json.dumps({"mode": "dry_run", **printable}, ensure_ascii=False, indent=2))
        return 0
    audit = apply_release(plan)
    print(json.dumps({"mode": "applied", **printable, "rollback_available": audit["rollback_available"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
