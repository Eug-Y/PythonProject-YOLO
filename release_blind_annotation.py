"""Manual RGB-only annotation and sealing for the release blind holdout.

The immutable selection manifest is never modified.  Human annotations are
tracked separately and can only be sealed after every image has an explicit
label (including explicitly confirmed empty images).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from difficult_sample_correction_app import (
    _atomic_write,
    _clean_workspace,
    _format_labels,
    _parse_label,
)
from label_quality_audit import polygon_area


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "Data_release_blind_v1"
STATE_VERSION = "release_blind_manual_annotation_state_v1"
SEAL_VERSION = "release_blind_manual_annotation_seal_v1"
ATTESTATION = "I_CONFIRM_MANUAL_RGB_ONLY_NO_MODEL_PREDICTIONS"
WARNING_LABELS = {
    "tiny_polygon": "轮廓面积过小",
    "excessive_points": "轮廓点数过多",
    "duplicate_polygon": "存在重复轮廓",
    "overlapping_polygons": "轮廓高度重叠或相互包含",
    "summed_area_exceeds_image": "轮廓总面积超过整图",
    "polygon_count_outlier": "本图轮廓数量明显离群",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selection(dataset_root: Path) -> dict[str, Any]:
    """Verify the immutable image/feature contract without reading labels."""
    dataset_root = Path(dataset_root).resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "release_blind_holdout_v1":
        raise ValueError("不支持的盲测集 manifest 版本")
    records = manifest.get("records")
    if manifest.get("selected_images") != 30 or not isinstance(records, list) or len(records) != 30:
        raise ValueError("盲测集必须严格包含 30 张图片")
    stems = []
    for record in records:
        stem = str(record["stem"])
        image = dataset_root / "test" / "images" / f"{stem}.JPG"
        feature = dataset_root / "test" / "features" / f"{stem}.npy"
        if not image.is_file() or not feature.is_file():
            raise FileNotFoundError(f"盲测文件缺失: {stem}")
        if sha256_file(image) != record.get("image_sha256"):
            raise ValueError(f"盲测图片哈希漂移: {stem}")
        if sha256_file(feature) != record.get("feature_sha256"):
            raise ValueError(f"盲测特征哈希漂移: {stem}")
        stems.append(stem)
    signature = hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()
    if signature != manifest.get("selected_stem_signature"):
        raise ValueError("盲测样本顺序签名漂移")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "stems": stems,
        "images_verified": len(stems),
        "features_verified": len(stems),
    }


def _clean_actor(value: Any) -> str:
    actor = str(value or "").strip()
    if not 2 <= len(actor) <= 80:
        raise ValueError("请填写 2 至 80 个字符的人工标注员标识")
    return actor


def annotation_warnings(polygons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-blocking quality warnings for a formally saved label set."""
    warnings: list[dict[str, Any]] = []
    arrays = [np.asarray(polygon["points"], dtype=np.float64) for polygon in polygons]
    areas = [polygon_area(points) for points in arrays]
    for index, (points, area) in enumerate(zip(arrays, areas)):
        if area < 1e-5:
            warnings.append({
                "code": "tiny_polygon", "label": WARNING_LABELS["tiny_polygon"],
                "polygon_indices": [index], "detail": f"轮廓 {index + 1} 归一化面积仅 {area:.8f}",
            })
        if len(points) > 200:
            warnings.append({
                "code": "excessive_points", "label": WARNING_LABELS["excessive_points"],
                "polygon_indices": [index], "detail": f"轮廓 {index + 1} 包含 {len(points)} 个点",
            })

    signatures: dict[tuple[float, ...], int] = {}
    duplicate_pairs: set[tuple[int, int]] = set()
    for index, points in enumerate(arrays):
        signature = tuple(np.round(points.reshape(-1), 8))
        if signature in signatures:
            pair = (signatures[signature], index)
            duplicate_pairs.add(pair)
            warnings.append({
                "code": "duplicate_polygon", "label": WARNING_LABELS["duplicate_polygon"],
                "polygon_indices": list(pair),
                "detail": f"轮廓 {pair[0] + 1} 与 {pair[1] + 1} 完全重复",
            })
        else:
            signatures[signature] = index

    # A small normalized raster makes pairwise overlap checks deterministic
    # and inexpensive while remaining independent of image resolution.
    raster_size = 384
    rasterized: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for points in arrays:
        pixels = np.rint(points * (raster_size - 1)).astype(np.int32)
        mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pixels], 1)
        rows, columns = np.nonzero(mask)
        bbox = (
            int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)
        ) if len(rows) else (0, 0, 0, 0)
        rasterized.append((mask, bbox, int(mask.sum())))
    for first in range(len(rasterized)):
        first_mask, first_bbox, first_area = rasterized[first]
        for second in range(first + 1, len(rasterized)):
            if (first, second) in duplicate_pairs:
                continue
            second_mask, second_bbox, second_area = rasterized[second]
            x1, y1 = max(first_bbox[0], second_bbox[0]), max(first_bbox[1], second_bbox[1])
            x2, y2 = min(first_bbox[2], second_bbox[2]), min(first_bbox[3], second_bbox[3])
            if x1 >= x2 or y1 >= y2 or min(first_area, second_area) == 0:
                continue
            intersection = int(np.logical_and(
                first_mask[y1:y2, x1:x2], second_mask[y1:y2, x1:x2]
            ).sum())
            union = first_area + second_area - intersection
            iou = intersection / union if union else 0.0
            coverage = intersection / min(first_area, second_area)
            if iou >= 0.5 or coverage >= 0.9:
                warnings.append({
                    "code": "overlapping_polygons", "label": WARNING_LABELS["overlapping_polygons"],
                    "polygon_indices": [first, second],
                    "detail": (
                        f"轮廓 {first + 1} 与 {second + 1} 高度重叠"
                        f"（IoU {iou:.2f}，较小轮廓覆盖率 {coverage:.2f}）"
                    ),
                })
    if sum(areas) > 1.05:
        warnings.append({
            "code": "summed_area_exceeds_image", "label": WARNING_LABELS["summed_area_exceeds_image"],
            "polygon_indices": [], "detail": f"所有轮廓面积之和为整图的 {sum(areas):.1%}",
        })
    return warnings


class BlindAnnotationStore:
    def __init__(self, dataset_root: Path = DEFAULT_DATASET, verify_hashes: bool = True) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest_path = self.dataset_root / "manifest.json"
        self.state_path = self.dataset_root / "annotation_state.json"
        self.seal_path = self.dataset_root / "annotation_manifest.json"
        self.labels_dir = self.dataset_root / "test" / "labels"
        self.images_dir = self.dataset_root / "test" / "images"
        self.lock = threading.RLock()
        if verify_hashes:
            verified = verify_selection(self.dataset_root)
        else:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            verified = {
                "manifest": manifest,
                "manifest_sha256": sha256_file(self.manifest_path),
                "stems": [str(row["stem"]) for row in manifest["records"]],
            }
        self.manifest = verified["manifest"]
        self.manifest_sha256 = verified["manifest_sha256"]
        self.stems = verified["stems"]
        self.by_stem = {str(row["stem"]): row for row in self.manifest["records"]}
        self._warning_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_or_initialize()
        self._verify_state_files()

    def _load_or_initialize(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") != STATE_VERSION:
                raise ValueError("不支持的人工标注状态版本")
            if state.get("selection_manifest_sha256") != self.manifest_sha256:
                raise ValueError("人工标注状态与冻结盲测 manifest 不匹配")
            state.setdefault("drafts", {})
            state.setdefault("history", [])
            return state
        unexpected = [path.name for path in self.labels_dir.glob("*.txt")]
        if unexpected:
            raise ValueError("发现未纳入状态追踪的标签，拒绝初始化: " + ", ".join(unexpected[:5]))
        created = now_iso()
        state = {
            "version": STATE_VERSION,
            "created_at": created,
            "updated_at": created,
            "selection_manifest_sha256": self.manifest_sha256,
            "annotation_mode": "manual_rgb_only",
            "model_predictions_exposed": False,
            "sealed": False,
            "items": {
                stem: {"status": "pending", "label_sha256": None, "polygon_count": None,
                       "actor": None, "updated_at": None}
                for stem in self.stems
            },
            "drafts": {},
            "history": [],
        }
        self._save_state(state)
        return state

    def _save_state(self, state: dict[str, Any] | None = None) -> None:
        if state is not None:
            self.state = state
        self.state["updated_at"] = now_iso()
        _atomic_write(self.state_path, json.dumps(self.state, ensure_ascii=False, indent=2))

    def _label_path(self, stem: str) -> Path:
        return self.labels_dir / f"{stem}.txt"

    def _verify_state_files(self) -> None:
        if set(self.state.get("items", {})) != set(self.stems):
            raise ValueError("人工标注状态的样本集合与冻结 manifest 不一致")
        for stem in self.stems:
            item = self.state["items"][stem]
            path = self._label_path(stem)
            if item["status"] == "annotated":
                if not path.is_file() or sha256_file(path) != item.get("label_sha256"):
                    raise ValueError(f"人工标签文件漂移: {stem}")
                _parse_label(path.read_text(encoding="utf-8"))
            elif item["status"] == "pending":
                if path.exists():
                    raise ValueError(f"待标注样本意外存在正式标签: {stem}")
            else:
                raise ValueError(f"未知人工标注状态: {stem}")
        extras = {path.stem for path in self.labels_dir.glob("*.txt")} - set(self.stems)
        if extras:
            raise ValueError("标签目录含盲测集外文件: " + ", ".join(sorted(extras)))
        if self.state.get("sealed"):
            if not self.seal_path.is_file() or sha256_file(self.seal_path) != self.state.get("seal_sha256"):
                raise ValueError("人工标注封存清单漂移或缺失")

    def _warning_map(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        completed_counts: list[tuple[str, int]] = []
        for stem in self.stems:
            item = self.state["items"][stem]
            if item["status"] != "annotated":
                continue
            cache_key = (stem, str(item.get("label_sha256")))
            warnings = self._warning_cache.get(cache_key)
            if warnings is None:
                polygons = _parse_label(self._label_path(stem).read_text(encoding="utf-8"))
                warnings = annotation_warnings(polygons)
                self._warning_cache[cache_key] = warnings
            result[stem] = list(warnings)
            completed_counts.append((stem, int(item.get("polygon_count") or 0)))
        if len(completed_counts) >= 5:
            counts = [count for _, count in completed_counts]
            median = statistics.median(counts)
            mad = statistics.median(abs(count - median) for count in counts)
            threshold = max(10.0, 3.5 * 1.4826 * mad)
            for stem, count in completed_counts:
                if abs(count - median) > threshold:
                    result[stem].append({
                        "code": "polygon_count_outlier", "label": WARNING_LABELS["polygon_count_outlier"],
                        "polygon_indices": [],
                        "detail": f"本图 {count} 个轮廓；已完成图片中位数 {median:.1f}，警戒差值 {threshold:.1f}",
                    })
        return result

    def status(self, warning_map: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
        completed = sum(self.state["items"][stem]["status"] == "annotated" for stem in self.stems)
        warnings = warning_map if warning_map is not None else self._warning_map()
        return {
            "total": len(self.stems),
            "completed": completed,
            "remaining": len(self.stems) - completed,
            "drafts": len(self.state.get("drafts", {})),
            "sealed": bool(self.state.get("sealed")),
            "can_undo": bool(self.state.get("history")) and not self.state.get("sealed"),
            "warning_images": sum(bool(items) for items in warnings.values()),
            "warning_count": sum(len(items) for items in warnings.values()),
        }

    def list_items(self) -> dict[str, Any]:
        warning_map = self._warning_map()
        rows = []
        for index, stem in enumerate(self.stems, 1):
            item = self.state["items"][stem]
            warnings = warning_map.get(stem, [])
            rows.append({
                "index": index,
                "stem": stem,
                "image": f"{stem}.JPG",
                "status": item["status"],
                "polygon_count": item.get("polygon_count"),
                "actor": item.get("actor"),
                "updated_at": item.get("updated_at"),
                "has_draft": stem in self.state.get("drafts", {}),
                "has_warning": bool(warnings),
                "warning_count": len(warnings),
                "warning_codes": sorted({warning["code"] for warning in warnings}),
                "warning_summary": "；".join(dict.fromkeys(warning["label"] for warning in warnings)),
            })
        return {"items": rows, "status": self.status(warning_map)}

    def get_item(self, stem: str) -> dict[str, Any]:
        if stem not in self.by_stem:
            raise KeyError(stem)
        state_item = self.state["items"][stem]
        if state_item["status"] == "annotated":
            polygons = _parse_label(self._label_path(stem).read_text(encoding="utf-8"))
        else:
            polygons = []
        stored = self.state.get("drafts", {}).get(stem)
        warnings = self._warning_map().get(stem, [])
        workspace = stored or {"polygons": polygons, "draft": [], "mode": None, "selected": -1}
        return {
            "index": self.stems.index(stem) + 1,
            "stem": stem,
            "image": f"{stem}.JPG",
            "status": state_item["status"],
            "polygon_count": len(workspace["polygons"]),
            "actor": state_item.get("actor"),
            "updated_at": state_item.get("updated_at"),
            "has_draft": stored is not None,
            "workspace": workspace,
            "image_url": f"/api/image/{stem}",
            "sealed": bool(self.state.get("sealed")),
            "has_warning": bool(warnings),
            "warning_count": len(warnings),
            "warnings": warnings,
        }

    def save_draft(self, stem: str, raw_workspace: Any) -> dict[str, Any]:
        if stem not in self.by_stem:
            raise KeyError(stem)
        if self.state.get("sealed"):
            raise ValueError("人工标签已封存，禁止继续修改")
        workspace = _clean_workspace(raw_workspace)
        with self.lock:
            self.state.setdefault("drafts", {})[stem] = {**workspace, "updated_at": now_iso()}
            self._save_state()
        return self.status()

    def save_item(self, stem: str, raw_polygons: Any, actor: Any, confirm_empty: bool = False) -> dict[str, Any]:
        if stem not in self.by_stem:
            raise KeyError(stem)
        if self.state.get("sealed"):
            raise ValueError("人工标签已封存，禁止继续修改")
        actor_text = _clean_actor(actor)
        text = _format_labels(raw_polygons)
        polygon_count = len(raw_polygons) if isinstance(raw_polygons, list) else 0
        if polygon_count == 0 and confirm_empty is not True:
            raise ValueError("空场景必须显式二次确认")
        path = self._label_path(stem)
        with self.lock:
            previous_item = dict(self.state["items"][stem])
            previous_text = path.read_text(encoding="utf-8") if path.is_file() else None
            _atomic_write(path, text)
            label_hash = sha256_file(path)
            current_time = now_iso()
            self.state["items"][stem] = {
                "status": "annotated",
                "label_sha256": label_hash,
                "polygon_count": polygon_count,
                "actor": actor_text,
                "updated_at": current_time,
            }
            self.state["history"].append({
                "stem": stem,
                "previous_item": previous_item,
                "previous_text": previous_text,
                "after_sha256": label_hash,
                "actor": actor_text,
                "created_at": current_time,
            })
            self.state.setdefault("drafts", {}).pop(stem, None)
            self._save_state()
        return self.get_item(stem)

    def undo(self) -> dict[str, Any]:
        with self.lock:
            if self.state.get("sealed"):
                raise ValueError("人工标签已封存，禁止撤销")
            if not self.state.get("history"):
                raise ValueError("没有可撤销的正式保存")
            event = self.state["history"][-1]
            path = self._label_path(event["stem"])
            if not path.is_file() or sha256_file(path) != event["after_sha256"]:
                raise ValueError("标签已被其他程序修改，拒绝撤销")
            if event["previous_text"] is None:
                path.unlink()
            else:
                _atomic_write(path, event["previous_text"])
            self.state["items"][event["stem"]] = event["previous_item"]
            self.state["history"].pop()
            self._save_state()
            return {"stem": event["stem"], **self.status()}

    def seal(self, actor: Any, attestation: Any) -> dict[str, Any]:
        actor_text = _clean_actor(actor)
        if attestation != ATTESTATION:
            raise ValueError("必须确认标签由人工仅参考 RGB 原图绘制，未参考模型预测")
        with self.lock:
            if self.state.get("sealed"):
                return json.loads(self.seal_path.read_text(encoding="utf-8"))
            status = self.status()
            if status["remaining"] or status["drafts"]:
                raise ValueError("30 张图片必须全部正式保存且无草稿后才能封存")
            verified = verify_selection(self.dataset_root)
            if verified["manifest_sha256"] != self.manifest_sha256:
                raise ValueError("冻结盲测 manifest 在标注期间发生变化")
            records = []
            signature_parts = []
            total_polygons = 0
            for stem in self.stems:
                path = self._label_path(stem)
                if sha256_file(path) != self.state["items"][stem]["label_sha256"]:
                    raise ValueError(f"封存前标签哈希漂移: {stem}")
                polygons = _parse_label(path.read_text(encoding="utf-8"))
                label_hash = sha256_file(path)
                total_polygons += len(polygons)
                signature_parts.append(f"{stem}\0{label_hash}")
                records.append({
                    "stem": stem,
                    "label_sha256": label_hash,
                    "polygon_count": len(polygons),
                    "annotator": self.state["items"][stem]["actor"],
                    "annotated_at": self.state["items"][stem]["updated_at"],
                })
            sealed = {
                "version": SEAL_VERSION,
                "sealed_at": now_iso(),
                "sealed_by": actor_text,
                "state": "sealed_manual_ground_truth",
                "selection_manifest_sha256": self.manifest_sha256,
                "selected_stem_signature": self.manifest["selected_stem_signature"],
                "label_set_signature": hashlib.sha256("\n".join(signature_parts).encode("utf-8")).hexdigest(),
                "images": len(records),
                "labels": len(records),
                "total_polygons": total_polygons,
                "annotation_mode": "manual_rgb_only",
                "model_predictions_exposed": False,
                "pseudo_labels_used": False,
                "held_out_from_training_validation_and_model_selection": True,
                "records": records,
            }
            _atomic_write(self.seal_path, json.dumps(sealed, ensure_ascii=False, indent=2))
            self.state["sealed"] = True
            self.state["sealed_at"] = sealed["sealed_at"]
            self.state["sealed_by"] = actor_text
            self.state["seal_sha256"] = sha256_file(self.seal_path)
            self._save_state()
            return sealed


def create_app(dataset_root: Path = DEFAULT_DATASET, verify_hashes: bool = True) -> Flask:
    app = Flask(__name__)
    store = BlindAnnotationStore(dataset_root, verify_hashes=verify_hashes)
    app.config["BLIND_ANNOTATION_STORE"] = store

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("release_blind_annotation.html", attestation=ATTESTATION)

    @app.get("/api/items")
    def items():
        return jsonify(store.list_items())

    @app.get("/api/item/<stem>")
    def item(stem: str):
        try:
            return jsonify(store.get_item(stem))
        except KeyError:
            return jsonify({"error": "未知盲测样本"}), 404

    @app.get("/api/image/<stem>")
    def image(stem: str):
        if stem not in store.by_stem:
            return jsonify({"error": "未知盲测样本"}), 404
        return send_file(store.images_dir / f"{stem}.JPG", conditional=True)

    @app.post("/api/item/<stem>/draft")
    def save_draft(stem: str):
        payload = request.get_json(silent=True) or {}
        try:
            status = store.save_draft(stem, payload.get("workspace"))
            return jsonify({"ok": True, "status": status})
        except KeyError:
            return jsonify({"error": "未知盲测样本"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/item/<stem>/save")
    def save(stem: str):
        payload = request.get_json(silent=True) or {}
        try:
            saved = store.save_item(
                stem, payload.get("polygons"), payload.get("actor"), payload.get("confirm_empty") is True
            )
            return jsonify({"ok": True, "item": saved, "status": store.status()})
        except KeyError:
            return jsonify({"error": "未知盲测样本"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/undo")
    def undo():
        try:
            return jsonify({"ok": True, **store.undo()})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/seal")
    def seal():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify({"ok": True, "manifest": store.seal(payload.get("actor"), payload.get("attestation"))})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5058)
    args = parser.parse_args()
    if args.verify_only:
        result = verify_selection(args.dataset)
        print(json.dumps({
            "ok": True,
            "manifest_sha256": result["manifest_sha256"],
            "images_verified": result["images_verified"],
            "features_verified": result["features_verified"],
        }, ensure_ascii=False, indent=2))
        return 0
    create_app(args.dataset).run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
