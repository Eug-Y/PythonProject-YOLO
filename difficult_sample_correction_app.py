"""Persistent whole-image YOLO polygon correction UI for difficult validation samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from label_quality_audit import has_self_intersection, polygon_area


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "reports" / "Data_v3_difficult_sample_correction_v1"
DEFAULT_QUEUE = DEFAULT_REPORT / "correction_queue.csv"
DEFAULT_STATE = DEFAULT_REPORT / "correction_state.json"
DEFAULT_DATASET = ROOT / "datasets" / "Data_v3"
DEFAULT_REFERENCE = ROOT / "reports" / "Data_v3_deployment_valid_errors_v1"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".correction-tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _polygon_issues(points: np.ndarray) -> list[str]:
    issues = []
    if len(np.unique(np.round(points, 10), axis=0)) != len(points):
        issues.append("duplicate_vertex")
    if has_self_intersection(points):
        issues.append("self_intersection")
    if polygon_area(points) < 1e-7:
        issues.append("tiny_polygon")
    return issues


def _annotated_polygon(class_id: str, points: np.ndarray) -> dict[str, Any]:
    return {"class_id": class_id, "points": points.tolist(), "issues": _polygon_issues(points)}


def _parse_label(text: str) -> list[dict[str, Any]]:
    polygons = []
    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(f"标签第 {line_number} 行不是有效的 YOLO 分割格式")
        points = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
        polygons.append(_annotated_polygon(parts[0], points))
    return polygons


def _validate_polygon(raw: Any) -> tuple[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise ValueError("轮廓格式无效")
    class_id = str(raw.get("class_id", "0")).strip()
    if not class_id.isdigit():
        raise ValueError("类别编号必须是非负整数")
    points = np.asarray(raw.get("points"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 1000:
        raise ValueError("每个轮廓必须包含 3 至 1000 个点")
    if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
        raise ValueError("轮廓坐标必须位于 0 至 1")
    if len(np.unique(np.round(points, 10), axis=0)) != len(points):
        raise ValueError("轮廓包含重复顶点")
    if has_self_intersection(points):
        raise ValueError("轮廓存在自交，请重新绘制")
    if polygon_area(points) < 1e-7:
        raise ValueError("轮廓面积过小")
    return class_id, points


def _format_labels(raw_polygons: Any) -> str:
    if not isinstance(raw_polygons, list) or len(raw_polygons) > 2000:
        raise ValueError("整图轮廓列表无效")
    lines = []
    for index, raw in enumerate(raw_polygons, 1):
        try:
            class_id, points = _validate_polygon(raw)
        except ValueError as exc:
            raise ValueError(f"第 {index} 个轮廓：{exc}") from exc
        values = " ".join(f"{value:.8f}" for value in points.reshape(-1))
        lines.append(f"{class_id} {values}")
    return ("\n".join(lines) + "\n") if lines else ""


def _clean_workspace(raw: Any) -> dict[str, Any]:
    """Validate a resumable draft without rejecting pre-existing geometry issues."""
    if not isinstance(raw, dict):
        raise ValueError("草稿格式无效")
    raw_polygons = raw.get("polygons")
    if not isinstance(raw_polygons, list) or len(raw_polygons) > 2000:
        raise ValueError("草稿轮廓列表无效")

    polygons = []
    for index, raw_polygon in enumerate(raw_polygons, 1):
        if not isinstance(raw_polygon, dict):
            raise ValueError(f"草稿第 {index} 个轮廓格式无效")
        class_id = str(raw_polygon.get("class_id", "0")).strip()
        if not class_id.isdigit():
            raise ValueError(f"草稿第 {index} 个轮廓类别无效")
        points = np.asarray(raw_polygon.get("points"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 1000:
            raise ValueError(f"草稿第 {index} 个轮廓点数无效")
        if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
            raise ValueError(f"草稿第 {index} 个轮廓坐标无效")
        polygons.append({"class_id": class_id, "points": points.tolist()})

    raw_draft = raw.get("draft", [])
    draft = np.asarray(raw_draft, dtype=np.float64)
    if draft.size == 0:
        clean_draft: list[list[float]] = []
    else:
        if draft.ndim != 2 or draft.shape[1] != 2 or len(draft) > 1000:
            raise ValueError("当前未完成轮廓格式无效")
        if not np.all(np.isfinite(draft)) or np.any(draft < 0) or np.any(draft > 1):
            raise ValueError("当前未完成轮廓坐标无效")
        clean_draft = draft.tolist()

    mode = raw.get("mode")
    if mode not in {None, "add", "redraw"}:
        raise ValueError("草稿绘制模式无效")
    selected = int(raw.get("selected", -1))
    if selected < -1 or selected >= len(polygons):
        selected = -1
    return {"polygons": polygons, "draft": clean_draft, "mode": mode, "selected": selected}


class CorrectionStore:
    def __init__(
        self,
        queue_path: Path = DEFAULT_QUEUE,
        state_path: Path = DEFAULT_STATE,
        dataset_root: Path = DEFAULT_DATASET,
        project_root: Path = ROOT,
    ) -> None:
        self.queue_path = Path(queue_path).resolve()
        self.state_path = Path(state_path).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.project_root = Path(project_root).resolve()
        self.lock = threading.RLock()
        with self.queue_path.open(encoding="utf-8-sig", newline="") as handle:
            self.queue = list(csv.DictReader(handle))
        self.by_image = {row["image"]: row for row in self.queue}
        self.state = self._load_state()
        self._initialize()

    def _load_state(self) -> dict:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") != "difficult_sample_correction_state_v1":
                raise ValueError("不支持的修正状态版本")
            state.setdefault("items", {})
            state.setdefault("history", [])
            state.setdefault("drafts", {})
            return state
        return {"version": "difficult_sample_correction_state_v1", "items": {}, "history": [], "drafts": {}}

    def _label_path(self, row: dict) -> Path:
        return self.dataset_root / row.get("split", "valid") / "labels" / f"{row['stem']}.txt"

    def _image_path(self, row: dict) -> Path:
        split = row.get("split", "valid")
        image = row["image"]
        path = self.dataset_root / split / "images" / image
        if path.is_file():
            return path
        matches = list((self.dataset_root / split / "images").glob(f"{row['stem']}.*"))
        if not matches:
            raise FileNotFoundError(f"找不到图片: {image}")
        return matches[0]

    def _canonical_path(self, stem: str) -> Path:
        candidates = [
            self.project_root / "datasets" / "Data_v2" / split / "labels" / f"{stem}.txt"
            for split in ("train", "valid", "test")
        ]
        candidates.append(self.project_root / "待标注_100张" / "labels" / f"{stem}.txt")
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(f"{stem} 的规范源标签数量应为 1，实际为 {len(existing)}")
        return existing[0]

    def _initialize(self) -> None:
        changed = False
        for row in self.queue:
            image = row["image"]
            if image in self.state["items"]:
                continue
            text = self._label_path(row).read_text(encoding="utf-8")
            self.state["items"][image] = {
                "status": "pending",
                "original_hash": _hash(text),
                "current_hash": _hash(text),
                "updated_at": None,
            }
            changed = True
        if changed:
            self._save_state()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _atomic_write(self.state_path, json.dumps(self.state, ensure_ascii=False, indent=2))

    def status(self) -> dict:
        counts = {"pending": 0, "corrected": 0}
        for row in self.queue:
            status = self.state["items"][row["image"]]["status"]
            counts[status] = counts.get(status, 0) + 1
        return {
            "total": len(self.queue),
            "completed": counts.get("corrected", 0),
            "remaining": counts.get("pending", 0),
            "drafts": len(self.state.get("drafts", {})),
            "counts": counts,
            "can_undo": bool(self.state["history"]),
        }

    def list_items(self) -> dict:
        rows = []
        for row in self.queue:
            item = self.state["items"][row["image"]]
            rows.append(
                {
                    **row,
                    "status": item["status"],
                    "updated_at": item.get("updated_at"),
                    "has_draft": row["image"] in self.state.get("drafts", {}),
                }
            )
        rows.sort(key=lambda row: (row["status"] != "pending", int(row.get("rank", 999999))))
        return {"items": rows, "status": self.status()}

    def get_item(self, image: str) -> dict:
        row = self.by_image[image]
        label_path = self._label_path(row)
        text = label_path.read_text(encoding="utf-8")
        state = self.state["items"][image]
        saved_polygons = _parse_label(text)
        stored_workspace = self.state.get("drafts", {}).get(image)
        if stored_workspace:
            workspace = {
                **stored_workspace,
                "polygons": [
                    _annotated_polygon(raw["class_id"], np.asarray(raw["points"], dtype=np.float64))
                    for raw in stored_workspace["polygons"]
                ],
            }
        else:
            workspace = {
                "polygons": saved_polygons,
                "draft": [],
                "mode": None,
                "selected": -1,
                "updated_at": None,
            }
        invalid_count = sum(bool(polygon.get("issues")) for polygon in workspace["polygons"])
        return {
            **row,
            "status": state["status"],
            "polygons": workspace["polygons"],
            "polygon_count": len(workspace["polygons"]),
            "saved_polygon_count": len(saved_polygons),
            "invalid_polygon_count": invalid_count,
            "has_draft": stored_workspace is not None,
            "workspace": workspace,
            "image_url": f"/api/image/{image}",
            "prediction_url": f"/api/prediction/{image}",
            "label_path": str(label_path),
            "canonical_path": str(self._canonical_path(row["stem"])),
        }

    def save_draft(self, image: str, raw_workspace: Any) -> dict:
        if image not in self.by_image:
            raise KeyError(image)
        workspace = _clean_workspace(raw_workspace)
        with self.lock:
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            self.state.setdefault("drafts", {})[image] = {**workspace, "updated_at": now}
            self._save_state()
        return {"image": image, "updated_at": now, "status": self.status()}

    def clear_draft(self, image: str) -> dict:
        if image not in self.by_image:
            raise KeyError(image)
        with self.lock:
            self.state.setdefault("drafts", {}).pop(image, None)
            self._save_state()
        return self.status()

    def save_item(self, image: str, polygons: Any) -> dict:
        if image not in self.by_image:
            raise KeyError(image)
        after = _format_labels(polygons)
        with self.lock:
            row = self.by_image[image]
            dataset_path = self._label_path(row)
            canonical_path = self._canonical_path(row["stem"])
            before_dataset = dataset_path.read_text(encoding="utf-8")
            before_canonical = canonical_path.read_text(encoding="utf-8")
            previous_state = dict(self.state["items"][image])
            written: list[tuple[Path, str]] = []
            try:
                for path, before in ((dataset_path, before_dataset), (canonical_path, before_canonical)):
                    _atomic_write(path, after)
                    written.append((path, before))
            except Exception:
                for path, before in reversed(written):
                    _atomic_write(path, before)
                raise
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            self.state["items"][image] = {
                **previous_state,
                "status": "corrected",
                "current_hash": _hash(after),
                "updated_at": now,
            }
            self.state["history"].append(
                {
                    "image": image,
                    "previous_state": previous_state,
                    "files": [
                        {"path": str(dataset_path), "before": before_dataset, "after_hash": _hash(after)},
                        {"path": str(canonical_path), "before": before_canonical, "after_hash": _hash(after)},
                    ],
                    "created_at": now,
                }
            )
            self.state.setdefault("drafts", {}).pop(image, None)
            self._save_state()
        return self.get_item(image)

    def undo(self) -> dict:
        with self.lock:
            if not self.state["history"]:
                raise ValueError("没有可撤销的保存操作")
            event = self.state["history"][-1]
            for file_record in event["files"]:
                path = Path(file_record["path"])
                if _hash(path.read_text(encoding="utf-8")) != file_record["after_hash"]:
                    raise ValueError(f"标签已被其他程序修改，拒绝撤销: {path}")
            for file_record in event["files"]:
                _atomic_write(Path(file_record["path"]), file_record["before"])
            self.state["items"][event["image"]] = event["previous_state"]
            self.state["history"].pop()
            self._save_state()
            return {"image": event["image"], **self.status()}


def create_app(
    queue_path: Path = DEFAULT_QUEUE,
    state_path: Path = DEFAULT_STATE,
    dataset_root: Path = DEFAULT_DATASET,
    reference_dir: Path = DEFAULT_REFERENCE,
    project_root: Path = ROOT,
) -> Flask:
    app = Flask(__name__)
    store = CorrectionStore(queue_path, state_path, dataset_root, project_root)
    reference_dir = Path(reference_dir).resolve()
    app.config["CORRECTION_STORE"] = store

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("difficult_sample_correction.html")

    @app.get("/api/items")
    def items():
        return jsonify(store.list_items())

    @app.get("/api/item/<path:image>")
    def item(image: str):
        if image not in store.by_image:
            return jsonify({"error": "未知图片"}), 404
        return jsonify(store.get_item(image))

    @app.get("/api/image/<path:image>")
    def source_image(image: str):
        if image not in store.by_image:
            return jsonify({"error": "未知图片"}), 404
        return send_file(store._image_path(store.by_image[image]), conditional=True)

    @app.get("/api/prediction/<path:image>")
    def prediction(image: str):
        if image not in store.by_image:
            return jsonify({"error": "未知图片"}), 404
        batch = int(store.by_image[image]["batch_index"])
        return send_file(reference_dir / f"val_batch{batch}_pred.jpg", conditional=True)

    @app.post("/api/item/<path:image>/save")
    def save(image: str):
        payload = request.get_json(silent=True) or {}
        try:
            saved = store.save_item(image, payload.get("polygons"))
        except KeyError:
            return jsonify({"error": "未知图片"}), 404
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "item": saved, "status": store.status()})

    @app.post("/api/item/<path:image>/draft")
    def save_draft(image: str):
        payload = request.get_json(silent=True) or {}
        try:
            saved = store.save_draft(image, payload)
        except KeyError:
            return jsonify({"error": "未知图片"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, **saved})

    @app.delete("/api/item/<path:image>/draft")
    def clear_draft(image: str):
        try:
            status = store.clear_draft(image)
        except KeyError:
            return jsonify({"error": "未知图片"}), 404
        return jsonify({"ok": True, "status": status})

    @app.post("/api/undo")
    def undo():
        try:
            return jsonify({"ok": True, **store.undo()})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        args.queue.resolve(), args.state.resolve(), args.dataset_root.resolve(), args.reference.resolve()
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
