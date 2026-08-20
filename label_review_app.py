"""Local review and correction UI for prioritized YOLO segmentation labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from label_quality_audit import has_self_intersection, polygon_area
from multispectral import PROJECT_ROOT, read_image


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".review-tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parse_line(line: str) -> tuple[str, np.ndarray]:
    parts = line.strip().split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError("标签行不是有效的 YOLO 分割格式")
    points = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
    return parts[0], points


def _format_line(class_id: str, points: np.ndarray) -> str:
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in points.reshape(-1))


def _item_id(split: str, stem: str, line: int, codes: str) -> str:
    raw = f"{split}:{stem}:{line}:{codes}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _clean_draft_workspace(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("草稿格式无效")

    def clean_points(value: Any, *, minimum: int) -> list[list[float]]:
        points = np.asarray(value, dtype=np.float64)
        if points.size == 0 and minimum == 0:
            return []
        if points.ndim != 2 or points.shape[1] != 2 or not minimum <= len(points) <= 200:
            raise ValueError("草稿轮廓点数无效")
        if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
            raise ValueError("草稿坐标必须位于 0 到 1")
        return points.tolist()

    raw_contours = raw.get("draft_contours", [])
    if not isinstance(raw_contours, list) or len(raw_contours) > 50:
        raise ValueError("草稿轮廓列表无效")
    contours = [clean_points(points, minimum=3) for points in raw_contours]
    draft = clean_points(raw.get("draft", []), minimum=0)
    mode = raw.get("mode")
    if mode not in {"redraw", "add", "delete"}:
        raise ValueError("草稿编辑模式无效")
    vertex_tool = raw.get("vertex_tool", "move")
    if vertex_tool not in {"move", "insert", "delete"}:
        raise ValueError("草稿顶点工具无效")
    selected_hash = raw.get("selected_hash")
    if selected_hash is not None and not isinstance(selected_hash, str):
        raise ValueError("草稿轮廓选择无效")
    return {
        "mode": mode,
        "vertex_tool": vertex_tool,
        "draft": draft,
        "draft_contours": contours,
        "selected_hash": selected_hash,
        "freehand": bool(raw.get("freehand", False)),
        "snap": bool(raw.get("snap", True)),
    }


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    split: str
    stem: str
    initial_line: int
    codes: tuple[str, ...]
    source: str
    label_path: Path
    image_path: Path


class ReviewStore:
    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        dataset_root: Path | None = None,
        report_dir: Path | None = None,
        geometry_csv: Path | None = None,
        state_path: Path | None = None,
        priority: str = "P1-sam2-geometry",
    ) -> None:
        self.project_root = project_root.resolve()
        self.dataset_root = (dataset_root or self.project_root / "datasets" / "Data_v3").resolve()
        self.report_dir = (report_dir or self.project_root / "reports" / "Data_v3_label_quality").resolve()
        self.geometry_csv = (
            geometry_csv or self.project_root / "reports" / "Data_v3_geometry_review" / "geometry_review.csv"
        ).resolve()
        self.state_path = (state_path or self.project_root / "reports" / "Data_v3_label_review_state.json").resolve()
        self.priority = priority
        self.lock = threading.RLock()
        self.state = self._load_state()
        self.items = self._load_items()
        self._initialize_item_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"version": 1, "items": {}, "history": [], "drafts": {}}
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("version") != 1:
            raise ValueError(f"不支持的复核状态版本: {state.get('version')}")
        state.setdefault("items", {})
        state.setdefault("history", [])
        state.setdefault("drafts", {})
        return state

    def _load_items(self) -> dict[str, ReviewItem]:
        if not self.geometry_csv.is_file():
            raise FileNotFoundError(f"缺少几何复核明细: {self.geometry_csv}")
        source_map: dict[tuple[str, str], str] = {}
        queue_path = self.report_dir / "review_queue.csv"
        with queue_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["priority"] == self.priority:
                    source_map[(row["split"], row["stem"])] = row["source"]

        items: dict[str, ReviewItem] = {}
        with self.geometry_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["split"], row["stem"])
                if key not in source_map:
                    continue
                line = int(row["line"])
                codes = tuple(sorted(filter(None, row["codes"].split(","))))
                item_id = _item_id(row["split"], row["stem"], line, ",".join(codes))
                items[item_id] = ReviewItem(
                    item_id=item_id,
                    split=row["split"],
                    stem=row["stem"],
                    initial_line=line,
                    codes=codes,
                    source=source_map[key],
                    label_path=self.dataset_root / row["split"] / "labels" / f"{row['stem']}.txt",
                    image_path=self.dataset_root / row["split"] / "images" / f"{row['stem']}.JPG",
                )
        return items

    def _initialize_item_state(self) -> None:
        changed = False
        for item in self.items.values():
            if item.item_id in self.state["items"]:
                continue
            lines = item.label_path.read_text(encoding="utf-8").splitlines()
            if item.initial_line < 1 or item.initial_line > len(lines):
                raise ValueError(f"复核行越界: {item.label_path}:{item.initial_line}")
            original = lines[item.initial_line - 1]
            self.state["items"][item.item_id] = {
                "status": "pending",
                "original_line": original,
                "original_hash": _hash_text(original),
                "current_hash": _hash_text(original),
                "current_hashes": [_hash_text(original)],
                "updated_at": None,
            }
            changed = True
        if changed:
            self._save_state()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.state_path, json.dumps(self.state, ensure_ascii=False, indent=2))

    def _canonical_label(self, item: ReviewItem) -> Path:
        if item.source == "sam2_new":
            path = self.project_root / "待标注_100张" / "labels" / f"{item.stem}.txt"
            if path.is_file():
                return path
        for split in ("train", "valid", "test"):
            path = self.project_root / "datasets" / "Data_v2" / split / "labels" / f"{item.stem}.txt"
            if path.is_file():
                return path
        raise FileNotFoundError(f"找不到规范源标签: {item.stem}")

    @staticmethod
    def _find_line(lines: list[str], line_hash: str | None, initial_line: int) -> int | None:
        if line_hash is None:
            return None
        matches = [index for index, line in enumerate(lines) if _hash_text(line) == line_hash]
        if not matches:
            return None
        return min(matches, key=lambda index: abs(index - (initial_line - 1)))

    def _resolved_targets(self, item: ReviewItem, lines: list[str]) -> list[tuple[int, str, np.ndarray]]:
        item_state = self.state["items"][item.item_id]
        hashes = item_state.get("current_hashes")
        if hashes is None:
            hashes = [item_state.get("current_hash")] if item_state.get("current_hash") else []
        resolved = []
        seen_indices: set[int] = set()
        for line_hash in hashes:
            index = self._find_line(lines, line_hash, item.initial_line)
            if index is None or index in seen_indices:
                continue
            class_id, points = _parse_line(lines[index])
            resolved.append((index, class_id, points))
            seen_indices.add(index)
        return sorted(resolved, key=lambda value: value[0])

    def _summary_row(self, item: ReviewItem) -> dict[str, Any]:
        state = self.state["items"][item.item_id]
        return {
            "id": item.item_id,
            "stem": item.stem,
            "split": item.split,
            "line": item.initial_line,
            "codes": list(item.codes),
            "status": state["status"],
            "has_draft": item.item_id in self.state.get("drafts", {}),
        }

    def list_items(self) -> dict[str, Any]:
        order = {"pending": 0, "redrawn": 1, "deleted": 2, "kept": 3}
        items = sorted(
            (self._summary_row(item) for item in self.items.values()),
            key=lambda row: (order.get(row["status"], 9), row["stem"], row["line"]),
        )
        counts: dict[str, int] = {status: 0 for status in ("pending", "kept", "redrawn", "deleted")}
        for row in items:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {
            "items": items,
            "counts": counts,
            "total": len(items),
            "drafts": len(self.state.get("drafts", {})),
            "can_undo": bool(self.state["history"]),
        }

    def get_item(self, item_id: str) -> dict[str, Any]:
        item = self.items[item_id]
        lines = item.label_path.read_text(encoding="utf-8").splitlines()
        targets = self._resolved_targets(item, lines)
        target_indices = {target[0] for target in targets}
        target_contours = [target[2] for target in targets]
        polygons = []
        for index, line in enumerate(lines):
            try:
                class_id, points = _parse_line(line)
            except ValueError:
                continue
            polygons.append(
                {
                    "line": index + 1,
                    "hash": _hash_text(line),
                    "class_id": class_id,
                    "points": points.tolist(),
                    "target": index in target_indices,
                }
            )
        image = read_image(item.image_path)
        item_state = self.state["items"][item_id]
        return {
            **self._summary_row(item),
            "source": item.source,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "image_url": f"/api/review/image/{item_id}",
            "label_path": str(item.label_path),
            "target_line": targets[0][0] + 1 if targets else None,
            "target_points": target_contours[0].tolist() if target_contours else [],
            "target_contours": [points.tolist() for points in target_contours],
            "polygons": polygons,
            "workspace": self.state.get("drafts", {}).get(item_id),
            "has_draft": item_id in self.state.get("drafts", {}),
            "updated_at": item_state.get("updated_at"),
        }

    def save_draft(self, item_id: str, raw_workspace: Any) -> dict[str, Any]:
        if item_id not in self.items:
            raise KeyError(item_id)
        workspace = _clean_draft_workspace(raw_workspace)
        with self.lock:
            if self.state["items"][item_id]["status"] != "pending":
                raise ValueError("已处理项目不能保存编辑草稿")
            workspace["updated_at"] = datetime.now().astimezone().isoformat()
            self.state.setdefault("drafts", {})[item_id] = workspace
            self._save_state()
        return {"workspace": workspace, "summary": self.list_items()}

    def clear_draft(self, item_id: str) -> dict[str, Any]:
        if item_id not in self.items:
            raise KeyError(item_id)
        with self.lock:
            self.state.setdefault("drafts", {}).pop(item_id, None)
            self._save_state()
        return self.list_items()

    @staticmethod
    def _validated_points(raw_points: Any) -> np.ndarray:
        points = np.asarray(raw_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 200:
            raise ValueError("重画轮廓必须包含 3 到 200 个点")
        if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
            raise ValueError("轮廓坐标必须位于 0 到 1")
        if len(np.unique(np.round(points, 10), axis=0)) != len(points):
            raise ValueError("轮廓包含重复顶点")
        if has_self_intersection(points):
            raise ValueError("重画轮廓仍然自交")
        if polygon_area(points) < 1e-5:
            raise ValueError("轮廓面积过小")
        return points

    @classmethod
    def _validated_contours(cls, raw_contours: Any) -> list[np.ndarray]:
        if not isinstance(raw_contours, list) or not 1 <= len(raw_contours) <= 50:
            raise ValueError("一次重画必须包含 1 到 50 个独立轮廓")
        return [cls._validated_points(points) for points in raw_contours]

    def _record_state_action(self, item: ReviewItem, action: str, previous: dict[str, Any]) -> None:
        now = datetime.now().astimezone().isoformat()
        item_state = self.state["items"][item.item_id]
        item_state["status"] = action
        item_state["updated_at"] = now
        self.state["history"].append(
            {
                "item_id": item.item_id,
                "action": action,
                "previous_state": previous,
                "files": [],
                "created_at": now,
            }
        )
        self._save_state()

    def apply_action(
        self,
        item_id: str,
        action: str,
        raw_points: Any = None,
        raw_contours: Any = None,
        raw_polygon_hashes: Any = None,
        raw_class_id: Any = "0",
    ) -> dict[str, Any]:
        if action not in {"kept", "deleted", "redrawn", "added", "deleted_selected"}:
            raise ValueError(f"不支持的操作: {action}")
        with self.lock:
            item = self.items[item_id]
            item_state = self.state["items"][item_id]
            previous_state = dict(item_state)
            if item_state["status"] != "pending":
                raise ValueError("该项已经处理；如需修改，请先撤销上次操作")
            if action == "kept":
                current_lines = item.label_path.read_text(encoding="utf-8").splitlines()
                current_hash = item_state.get("current_hash")
                if self._find_line(current_lines, current_hash, item.initial_line) is not None:
                    self.state.setdefault("drafts", {}).pop(item_id, None)
                    self._record_state_action(item, action, previous_state)
                    return self.get_item(item_id)

                original_line = str(item_state.get("original_line", "")).strip()
                if not original_line or _hash_text(original_line) != item_state.get("original_hash"):
                    raise ValueError("原始轮廓记录损坏，无法恢复并保留")
                file_records = []
                prepared: list[tuple[Path, str, str]] = []
                for path in (item.label_path, self._canonical_label(item)):
                    before = path.read_text(encoding="utf-8")
                    lines = before.splitlines()
                    if not any(_hash_text(line) == item_state["original_hash"] for line in lines):
                        lines.append(original_line)
                    after = ("\n".join(lines) + "\n") if lines else ""
                    prepared.append((path, before, after))
                    file_records.append({"path": str(path), "before": before, "after_hash": _hash_text(after)})
                written: list[tuple[Path, str]] = []
                try:
                    for path, before, after in prepared:
                        _atomic_write(path, after)
                        written.append((path, before))
                except Exception:
                    for path, before in reversed(written):
                        _atomic_write(path, before)
                    raise
                now = datetime.now().astimezone().isoformat()
                item_state["status"] = "kept"
                item_state["current_hash"] = item_state["original_hash"]
                item_state["current_hashes"] = [item_state["original_hash"]]
                item_state["updated_at"] = now
                self.state["history"].append(
                    {
                        "item_id": item_id,
                        "action": "restored_kept",
                        "previous_state": previous_state,
                        "files": file_records,
                        "created_at": now,
                    }
                )
                self.state.setdefault("drafts", {}).pop(item_id, None)
                self._save_state()
                return self.get_item(item_id)

            contours = None
            if action in {"redrawn", "added"}:
                contours = self._validated_contours(raw_contours if raw_contours is not None else [raw_points])
            class_id = str(raw_class_id).strip()
            if action == "added" and not class_id.isdigit():
                raise ValueError("新增轮廓的类别编号必须是非负整数")
            polygon_hashes: list[str] = []
            if action == "deleted_selected":
                if not isinstance(raw_polygon_hashes, list) or not 1 <= len(raw_polygon_hashes) <= 50:
                    raise ValueError("请选择 1 到 50 个待删除轮廓")
                polygon_hashes = [str(value) for value in raw_polygon_hashes]
                target_hashes = set(item_state.get("current_hashes") or [item_state.get("current_hash")])
                if any(value in target_hashes for value in polygon_hashes):
                    raise ValueError("当前异常轮廓请使用“删除异常轮廓”操作")
                protected: dict[str, ReviewItem] = {}
                for other_item in self.items.values():
                    if other_item.item_id == item_id or other_item.label_path != item.label_path:
                        continue
                    other_state = self.state["items"][other_item.item_id]
                    if other_state.get("status") != "pending":
                        continue
                    for value in other_state.get("current_hashes") or [other_state.get("current_hash")]:
                        if value:
                            protected[value] = other_item
                conflicts = [protected[value] for value in polygon_hashes if value in protected]
                if conflicts:
                    conflict = conflicts[0]
                    raise ValueError(
                        f"所选轮廓属于待复核项 {conflict.stem} 第 {conflict.initial_line} 行，"
                        "请切换到该项后使用保留、重画或删除异常轮廓"
                    )
            dataset_path = item.label_path
            canonical_path = self._canonical_label(item)
            file_records = []
            prepared: list[tuple[Path, str, str]] = []
            replacement_hash: str | None = None
            replacement_hashes: list[str] = []
            for path in (dataset_path, canonical_path):
                before = path.read_text(encoding="utf-8")
                lines = before.splitlines()
                if action == "added":
                    additions = [_format_line(class_id, points) for points in contours]
                    lines.extend(additions)
                    after = ("\n".join(lines) + "\n") if lines else ""
                    prepared.append((path, before, after))
                    file_records.append({"path": str(path), "before": before, "after_hash": _hash_text(after)})
                    continue
                if action == "deleted_selected":
                    indices: list[int] = []
                    for polygon_hash in polygon_hashes:
                        matches = [index for index, line in enumerate(lines) if _hash_text(line) == polygon_hash]
                        if len(matches) != 1:
                            raise ValueError(f"无法唯一定位待删除轮廓: {path}")
                        indices.append(matches[0])
                    if len(set(indices)) != len(indices):
                        raise ValueError("待删除轮廓列表包含重复项")
                    for index in sorted(indices, reverse=True):
                        lines.pop(index)
                    after = ("\n".join(lines) + "\n") if lines else ""
                    prepared.append((path, before, after))
                    file_records.append({"path": str(path), "before": before, "after_hash": _hash_text(after)})
                    continue
                index = self._find_line(lines, item_state.get("current_hash"), item.initial_line)
                if index is None:
                    raise ValueError(f"无法在标签中定位当前轮廓: {path}")
                class_id, _ = _parse_line(lines[index])
                if action == "deleted":
                    lines.pop(index)
                    replacement_hash = None
                    local_hashes: list[str] = []
                else:
                    replacements = [_format_line(class_id, points) for points in contours]
                    lines[index:index + 1] = replacements
                    local_hashes = [_hash_text(replacement) for replacement in replacements]
                    replacement_hash = local_hashes[0]
                    replacement_hashes = local_hashes
                after = "\n".join(lines) + "\n"
                prepared.append((path, before, after))
                file_records.append({"path": str(path), "before": before, "after_hash": _hash_text(after)})

            written: list[tuple[Path, str]] = []
            try:
                for path, before, after in prepared:
                    _atomic_write(path, after)
                    written.append((path, before))
            except Exception:
                for path, before in reversed(written):
                    _atomic_write(path, before)
                raise

            now = datetime.now().astimezone().isoformat()
            if action in {"deleted", "redrawn"}:
                item_state["status"] = action
                item_state["current_hash"] = replacement_hash
                item_state["current_hashes"] = replacement_hashes
            item_state["updated_at"] = now
            self.state["history"].append(
                {
                    "item_id": item_id,
                    "action": action,
                    "previous_state": previous_state,
                    "files": file_records,
                    "created_at": now,
                }
            )
            self.state.setdefault("drafts", {}).pop(item_id, None)
            self._save_state()
            return self.get_item(item_id)

    def undo(self) -> dict[str, Any]:
        with self.lock:
            if not self.state["history"]:
                raise ValueError("没有可撤销的操作")
            event = self.state["history"][-1]
            for file_record in event["files"]:
                path = Path(file_record["path"])
                current = path.read_text(encoding="utf-8")
                if _hash_text(current) != file_record["after_hash"]:
                    raise ValueError(f"标签已被其他程序修改，拒绝撤销: {path}")
            for file_record in event["files"]:
                _atomic_write(Path(file_record["path"]), file_record["before"])
            self.state["items"][event["item_id"]] = event["previous_state"]
            self.state["history"].pop()
            self._save_state()
            return {"undone": event["action"], "item_id": event["item_id"], **self.list_items()}


def create_app(store: ReviewStore | None = None) -> Flask:
    app = Flask(__name__)
    review_store = store or ReviewStore()
    app.config["REVIEW_STORE"] = review_store

    @app.get("/")
    def index():
        return render_template("label_review.html")

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/api/review/health")
    def health():
        return jsonify({"ok": True, "dataset": str(review_store.dataset_root), "items": len(review_store.items)})

    @app.get("/api/review/items")
    def list_items():
        return jsonify(review_store.list_items())

    @app.get("/api/review/item/<item_id>")
    def get_item(item_id: str):
        if item_id not in review_store.items:
            return jsonify({"error": "复核项不存在"}), 404
        return jsonify(review_store.get_item(item_id))

    @app.get("/api/review/image/<item_id>")
    def get_image(item_id: str):
        if item_id not in review_store.items:
            return jsonify({"error": "复核项不存在"}), 404
        return send_file(review_store.items[item_id].image_path, mimetype="image/jpeg", conditional=True)

    @app.post("/api/review/item/<item_id>/action")
    def apply_action(item_id: str):
        if item_id not in review_store.items:
            return jsonify({"error": "复核项不存在"}), 404
        payload = request.get_json(silent=True) or {}
        try:
            result = review_store.apply_action(
                item_id,
                str(payload.get("action", "")),
                payload.get("points"),
                payload.get("contours"),
                payload.get("polygon_hashes"),
                payload.get("class_id", "0"),
            )
            return jsonify({"ok": True, "item": result, "summary": review_store.list_items()})
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/review/item/<item_id>/draft")
    def save_draft(item_id: str):
        if item_id not in review_store.items:
            return jsonify({"error": "复核项不存在"}), 404
        try:
            return jsonify({"ok": True, **review_store.save_draft(item_id, request.get_json(silent=True) or {})})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.delete("/api/review/item/<item_id>/draft")
    def clear_draft(item_id: str):
        if item_id not in review_store.items:
            return jsonify({"error": "复核项不存在"}), 404
        return jsonify({"ok": True, "summary": review_store.clear_draft(item_id)})

    @app.post("/api/review/undo")
    def undo():
        try:
            return jsonify({"ok": True, **review_store.undo()})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "Data_v3_label_quality")
    parser.add_argument(
        "--geometry-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "Data_v3_geometry_review" / "geometry_review.csv",
    )
    parser.add_argument("--state", type=Path, default=PROJECT_ROOT / "reports" / "Data_v3_label_review_state.json")
    parser.add_argument("--priority", default="P1-sam2-geometry")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    store = ReviewStore(
        dataset_root=args.dataset_root,
        report_dir=args.report_dir,
        geometry_csv=args.geometry_csv,
        state_path=args.state,
        priority=args.priority,
    )
    create_app(store).run(host=args.host, port=args.port, debug=False, threaded=True)
