"""Persistent image-level review UI for validation difficult samples."""

from __future__ import annotations

import argparse
import csv
import json
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "reports" / "Data_v3_difficult_sample_review_v1"
DEFAULT_QUEUE = DEFAULT_REPORT / "review_queue.csv"
DEFAULT_STATE = DEFAULT_REPORT / "review_state.json"
DEFAULT_SOURCE = ROOT / "reports" / "Data_v3_deployment_valid_errors_v1"
DEFAULT_IMAGES = ROOT / "datasets" / "Data_v3" / "valid" / "images"
DECISIONS = {
    "label_ok_hard": "标签正确，保留困难样本",
    "needs_label_fix": "标签需要修正",
    "defer": "暂不处理",
}


class DifficultReviewStore:
    def __init__(self, queue_path: Path, state_path: Path):
        self.queue_path = queue_path
        self.state_path = state_path
        self.lock = threading.Lock()
        with queue_path.open(encoding="utf-8-sig", newline="") as handle:
            self.queue = list(csv.DictReader(handle))
        self.by_image = {row["image"]: row for row in self.queue}
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(state.get("decisions"), dict):
                return state
        return {"version": "difficult_sample_review_state_v1", "decisions": {}}

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def serialized_queue(self) -> list[dict]:
        decisions = self.state["decisions"]
        return [
            {
                **row,
                "decision": decisions.get(row["image"], {}).get("decision"),
                "decision_label": DECISIONS.get(decisions.get(row["image"], {}).get("decision", ""), ""),
                "note": decisions.get(row["image"], {}).get("note", ""),
            }
            for row in self.queue
        ]

    def status(self) -> dict:
        counts = {key: 0 for key in DECISIONS}
        for item in self.state["decisions"].values():
            decision = item.get("decision")
            if decision in counts:
                counts[decision] += 1
        completed = sum(counts.values())
        return {
            "total": len(self.queue),
            "completed": completed,
            "remaining": len(self.queue) - completed,
            "counts": counts,
            "labels": DECISIONS,
        }

    def decide(self, image: str, decision: str, note: str = "") -> dict:
        if image not in self.by_image:
            raise KeyError(image)
        if decision not in DECISIONS:
            raise ValueError(decision)
        with self.lock:
            self.state["decisions"][image] = {
                "decision": decision,
                "note": note.strip()[:500],
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            self._save()
        return self.status()

    def undo(self, image: str) -> dict:
        if image not in self.by_image:
            raise KeyError(image)
        with self.lock:
            self.state["decisions"].pop(image, None)
            self._save()
        return self.status()


def create_app(
    queue_path: Path = DEFAULT_QUEUE,
    state_path: Path = DEFAULT_STATE,
    source_dir: Path = DEFAULT_SOURCE,
    image_dir: Path = DEFAULT_IMAGES,
) -> Flask:
    app = Flask(__name__)
    store = DifficultReviewStore(Path(queue_path), Path(state_path))
    source_dir = Path(source_dir)
    image_dir = Path(image_dir)
    app.config["REVIEW_STORE"] = store

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("difficult_sample_review.html")

    @app.get("/api/queue")
    def api_queue():
        return jsonify({"items": store.serialized_queue(), "status": store.status()})

    @app.post("/api/decision")
    def api_decision():
        payload = request.get_json(silent=True) or {}
        try:
            status = store.decide(str(payload.get("image", "")), str(payload.get("decision", "")), str(payload.get("note", "")))
        except KeyError:
            return jsonify({"error": "未知图片"}), 404
        except ValueError:
            return jsonify({"error": "未知复核结论"}), 400
        return jsonify({"ok": True, "status": status})

    @app.post("/api/undo")
    def api_undo():
        payload = request.get_json(silent=True) or {}
        try:
            status = store.undo(str(payload.get("image", "")))
        except KeyError:
            return jsonify({"error": "未知图片"}), 404
        return jsonify({"ok": True, "status": status})

    @app.get("/media/labels/<int:batch_index>")
    def media_labels(batch_index: int):
        return send_from_directory(source_dir, f"val_batch{batch_index}_labels.jpg")

    @app.get("/media/pred/<int:batch_index>")
    def media_pred(batch_index: int):
        return send_from_directory(source_dir, f"val_batch{batch_index}_pred.jpg")

    @app.get("/media/original/<path:filename>")
    def media_original(filename: str):
        return send_from_directory(image_dir, filename)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5056)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.queue.resolve(), args.state.resolve(), args.source.resolve(), args.images.resolve())
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
