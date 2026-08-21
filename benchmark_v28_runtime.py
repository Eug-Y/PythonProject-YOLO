"""Repeatable V28 web-runtime benchmark restricted to train/valid data."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from web_viewer import DEFAULT_YOLO_CONF, PREFERRED_MODEL_NAME, app


PROJECT_ROOT = Path(__file__).resolve().parent
ALLOWED_ROOTS = (
    (PROJECT_ROOT / "datasets" / "Data_v3" / "train" / "images").resolve(),
    (PROJECT_ROOT / "datasets" / "Data_v3" / "valid" / "images").resolve(),
)
FORBIDDEN_PARTS = {"data_release_blind_v1", "test"}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_image_path(path: Path) -> Path:
    resolved = path.resolve()
    lowered = {part.lower() for part in resolved.parts}
    if lowered & FORBIDDEN_PARTS:
        raise ValueError(f"V28 benchmark refuses test/blind data: {resolved}")
    if not any(_is_relative_to(resolved, root) for root in ALLOWED_ROOTS):
        raise ValueError(f"V28 benchmark only accepts Data_v3 train/valid images: {resolved}")
    if resolved.suffix.lower() not in {".jpg", ".jpeg"} or not resolved.is_file():
        raise ValueError(f"Not a readable JPG image: {resolved}")
    return resolved


def post_image(client, path: Path, mode: str) -> tuple[float, dict]:
    started = time.perf_counter()
    response = client.post(
        "/api/extract",
        data={
            "images": (io.BytesIO(path.read_bytes()), path.name),
            "mode": mode,
            "threshold": "0.005",
            "ndvi_threshold": "0",
            "gsd": "0.028",
            "model": PREFERRED_MODEL_NAME,
            "yolo_conf": str(DEFAULT_YOLO_CONF),
        },
        content_type="multipart/form-data",
    )
    elapsed = time.perf_counter() - started
    payload = response.get_json()
    if response.status_code != 200 or not payload or not payload.get("results"):
        raise RuntimeError(f"Benchmark request failed: HTTP {response.status_code} {payload}")
    item = payload["results"][0]
    if item.get("status") != "ok":
        raise RuntimeError(f"Benchmark item failed: {item}")
    return elapsed, item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--mode", choices=("all", "weed", "predict"), default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    paths = [validate_image_path(path) for path in args.images]
    client = app.test_client()
    if args.warmup and args.mode in {"all", "predict"}:
        post_image(client, paths[0], "predict")

    rows = []
    for path in paths:
        elapsed, item = post_image(client, path, args.mode)
        result = item["results"]
        rows.append({
            "image": path.relative_to(PROJECT_ROOT).as_posix(),
            "seconds": round(elapsed, 4),
            "layers": sorted(item["images"]),
            "response_image_mb": round(sum(len(value) for value in item["images"].values()) / 1024 / 1024, 3),
            "yolo_detections": result.get("yolo_detections"),
            "yolo_error": result.get("yolo_error"),
            "weed_candidate_count": result.get("weed_candidate_count"),
            "weed_error": result.get("weed_error"),
        })

    seconds = [row["seconds"] for row in rows]
    summary = {
        "version": "v28_runtime_benchmark_v1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_policy": "Data_v3 train/valid only; test and release blind are rejected",
        "mode": args.mode,
        "model": PREFERRED_MODEL_NAME,
        "warmup": bool(args.warmup),
        "sample_count": len(rows),
        "median_seconds": round(statistics.median(seconds), 4),
        "mean_seconds": round(statistics.mean(seconds), 4),
        "max_seconds": round(max(seconds), 4),
        "results": rows,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.resolve()
        if not _is_relative_to(output, PROJECT_ROOT):
            raise ValueError("Benchmark output must stay inside the project root")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
