"""Plan or apply mask-preserving repairs for invalid YOLO polygon paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from label_quality_audit import has_self_intersection
from multispectral import PROJECT_ROOT, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=PROJECT_ROOT / "datasets" / "Data_v3")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "Data_v3_label_quality")
    parser.add_argument("--priority", default="P1-sam2-geometry")
    parser.add_argument("--apply", action="store_true", help="Apply eligible repairs after writing the manifest")
    parser.add_argument("--revert", action="store_true", help="Revert an applied repair manifest")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports" / "Data_v3_geometry_repair_plan.json")
    return parser.parse_args()


def _line_parts(line: str) -> tuple[str, np.ndarray]:
    parts = line.strip().split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError("invalid YOLO segmentation row")
    return parts[0], np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)


def _rasterize(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pixels = np.rint(points * np.asarray([width, height])).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixels], 1)
    return mask


def repair_polygon(points: np.ndarray, width: int, height: int, max_points: int = 200) -> tuple[np.ndarray | None, str]:
    original_mask = _rasterize(points, width, height)
    if not np.any(original_mask):
        return None, "empty rasterized mask"
    contours, hierarchy = cv2.findContours(original_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None, "no contour"
    hierarchy = hierarchy[0]
    external = [index for index, values in enumerate(hierarchy) if values[3] == -1]
    holes = [index for index, values in enumerate(hierarchy) if values[3] != -1]
    if len(external) != 1:
        return None, f"mask has {len(external)} connected components"
    if holes:
        return None, f"mask has {len(holes)} holes"

    contour = contours[external[0]]
    if len(contour) < 3:
        return None, "repaired contour has fewer than 3 points"

    candidates = [contour]
    candidates.extend(cv2.approxPolyDP(contour, epsilon, True) for epsilon in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0))
    eligible: list[np.ndarray] = []
    reasons: list[str] = []
    for candidate in candidates:
        repaired = candidate[:, 0, :].astype(np.float64) / np.asarray([width, height])
        if len(repaired) < 3:
            continue
        if has_self_intersection(repaired):
            reasons.append("repaired contour still self-intersects")
            continue
        repaired_mask = _rasterize(repaired, width, height)
        if not np.array_equal(original_mask, repaired_mask):
            changed = int(np.count_nonzero(original_mask != repaired_mask))
            reasons.append(f"repair changes {changed} mask pixels")
            continue
        if len(repaired) > max_points:
            reasons.append(f"repair requires {len(repaired)} points (limit={max_points})")
            continue
        eligible.append(repaired)
    if not eligible:
        return None, reasons[0] if reasons else "no valid repaired contour"
    return min(eligible, key=len), "eligible"


def _format_line(class_id: str, points: np.ndarray) -> str:
    coordinates = " ".join(f"{value:.8f}" for value in points.reshape(-1))
    return f"{class_id} {coordinates}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_source(stem: str, source: str) -> Path | None:
    if source == "sam2_new":
        candidate = PROJECT_ROOT / "待标注_100张" / "labels" / f"{stem}.txt"
        return candidate if candidate.is_file() else None
    for split in ("train", "valid", "test"):
        candidate = PROJECT_ROOT / "datasets" / "Data_v2" / split / "labels" / f"{stem}.txt"
        if candidate.is_file():
            return candidate
    return None


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def revert_manifest(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "applied":
        raise ValueError(f"Manifest status is not applied: {manifest.get('status')}")
    changes_by_path: dict[Path, dict[int, str]] = defaultdict(dict)
    originals_by_path: dict[Path, list[str]] = {}
    for repair in manifest["repairs"]:
        for key in ("label_path", "canonical_source"):
            path = Path(repair[key])
            lines = originals_by_path.setdefault(path, path.read_text(encoding="utf-8").splitlines())
            line_index = int(repair["line"]) - 1
            if lines[line_index] != repair["repaired_line"]:
                raise ValueError(f"Current label differs from manifest at {path}:{line_index + 1}")
            changes_by_path[path][line_index] = repair["original_line"]
    for path, changes in changes_by_path.items():
        lines = originals_by_path[path].copy()
        for line_index, replacement in changes.items():
            lines[line_index] = replacement
        _atomic_write(path, "\n".join(lines) + "\n")
    manifest["status"] = "reverted"
    manifest["reverted_at"] = datetime.now().astimezone().isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reverted": len(manifest["repairs"]), "status": "reverted"}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.apply and args.revert:
        raise SystemExit("--apply and --revert are mutually exclusive")
    if args.revert:
        return revert_manifest(args.manifest.resolve())
    dataset = args.dataset.resolve()
    report = args.report.resolve()
    queue_path = report / "review_queue.csv"
    if not queue_path.is_file():
        raise SystemExit(f"Review queue does not exist: {queue_path}")

    with queue_path.open(encoding="utf-8-sig", newline="") as handle:
        queue = [row for row in csv.DictReader(handle) if row["priority"] == args.priority]

    repairs: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    changes_by_path: dict[Path, dict[int, str]] = defaultdict(dict)
    originals_by_path: dict[Path, list[str]] = {}
    seen: set[tuple[str, str, int]] = set()

    for row in queue:
        label_path = Path(row["label_path"])
        image_path = Path(row["image_path"])
        lines = originals_by_path.setdefault(label_path, label_path.read_text(encoding="utf-8").splitlines())
        image = read_image(image_path)
        height, width = image.shape[:2]
        issue_tokens = [token for token in row["issue_lines"].split(";") if token]
        target_lines = sorted(
            {
                int(token.rsplit(":", 1)[1])
                for token in issue_tokens
                if token.startswith(("self_intersection:", "duplicate_vertex:"))
            }
        )
        for line_number in target_lines:
            key = (row["split"], row["stem"], line_number)
            if key in seen:
                continue
            seen.add(key)
            original_line = lines[line_number - 1]
            class_id, points = _line_parts(original_line)
            repaired, reason = repair_polygon(points, width, height)
            base = {
                "split": row["split"],
                "stem": row["stem"],
                "source": row["source"],
                "line": line_number,
                "label_path": str(label_path),
                "original_points": len(points),
            }
            if repaired is None:
                skipped.append({**base, "reason": reason})
                continue
            repaired_line = _format_line(class_id, repaired)
            changes_by_path[label_path][line_number - 1] = repaired_line
            repairs.append(
                {
                    **base,
                    "repaired_points": len(repaired),
                    "original_line": original_line,
                    "repaired_line": repaired_line,
                    "original_line_sha256": _sha256_text(original_line),
                }
            )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset),
        "priority": args.priority,
        "status": "planned",
        "criteria": "single component, no holes, no self-intersection, exact pixel-mask equality",
        "eligible_repairs": len(repairs),
        "skipped": len(skipped),
        "repairs": repairs,
        "skipped_rows": skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.apply and repairs:
        for repair in repairs:
            source_path = _canonical_source(str(repair["stem"]), str(repair["source"]))
            if source_path is None:
                raise FileNotFoundError(f"Canonical label not found for {repair['stem']}")
            source_lines = originals_by_path.setdefault(source_path, source_path.read_text(encoding="utf-8").splitlines())
            line_index = int(repair["line"]) - 1
            if source_lines[line_index] != repair["original_line"]:
                raise ValueError(f"Canonical label differs at {source_path}:{line_index + 1}")
            changes_by_path[source_path][line_index] = str(repair["repaired_line"])
            repair["canonical_source"] = str(source_path)

        for path, changes in changes_by_path.items():
            lines = originals_by_path[path].copy()
            for line_index, replacement in changes.items():
                lines[line_index] = replacement
            _atomic_write(path, "\n".join(lines) + "\n")
        manifest["status"] = "applied"
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"eligible": len(repairs), "skipped": len(skipped), "status": manifest["status"]}, indent=2))
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
