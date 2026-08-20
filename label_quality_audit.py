"""Audit YOLO segmentation labels and generate CSV, JSON, overlays, and HTML."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from multispectral import PROJECT_ROOT, read_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
AcceptedIssues = dict[tuple[str, str, str], set[str]]


@dataclass
class Issue:
    dataset: str
    split: str
    stem: str
    severity: str
    code: str
    line: int | None
    detail: str


@dataclass
class Polygon:
    class_id: int
    points: np.ndarray
    line: int
    area: float


@dataclass
class ImageAudit:
    dataset: str
    split: str
    stem: str
    source: str
    image_path: Path
    label_path: Path | None
    polygons: list[Polygon] = field(default_factory=list)
    polygon_count: int = 0
    point_count: int = 0
    average_points: float = 0.0
    summed_area_ratio: float = 0.0
    issue_count: int = 0


def add_issue(
    issues: list[Issue],
    dataset: str,
    split: str,
    stem: str,
    severity: str,
    code: str,
    detail: str,
    line: int | None = None,
) -> None:
    issues.append(Issue(dataset, split, stem, severity, code, line, detail))


def _review_item_id(split: str, stem: str, line: int, codes: str) -> str:
    raw = f"{split}:{stem}:{line}:{codes}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_review_acceptances(state_path: Path, geometry_csv: Path) -> tuple[AcceptedIssues, int]:
    """Load warning-level review acceptances, matched by exact label-row hash."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_items = state.get("items", {})
    accepted: AcceptedIssues = defaultdict(set)
    accepted_item_ids: set[str] = set()
    with geometry_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            codes = tuple(sorted(filter(None, row["codes"].split(","))))
            item_id = _review_item_id(row["split"], row["stem"], int(row["line"]), ",".join(codes))
            item_state = state_items.get(item_id, {})
            if item_state.get("status") != "kept":
                continue
            hashes = item_state.get("current_hashes") or [item_state.get("current_hash")]
            for line_hash in filter(None, hashes):
                accepted[(row["split"], row["stem"], line_hash)].update(codes)
            accepted_item_ids.add(item_id)
    return dict(accepted), len(accepted_item_ids)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) * 0.5


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float = 1e-10) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-10
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > eps and ab_d < -eps) or (ab_c < -eps and ab_d > eps)) and (
        (cd_a > eps and cd_b < -eps) or (cd_a < -eps and cd_b > eps)
    ):
        return True
    return any(
        (
            abs(value) <= eps and _on_segment(start, end, point)
            for value, start, end, point in (
                (ab_c, a, b, c),
                (ab_d, a, b, d),
                (cd_a, c, d, a),
                (cd_b, c, d, b),
            )
        )
    )


def has_self_intersection(points: np.ndarray) -> bool:
    count = len(points)
    if count < 4:
        return False
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count}:
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            if segments_intersect(a, b, c, d):
                return True
    return False


def parse_label(
    label_path: Path,
    dataset: str,
    split: str,
    stem: str,
    issues: list[Issue],
    expected_class: int,
    min_area: float,
    max_points: int,
    accepted_issues: AcceptedIssues | None = None,
) -> list[Polygon]:
    polygons: list[Polygon] = []
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        add_issue(issues, dataset, split, stem, "error", "encoding_error", str(exc))
        return polygons

    if not any(line.strip() for line in lines):
        # Empty YOLO label files are valid background/negative samples. Keep
        # them visible in the audit without incorrectly blocking training.
        add_issue(issues, dataset, split, stem, "warning", "empty_label", "Background image: label file is empty")
        return polygons

    seen_polygons: set[tuple[float, ...]] = set()
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        line_hash = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
        accepted_codes = (accepted_issues or {}).get((split, stem, line_hash), set())

        def add_warning(code: str, detail: str) -> None:
            if code not in accepted_codes:
                add_issue(issues, dataset, split, stem, "warning", code, detail, line_number)

        parts = raw_line.split()
        if len(parts) < 7:
            add_issue(
                issues, dataset, split, stem, "error", "too_few_values",
                f"Expected class plus at least 6 coordinates, got {len(parts)} values", line_number,
            )
            continue
        try:
            class_value = float(parts[0])
            values = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        except ValueError:
            add_issue(issues, dataset, split, stem, "error", "non_numeric", "Line contains non-numeric data", line_number)
            continue
        if not math.isfinite(class_value) or not np.isfinite(values).all():
            add_issue(issues, dataset, split, stem, "error", "non_finite", "Line contains NaN or infinity", line_number)
            continue
        if not class_value.is_integer():
            add_issue(issues, dataset, split, stem, "error", "invalid_class", f"Class is not an integer: {class_value}", line_number)
            continue
        class_id = int(class_value)
        if class_id != expected_class:
            add_issue(issues, dataset, split, stem, "error", "unexpected_class", f"Expected {expected_class}, got {class_id}", line_number)
        if len(values) % 2:
            add_issue(issues, dataset, split, stem, "error", "odd_coordinate_count", f"Got {len(values)} coordinates", line_number)
            continue
        if np.any(values < 0.0) or np.any(values > 1.0):
            add_issue(
                issues, dataset, split, stem, "error", "coordinate_out_of_range",
                f"Coordinate range is [{values.min():.6f}, {values.max():.6f}]", line_number,
            )
            continue

        points = values.reshape(-1, 2)
        if len(points) >= 2 and np.allclose(points[0], points[-1], atol=1e-10):
            add_warning("explicit_closing_point", "First and last points are duplicates")
            points = points[:-1]
        unique_points = np.unique(np.round(points, 10), axis=0)
        if len(unique_points) < 3:
            add_issue(issues, dataset, split, stem, "error", "degenerate_polygon", "Fewer than 3 unique points", line_number)
            continue
        if len(unique_points) != len(points):
            add_warning("duplicate_vertex", "Polygon contains duplicate vertices")

        area = polygon_area(points)
        if area < min_area:
            add_warning("tiny_polygon", f"Normalized area is {area:.8f}")
        if len(points) > max_points:
            add_warning("excessive_points", f"Polygon has {len(points)} points")
        if has_self_intersection(points):
            add_warning("self_intersection", "Polygon edges intersect")

        signature = tuple(np.round(points.flatten(), 8))
        if signature in seen_polygons:
            add_issue(issues, dataset, split, stem, "error", "duplicate_polygon", "Exact duplicate polygon", line_number)
        seen_polygons.add(signature)
        polygons.append(Polygon(class_id, points.astype(np.float32), line_number, area))
    return polygons


def discover_split(split_dir: Path) -> tuple[Path, Path] | None:
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if image_dir.is_dir() and label_dir.is_dir():
        return image_dir, label_dir
    return None


def source_stats(records: list[ImageAudit]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for source in sorted({record.source for record in records}):
        group = [record for record in records if record.source == source]
        polygons = sum(record.polygon_count for record in group)
        points = sum(record.point_count for record in group)
        output[source] = {
            "images": len(group),
            "polygons": polygons,
            "avg_polygons_per_image": round(polygons / max(len(group), 1), 3),
            "avg_points_per_polygon": round(points / max(polygons, 1), 3),
        }
    return output


def audit_dataset(
    dataset_root: Path,
    sam2_stems: set[str],
    expected_class: int,
    min_area: float,
    max_points: int,
    sequence_gap: int,
    accepted_issues: AcceptedIssues | None = None,
    splits: tuple[str, ...] = ("train", "valid", "val", "test"),
) -> tuple[list[ImageAudit], list[Issue], dict[str, object]]:
    dataset_root = dataset_root.resolve()
    dataset = dataset_root.name
    issues: list[Issue] = []
    records: list[ImageAudit] = []
    stem_splits: dict[str, list[str]] = defaultdict(list)

    for split_name in splits:
        found = discover_split(dataset_root / split_name)
        if found is None:
            continue
        image_dir, label_dir = found
        images: dict[str, Path] = {}
        for path in image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                key = path.stem.lower()
                if key in images:
                    add_issue(issues, dataset, split_name, path.stem, "error", "duplicate_image_stem", f"Also found {images[key].name}")
                images[key] = path
        labels = {path.stem.lower(): path for path in label_dir.glob("*.txt")}

        for key in sorted(images.keys() - labels.keys()):
            path = images[key]
            add_issue(issues, dataset, split_name, path.stem, "error", "missing_label", "Image has no matching TXT")
        for key in sorted(labels.keys() - images.keys()):
            path = labels[key]
            add_issue(issues, dataset, split_name, path.stem, "error", "missing_image", "TXT has no matching image")

        for key, image_path in sorted(images.items()):
            label_path = labels.get(key)
            source = "sam2_new" if image_path.stem in sam2_stems else "base"
            polygons = []
            if label_path is not None:
                polygons = parse_label(
                    label_path, dataset, split_name, image_path.stem, issues,
                    expected_class, min_area, max_points, accepted_issues,
                )
            point_count = sum(len(polygon.points) for polygon in polygons)
            record = ImageAudit(
                dataset=dataset,
                split=split_name,
                stem=image_path.stem,
                source=source,
                image_path=image_path,
                label_path=label_path,
                polygons=polygons,
                polygon_count=len(polygons),
                point_count=point_count,
                average_points=point_count / max(len(polygons), 1),
                summed_area_ratio=sum(polygon.area for polygon in polygons),
            )
            records.append(record)
            stem_splits[image_path.stem].append(split_name)

    for stem, splits in stem_splits.items():
        if len(set(splits)) > 1:
            unique_splits = sorted(set(splits))
            for split in unique_splits:
                add_issue(issues, dataset, split, stem, "error", "duplicate_across_splits", f"Found in splits: {unique_splits}")

    counts = [record.polygon_count for record in records if record.label_path is not None]
    if counts:
        median = statistics.median(counts)
        mad = statistics.median(abs(value - median) for value in counts)
        threshold = max(10.0, 3.5 * 1.4826 * mad)
        for record in records:
            if record.label_path is not None and abs(record.polygon_count - median) > threshold:
                add_issue(
                    issues, dataset, record.split, record.stem, "warning", "polygon_count_outlier",
                    f"Count={record.polygon_count}, median={median:.1f}, threshold={threshold:.1f}",
                )

    numbered = []
    for record in records:
        match = re.search(r"(\d+)$", record.stem)
        if match:
            numbered.append((int(match.group(1)), record))
    numbered.sort(key=lambda item: item[0])
    for index, (number, record) in enumerate(numbered):
        for other_number, other in numbered[index + 1:]:
            gap = other_number - number
            if gap > sequence_gap:
                break
            if record.split != other.split:
                detail = f"{record.stem} ({record.split}) vs {other.stem} ({other.split}), gap={gap}"
                add_issue(issues, dataset, record.split, record.stem, "warning", "adjacent_cross_split", detail)

    issue_counts_by_stem = Counter((issue.split, issue.stem) for issue in issues)
    for record in records:
        record.issue_count = issue_counts_by_stem[(record.split, record.stem)]

    polygon_total = sum(record.polygon_count for record in records)
    point_total = sum(record.point_count for record in records)
    summary: dict[str, object] = {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "images": len(records),
        "labels": sum(record.label_path is not None for record in records),
        "polygons": polygon_total,
        "avg_polygons_per_image": round(polygon_total / max(len(records), 1), 3),
        "avg_points_per_polygon": round(point_total / max(polygon_total, 1), 3),
        "issue_rows": len(issues),
        "issue_images": len({issue.stem for issue in issues}),
        "errors": sum(issue.severity == "error" for issue in issues),
        "warnings": sum(issue.severity == "warning" for issue in issues),
        "issues_by_code": dict(sorted(Counter(issue.code for issue in issues).items())),
        "sources": source_stats(records),
    }
    return records, issues, summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_review_queue(
    records: list[ImageAudit],
    issues: list[Issue],
    overlay_paths: dict[tuple[str, str], Path],
    output_dir: Path,
) -> list[dict[str, object]]:
    issue_map: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in issues:
        issue_map[(issue.split, issue.stem)].append(issue)

    rows: list[dict[str, object]] = []
    for record in records:
        record_issues = issue_map[(record.split, record.stem)]
        if not record_issues:
            continue
        counts = Counter(issue.code for issue in record_issues)
        has_error = any(issue.severity == "error" for issue in record_issues)
        geometry_codes = {
            "self_intersection", "duplicate_vertex", "tiny_polygon", "polygon_count_outlier", "excessive_points"
        }
        has_geometry_issue = bool(geometry_codes.intersection(counts))
        if has_error:
            priority_rank, priority = 0, "P0-error"
        elif record.source == "sam2_new" and has_geometry_issue:
            priority_rank, priority = 1, "P1-sam2-geometry"
        elif counts["polygon_count_outlier"] or counts["duplicate_vertex"]:
            priority_rank, priority = 2, "P2-geometry"
        else:
            priority_rank, priority = 3, "P3-review"

        actions = []
        if has_error:
            actions.append("repair label syntax/pairing before training")
        if counts["polygon_count_outlier"]:
            actions.append("check the full image for over-segmentation or missing instances")
        if counts["self_intersection"] or counts["duplicate_vertex"]:
            actions.append("redraw the listed polygon lines without loops or repeated path vertices")
        if counts["tiny_polygon"]:
            actions.append("verify tiny masks and remove only clear artifacts")
        if counts["excessive_points"]:
            actions.append("simplify the polygon while preserving its mask")
        if counts["adjacent_cross_split"]:
            actions.append("keep adjacent frames in the same dataset split")

        line_details = []
        for issue in record_issues:
            suffix = f":{issue.line}" if issue.line is not None else ""
            line_details.append(f"{issue.code}{suffix}")
        overlay_path = overlay_paths.get((record.split, record.stem))
        rows.append(
            {
                "priority_rank": priority_rank,
                "priority": priority,
                "source": record.source,
                "split": record.split,
                "stem": record.stem,
                "polygons": record.polygon_count,
                "issue_count": len(record_issues),
                "issue_lines": ";".join(line_details),
                "recommended_action": "; ".join(actions) or "manual visual review",
                "image_path": str(record.image_path),
                "label_path": str(record.label_path or ""),
                "overlay_path": str(overlay_path.relative_to(output_dir)) if overlay_path else "",
            }
        )
    return sorted(rows, key=lambda row: (int(row["priority_rank"]), -int(row["issue_count"]), str(row["stem"])))


def save_jpeg(path: Path, image: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"Cannot encode overlay: {path}")
    encoded.tofile(str(path))


def render_overlay(record: ImageAudit, issue_codes: list[str], output_path: Path) -> None:
    image = read_image(record.image_path)
    height, width = image.shape[:2]
    overlay = image.copy()
    color = (40, 190, 80) if record.source == "base" else (30, 150, 240)
    for polygon in record.polygons:
        points = np.rint(polygon.points * np.asarray([width, height])).astype(np.int32)
        cv2.fillPoly(overlay, [points], color)
    image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    for polygon in record.polygons:
        points = np.rint(polygon.points * np.asarray([width, height])).astype(np.int32)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)

    banner_height = 72
    cv2.rectangle(image, (0, 0), (width, banner_height), (18, 18, 18), -1)
    title = f"{record.stem} | {record.split} | {record.source} | polygons={record.polygon_count}"
    cv2.putText(image, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    codes = ", ".join(issue_codes[:5]) if issue_codes else "sample: no automatic issue"
    cv2.putText(image, codes, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (80, 210, 255), 1, cv2.LINE_AA)

    max_width = 1100
    if width > max_width:
        ratio = max_width / width
        image = cv2.resize(image, (max_width, round(height * ratio)), interpolation=cv2.INTER_AREA)
    save_jpeg(output_path, image)


def select_overlays(records: list[ImageAudit], issues: list[Issue], maximum: int, sample_valid: int) -> list[ImageAudit]:
    ignored = {"adjacent_cross_split"}
    by_stem: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in issues:
        by_stem[(issue.split, issue.stem)].append(issue)

    def score(record: ImageAudit) -> tuple[int, int, str]:
        relevant = [issue for issue in by_stem[(record.split, record.stem)] if issue.code not in ignored]
        return (
            sum(100 if issue.severity == "error" else 10 for issue in relevant),
            len(relevant),
            record.stem,
        )

    problematic = [record for record in records if score(record)[0] > 0]
    problematic.sort(key=score, reverse=True)
    selected = problematic[:maximum]

    remaining = maximum - len(selected)
    if remaining > 0 and sample_valid > 0:
        selected_keys = {(record.split, record.stem) for record in selected}
        candidates = [
            record for record in records
            if (record.split, record.stem) not in selected_keys and not by_stem[(record.split, record.stem)]
        ]
        random.Random(42).shuffle(candidates)
        selected.extend(candidates[: min(remaining, sample_valid)])
    return selected


def write_html_report(
    output_path: Path,
    summary: dict[str, object],
    issues: list[Issue],
    overlay_rows: list[tuple[ImageAudit, Path]],
) -> None:
    code_rows = "".join(
        f"<tr><td>{html.escape(code)}</td><td>{count}</td></tr>"
        for code, count in summary["issues_by_code"].items()
    ) or '<tr><td colspan="2">No issues</td></tr>'
    issue_map: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in issues:
        issue_map[(issue.split, issue.stem)].append(issue)
    cards = []
    for record, overlay_path in overlay_rows:
        record_issues = issue_map[(record.split, record.stem)]
        codes = sorted({issue.code for issue in record_issues})
        details = "<br>".join(
            html.escape(f"[{issue.severity}] {issue.code}: {issue.detail}")
            for issue in record_issues[:8]
        ) or "Automatic checks passed; review sample."
        relative = overlay_path.relative_to(output_path.parent).as_posix()
        cards.append(
            f'<article><img src="{html.escape(relative)}" loading="lazy" alt="{html.escape(record.stem)}">'
            f'<h3>{html.escape(record.stem)}</h3>'
            f'<p>{html.escape(record.split)} / {html.escape(record.source)} / {record.polygon_count} polygons</p>'
            f'<p class="codes">{html.escape(", ".join(codes) or "sample")}</p><details><summary>Details</summary><p>{details}</p></details></article>'
        )

    source_rows = "".join(
        f"<tr><td>{html.escape(source)}</td><td>{values['images']}</td><td>{values['avg_polygons_per_image']}</td><td>{values['avg_points_per_polygon']}</td></tr>"
        for source, values in summary["sources"].items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(summary['dataset']))} label audit</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f7f8;color:#172126}}main{{max-width:1320px;margin:auto;padding:24px}}
h1{{margin:0 0 6px}}.muted{{color:#607078}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}
.metric{{background:white;border:1px solid #dbe2e5;padding:14px;border-radius:6px}}.metric strong{{display:block;font-size:24px;margin-top:5px}}
.tables{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:9px;border:1px solid #dbe2e5;text-align:left}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-top:18px}}
article{{background:white;border:1px solid #dbe2e5;border-radius:6px;overflow:hidden;padding-bottom:12px}}article img{{display:block;width:100%;aspect-ratio:16/11;object-fit:cover}}
article h3,article p,article details{{margin:9px 12px}}.codes{{color:#b43d27;font-size:13px}}a{{color:#087a62}}
</style></head><body><main>
<h1>{html.escape(str(summary['dataset']))} label quality audit</h1><p class="muted">Generated from {html.escape(str(summary['dataset_root']))}</p>
<p><a href="review_queue.csv">review_queue.csv</a> · <a href="issues.csv">issues.csv</a> · <a href="image_stats.csv">image_stats.csv</a> · <a href="summary.json">summary.json</a></p>
<section class="summary">
<div class="metric">Images<strong>{summary['images']}</strong></div><div class="metric">Polygons<strong>{summary['polygons']}</strong></div>
<div class="metric">Issue images<strong>{summary['issue_images']}</strong></div><div class="metric">Errors<strong>{summary['errors']}</strong></div>
<div class="metric">Warnings<strong>{summary['warnings']}</strong></div><div class="metric">Avg polygons/image<strong>{summary['avg_polygons_per_image']}</strong></div>
</section>
<section class="tables"><div><h2>Issues</h2><table><tr><th>Code</th><th>Count</th></tr>{code_rows}</table></div>
<div><h2>Sources</h2><table><tr><th>Source</th><th>Images</th><th>Polygons/image</th><th>Points/polygon</th></tr>{source_rows}</table></div></section>
<h2>Review overlays</h2><section class="grid">{''.join(cards)}</section>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def write_report(
    output_dir: Path,
    records: list[ImageAudit],
    issues: list[Issue],
    summary: dict[str, object],
    max_overlays: int,
    sample_valid: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(
        output_dir / "issues.csv",
        [asdict(issue) for issue in issues],
        ["dataset", "split", "stem", "severity", "code", "line", "detail"],
    )
    stats_rows = [
        {
            "dataset": record.dataset,
            "split": record.split,
            "stem": record.stem,
            "source": record.source,
            "image_path": str(record.image_path),
            "label_path": str(record.label_path or ""),
            "polygons": record.polygon_count,
            "points": record.point_count,
            "avg_points": round(record.average_points, 3),
            "summed_area_ratio": round(record.summed_area_ratio, 6),
            "issues": record.issue_count,
        }
        for record in records
    ]
    write_csv(
        output_dir / "image_stats.csv",
        stats_rows,
        ["dataset", "split", "stem", "source", "image_path", "label_path", "polygons", "points", "avg_points", "summed_area_ratio", "issues"],
    )

    issue_codes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for issue in issues:
        issue_codes[(issue.split, issue.stem)].append(issue.code)
    selected = select_overlays(records, issues, max_overlays, sample_valid)
    overlay_rows: list[tuple[ImageAudit, Path]] = []
    for record in selected:
        output_path = overlay_dir / f"{record.split}_{record.stem}.jpg"
        render_overlay(record, sorted(set(issue_codes[(record.split, record.stem)])), output_path)
        overlay_rows.append((record, output_path))
    review_rows = build_review_queue(
        records,
        issues,
        {(record.split, record.stem): path for record, path in overlay_rows},
        output_dir,
    )
    write_csv(
        output_dir / "review_queue.csv",
        review_rows,
        [
            "priority_rank", "priority", "source", "split", "stem", "polygons", "issue_count",
            "issue_lines", "recommended_action", "image_path", "label_path", "overlay_path",
        ],
    )
    write_html_report(output_dir / "index.html", summary, issues, overlay_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root containing train/valid/test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sam2-labels", type=Path, default=PROJECT_ROOT / "待标注_100张" / "labels")
    parser.add_argument("--expected-class", type=int, default=0)
    parser.add_argument("--min-area", type=float, default=1e-5)
    parser.add_argument("--max-points", type=int, default=200)
    parser.add_argument("--sequence-gap", type=int, default=10)
    parser.add_argument("--max-overlays", type=int, default=60)
    parser.add_argument("--sample-valid", type=int, default=20)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "val", "test"),
        default=("train", "valid", "val", "test"),
        help="Dataset splits to audit; use '--splits train valid' to preserve a frozen test split.",
    )
    parser.add_argument(
        "--review-state", type=Path, action="append",
        help="Review state containing accepted ('kept') items; repeat for multiple review batches.",
    )
    parser.add_argument(
        "--review-geometry", type=Path, action="append",
        help="Geometry CSV paired with --review-state; repeat in the same order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset does not exist: {dataset_root}")
    output_dir = (args.output or (PROJECT_ROOT / "reports" / f"{dataset_root.name}_label_quality")).resolve()
    sam2_stems = {path.stem for path in args.sam2_labels.glob("*.txt")} if args.sam2_labels.is_dir() else set()
    review_states = args.review_state or []
    review_geometries = args.review_geometry or []
    if len(review_states) != len(review_geometries):
        raise SystemExit("Provide the same number of --review-state and --review-geometry arguments")
    accepted_issues: AcceptedIssues = {}
    accepted_items = 0
    for state_path, geometry_path in zip(review_states, review_geometries):
        batch_issues, batch_items = load_review_acceptances(state_path.resolve(), geometry_path.resolve())
        for key, codes in batch_issues.items():
            accepted_issues.setdefault(key, set()).update(codes)
        accepted_items += batch_items

    records, issues, summary = audit_dataset(
        dataset_root,
        sam2_stems,
        args.expected_class,
        args.min_area,
        args.max_points,
        args.sequence_gap,
        accepted_issues,
        tuple(args.splits),
    )
    summary["accepted_review_items"] = accepted_items
    summary["accepted_warning_codes"] = sum(len(codes) for codes in accepted_issues.values())
    write_report(output_dir, records, issues, summary, args.max_overlays, args.sample_valid)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {output_dir / 'index.html'}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
