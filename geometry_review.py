"""Generate focused visual crops for high-priority polygon review."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from multispectral import PROJECT_ROOT, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "Data_v3_label_quality")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "Data_v3_geometry_review")
    parser.add_argument("--priority", default="P1-sam2-geometry")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "val", "test"),
        help="Only include these dataset splits (default: all splits).",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Limit the number of queued images.")
    parser.add_argument("--skip-images", type=int, default=0, help="Skip this many queued images before limiting.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit the number of polygon rows.")
    return parser.parse_args()


def save_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def duplicate_indices(points: np.ndarray) -> set[int]:
    groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, point in enumerate(np.round(points, 10)):
        groups[(float(point[0]), float(point[1]))].append(index)
    return {index for indices in groups.values() if len(indices) > 1 for index in indices}


def render_crop(image_path: Path, label_path: Path, line_number: int, codes: list[str], output_path: Path) -> None:
    image = read_image(image_path)
    height, width = image.shape[:2]
    line = label_path.read_text(encoding="utf-8").splitlines()[line_number - 1]
    values = np.asarray(line.split()[1:], dtype=np.float64).reshape(-1, 2)
    pixels = np.rint(values * np.asarray([width, height])).astype(np.int32)

    x, y, box_width, box_height = cv2.boundingRect(pixels)
    padding = max(40, round(max(box_width, box_height) * 0.18))
    center_x, center_y = x + box_width // 2, y + box_height // 2
    crop_width = max(box_width + 2 * padding, 240)
    crop_height = max(box_height + 2 * padding, 180)
    left = max(0, center_x - crop_width // 2)
    top = max(0, center_y - crop_height // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    left = max(0, right - crop_width)
    top = max(0, bottom - crop_height)

    overlay = image.copy()
    cv2.fillPoly(overlay, [pixels], (20, 30, 230))
    image = cv2.addWeighted(overlay, 0.32, image, 0.68, 0)
    cv2.polylines(image, [pixels], True, (20, 20, 255), 4, cv2.LINE_AA)
    if "duplicate_vertex" in codes:
        for index in duplicate_indices(values):
            point = tuple(int(value) for value in pixels[index])
            cv2.circle(image, point, 9, (255, 30, 210), -1, cv2.LINE_AA)
            cv2.circle(image, point, 12, (255, 255, 255), 2, cv2.LINE_AA)

    crop = image[top:bottom, left:right]
    max_width = 900
    if crop.shape[1] > max_width:
        ratio = max_width / crop.shape[1]
        crop = cv2.resize(crop, (max_width, round(crop.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
    elif crop.shape[1] < 480:
        ratio = 480 / crop.shape[1]
        crop = cv2.resize(crop, (480, round(crop.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
    banner = np.full((78, crop.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(
        banner, f"{image_path.stem} | line {line_number}", (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )
    code_text = ", ".join(codes)
    base_scale = 0.52
    text_width = cv2.getTextSize(code_text, cv2.FONT_HERSHEY_SIMPLEX, base_scale, 1)[0][0]
    scale = min(base_scale, base_scale * (crop.shape[1] - 28) / max(text_width, 1))
    cv2.putText(banner, code_text, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, scale, (80, 210, 255), 1, cv2.LINE_AA)
    save_jpeg(output_path, np.vstack((banner, crop)))


def write_html(output: Path, rows: list[dict[str, object]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f'<article><img src="{html.escape(str(row["crop_path"]))}" loading="lazy" '
            f'alt="{html.escape(str(row["stem"]))}"><div><h2>{html.escape(str(row["stem"]))}</h2>'
            f'<p>{html.escape(str(row["split"]))} / line {row["line"]} / {html.escape(str(row["codes"]))}</p>'
            f'<p class="path">{html.escape(str(row["label_path"]))}</p></div></article>'
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Data_v3 geometry review</title>
<style>
body{{margin:0;background:#f4f6f7;color:#172126;font-family:Segoe UI,Arial,sans-serif}}
header{{padding:22px 28px;background:#172126;color:white}}h1{{margin:0;font-size:24px}}header p{{margin:6px 0 0;color:#c8d1d5}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;padding:18px}}
article{{background:white;border:1px solid #d7dfe2;border-radius:6px;overflow:hidden}}article img{{width:100%;display:block}}
article div{{padding:12px}}h2{{font-size:17px;margin:0 0 6px}}p{{margin:4px 0}}.path{{font-size:12px;color:#607078;overflow-wrap:anywhere}}
</style></head><body><header><h1>Data_v3 geometry review</h1><p>{len(rows)} high-priority polygon rows</p></header>
<main>{''.join(cards)}</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = args.report.resolve()
    output = args.output.resolve()
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    with (report / "review_queue.csv").open(encoding="utf-8-sig", newline="") as handle:
        queue = [
            row
            for row in csv.DictReader(handle)
            if row["priority"] == args.priority and (not args.splits or row["split"] in args.splits)
        ]
    if args.skip_images < 0:
        raise ValueError("--skip-images must be non-negative")
    if args.skip_images:
        queue = queue[args.skip_images:]
    if args.max_images is not None:
        queue = queue[: args.max_images]

    rows: list[dict[str, object]] = []
    for record in queue:
        by_line: dict[int, set[str]] = defaultdict(set)
        for token in record["issue_lines"].split(";"):
            if ":" not in token:
                continue
            code, raw_line = token.rsplit(":", 1)
            if code in {"self_intersection", "duplicate_vertex", "tiny_polygon", "excessive_points"}:
                by_line[int(raw_line)].add(code)
        for line_number, code_set in sorted(by_line.items()):
            if args.max_rows is not None and len(rows) >= args.max_rows:
                break
            codes = sorted(code_set)
            filename = f"{record['split']}_{record['stem']}_line_{line_number}.jpg"
            crop_path = crops / filename
            render_crop(Path(record["image_path"]), Path(record["label_path"]), line_number, codes, crop_path)
            rows.append(
                {
                    "stem": record["stem"],
                    "split": record["split"],
                    "line": line_number,
                    "codes": ",".join(codes),
                    "label_path": record["label_path"],
                    "crop_path": crop_path.relative_to(output).as_posix(),
                }
            )
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break

    with (output / "geometry_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "split", "line", "codes", "label_path", "crop_path"])
        writer.writeheader()
        writer.writerows(rows)
    write_html(output, rows)
    print(f"Geometry rows: {len(rows)}")
    print(f"Report: {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
