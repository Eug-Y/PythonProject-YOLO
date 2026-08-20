from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.pdf)
    matrix = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)
    for index, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(args.output_dir / f"page-{index}.png")
    print(f"PAGES={document.page_count}")


if __name__ == "__main__":
    main()
