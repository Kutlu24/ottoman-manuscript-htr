from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .schema import LineAnnotation, PageAnnotation
from .via_export import build_via_project

# All size/position limits are fractions of the line's own height, not absolute pixels: scans in
# one corpus come from different repositories at very different resolutions, and an absolute-pixel
# filter tuned on one page starves the others (see docs/corpus_collection_plan.md, Tier 2).
MIN_SIDE_FRAC = 0.08
MAX_SIDE_FRAC = 0.40
MAX_ASPECT = 2.0
MIN_FILL = 0.45
EDGE_FRAC = 0.06  # top/bottom band dominated by ascender tips and neighbouring-line bleed
MIN_SIDE_PX = 2  # below this a component is scan noise regardless of line height
# A dot is ~10-20% of line height. Below ~25 px median line height that is 2-4 px, the same scale as
# scan grain and JPEG noise: on the real 190-400 px wide thumbnails this produced 15-25 "dots" per
# line covering the whole page. Such pages are left out of the dot project rather than reviewed.
MIN_MEDIAN_LINE_HEIGHT = 25


def find_dot_candidates(gray: np.ndarray, line: LineAnnotation) -> list[tuple[float, float]]:
    """Dot-like connected components inside one line's box, as page-coordinate (cx, cy) centers.

    `gray` is the whole page as a [0, 1] float array (dark = ink). Ink is picked by comparing each
    pixel to its local mean (window ~ line height), so uneven paper tone and stains don't flood the
    mask the way one global threshold would. A draft, not ground truth: precision and recall vary
    with scribal density and scan quality."""
    x0, y0 = int(round(line.bbox.x0)), int(round(line.bbox.y0))
    x1, y1 = int(round(line.bbox.x1)), int(round(line.bbox.y1))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, gray.shape[1]), min(y1, gray.shape[0])
    crop = gray[y0:y1, x0:x1]
    h = y1 - y0
    if crop.size == 0 or h < 4:
        return []

    window = max(3, int(h) | 1)
    local_mean = ndimage.uniform_filter(crop, size=window, mode="nearest")
    contrast = max(float(crop.std()) * 0.5, 0.03)
    ink = crop < (local_mean - contrast)

    labeled, n = ndimage.label(ink)
    if n == 0:
        return []
    candidates = []
    for idx, sl in enumerate(ndimage.find_objects(labeled), start=1):
        ch, cw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if min(ch, cw) < MIN_SIDE_PX:
            continue
        if not (MIN_SIDE_FRAC * h <= ch <= MAX_SIDE_FRAC * h and MIN_SIDE_FRAC * h <= cw <= MAX_SIDE_FRAC * h):
            continue
        if max(ch, cw) / min(ch, cw) > MAX_ASPECT:
            continue
        area = int((labeled[sl] == idx).sum())
        if area / (ch * cw) < MIN_FILL:
            continue
        cy = (sl[0].start + sl[0].stop) / 2
        if not (EDGE_FRAC * h <= cy <= (1 - EDGE_FRAC) * h):
            continue
        candidates.append((x0 + (sl[1].start + sl[1].stop) / 2, y0 + cy))
    return candidates


def build_dot_candidate_project(
    annotation_dir: str | Path,
    project_name: str = "tier2_dots",
    *,
    min_median_line_height: float = MIN_MEDIAN_LINE_HEIGHT,
    skipped: list[tuple[str, float]] | None = None,
) -> dict:
    """VIA2 project like via_export.build_via_project, plus every detected candidate as a
    `level=dot` point region, for a review pass (delete false positives, add missed dots).

    Pages whose median line height is under `min_median_line_height` px are too low-resolution for
    dot detection to mean anything and are left out of the project entirely; each is appended to
    `skipped` as (page filename, median line height) when given."""
    annotation_dir = Path(annotation_dir)
    project = build_via_project(annotation_dir, project_name)
    for ann_path in sorted(annotation_dir.glob("*.json")):
        page = PageAnnotation.from_json(ann_path)
        image_path = annotation_dir / page.image_path
        key = f"{Path(page.image_path).name}{image_path.stat().st_size if image_path.exists() else 0}"
        median_height = float(np.median([line.bbox.height for line in page.lines]))
        if median_height < min_median_line_height:
            del project["_via_img_metadata"][key]
            project["_via_image_id_list"].remove(key)
            if skipped is not None:
                skipped.append((Path(page.image_path).name, median_height))
            continue
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        regions = project["_via_img_metadata"][key]["regions"]
        for line in page.lines:
            for cx, cy in find_dot_candidates(gray, line):
                regions.append({
                    "shape_attributes": {"name": "point", "cx": round(cx), "cy": round(cy)},
                    "region_attributes": {"level": "dot", "text": ""},
                })
    return project


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a VIA2 project with auto-detected dot candidates (level=dot points) on top of "
            "each page's line reference boxes. A draft to review, not ground truth."
        )
    )
    parser.add_argument("annotation_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--project-name", default="tier2_dots")
    parser.add_argument("--min-line-height", type=float, default=MIN_MEDIAN_LINE_HEIGHT)
    args = parser.parse_args()

    skipped: list[tuple[str, float]] = []
    project = build_dot_candidate_project(
        args.annotation_dir, args.project_name, min_median_line_height=args.min_line_height, skipped=skipped
    )
    args.output_json.write_text(json.dumps(project, ensure_ascii=False, indent=2))
    per_page = {
        e["filename"]: sum(1 for r in e["regions"] if r["region_attributes"]["level"] == "dot")
        for e in project["_via_img_metadata"].values()
    }
    print(f"wrote {args.output_json}: {sum(per_page.values())} dot candidates over {len(per_page)} pages")
    for name, count in per_page.items():
        print(f"  {name}: {count}")
    for name, height in skipped:
        print(f"  skipped {name}: median line height {height:.0f}px < {args.min_line_height:g}px (too low-res)")


if __name__ == "__main__":
    main()
