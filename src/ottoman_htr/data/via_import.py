from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import BBox, CharAnnotation, LineAnnotation, PageAnnotation, Point, WordAnnotation


def _center_of(region: dict) -> tuple[float, float]:
    s = region["shape_attributes"]
    name = s.get("name")
    if name in ("point", "circle", "ellipse"):
        return s["cx"], s["cy"]
    return s["x"] + s.get("width", 0) / 2, s["y"] + s.get("height", 0) / 2


def _bbox_of(region: dict) -> BBox:
    s = region["shape_attributes"]
    if s.get("name") != "rect":
        raise ValueError(f"expected a rect region for a word/char annotation, got {s.get('name')!r}")
    x0, y0 = s["x"], s["y"]
    return BBox(x0=x0, y0=y0, x1=x0 + s["width"], y1=y0 + s["height"])


def _center_in_bbox(region: dict, bbox: BBox) -> bool:
    cx, cy = _center_of(region)
    return bbox.x0 <= cx <= bbox.x1 and bbox.y0 <= cy <= bbox.y1


def import_via_project(via_json_path: str | Path, annotation_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Merges word/char/dot regions from a VIA2 export back into the original PageAnnotation
    JSONs, reconstructing the line -> word -> char -> dot hierarchy purely from spatial
    containment (a region's center point falling inside its parent's box) -- VIA itself has no
    notion of nesting, so this is the only source of truth for parentage. Writes updated
    PageAnnotation JSON to output_dir, one per page found in the VIA export; pages from
    annotation_dir that aren't in the export are left untouched (not copied)."""
    via = json.loads(Path(via_json_path).read_text())
    img_metadata: dict[str, dict] = via["_via_img_metadata"]
    by_filename: dict[str, dict] = {entry["filename"]: entry for entry in img_metadata.values()}

    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    written = []
    for ann_path in sorted(annotation_dir.glob("*.json")):
        page = PageAnnotation.from_json(ann_path)
        entry = by_filename.get(Path(page.image_path).name)
        if entry is None:
            continue

        src_image = (annotation_dir / page.image_path).resolve()
        dst_image = images_out / Path(page.image_path).name
        if src_image.exists() and not dst_image.exists() and not dst_image.is_symlink():
            dst_image.symlink_to(src_image)

        regions = entry["regions"]
        word_regions = [r for r in regions if r["region_attributes"].get("level") == "word"]
        char_regions = [r for r in regions if r["region_attributes"].get("level") == "char"]
        dot_regions = [r for r in regions if r["region_attributes"].get("level") == "dot"]

        new_lines = []
        for line in page.lines:
            line_words = [w for w in word_regions if _center_in_bbox(w, line.bbox)]
            words = []
            for w in line_words:
                w_bbox = _bbox_of(w)
                w_chars = sorted(
                    (c for c in char_regions if _center_in_bbox(c, w_bbox)),
                    key=lambda c: -c["shape_attributes"]["x"],  # RTL: rightmost char first
                )
                chars = []
                for c in w_chars:
                    c_bbox = _bbox_of(c)
                    dots = [
                        Point(x=round(cx, 1), y=round(cy, 1))
                        for cx, cy in (_center_of(d) for d in dot_regions)
                        if c_bbox.x0 <= cx <= c_bbox.x1 and c_bbox.y0 <= cy <= c_bbox.y1
                    ]
                    label = c["region_attributes"].get("text") or None
                    chars.append(CharAnnotation(bbox=c_bbox, label=label, dot_points=dots))
                words.append(WordAnnotation(bbox=w_bbox, text=w["region_attributes"].get("text", ""), chars=chars))
            words.sort(key=lambda w: -w.bbox.x0)  # RTL: rightmost word first, matching data/example/README.md
            new_lines.append(
                LineAnnotation(bbox=line.bbox, text=line.text, baseline_angle_deg=line.baseline_angle_deg, words=words)
            )

        updated = PageAnnotation(
            image_path=page.image_path,
            manuscript_id=page.manuscript_id,
            folio=page.folio,
            transcription=page.transcription,
            scribe=page.scribe,
            lines=new_lines,
        )
        out_path = output_dir / ann_path.name
        updated.to_json(out_path)
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a VIA2 export (word/char/dot regions) back into PageAnnotation JSON."
    )
    parser.add_argument("via_json", type=Path, help="VIA export (Annotation > Export Annotations, as json)")
    parser.add_argument("annotation_dir", type=Path, help="Original Tier-2 annotation_dir (line-level only)")
    parser.add_argument("output_dir", type=Path, help="Where to write the updated PageAnnotation JSON")
    args = parser.parse_args()

    written = import_via_project(args.via_json, args.annotation_dir, args.output_dir)
    print(f"wrote {len(written)} updated PageAnnotation file(s) to {args.output_dir}")
    print(f"validate with: python -m ottoman_htr.data.validate {args.output_dir}")


if __name__ == "__main__":
    main()
