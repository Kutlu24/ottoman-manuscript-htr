from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import PageAnnotation

LEVEL_OPTIONS = {
    "line": "line (reference -- do not edit or move)",
    "word": "word",
    "char": "character",
    "dot": "diacritical dot",
}


def build_via_project(annotation_dir: str | Path, project_name: str = "tier2") -> dict:
    """Builds a VIA2 (VGG Image Annotator) project pre-loaded with each page's existing Tier-1
    line boxes as reference regions, so an annotator can draw word/char/dot regions on top of
    them without re-locating the lines. See docs/corpus_collection_plan.md, Tier 2."""
    annotation_dir = Path(annotation_dir)
    img_metadata: dict[str, dict] = {}

    for ann_path in sorted(annotation_dir.glob("*.json")):
        page = PageAnnotation.from_json(ann_path)
        image_path = annotation_dir / page.image_path
        filename = Path(page.image_path).name
        size = image_path.stat().st_size if image_path.exists() else 0
        key = f"{filename}{size}"

        regions = [
            {
                "shape_attributes": {
                    "name": "rect",
                    "x": round(line.bbox.x0),
                    "y": round(line.bbox.y0),
                    "width": round(line.bbox.width),
                    "height": round(line.bbox.height),
                },
                "region_attributes": {"level": "line", "text": line.text},
            }
            for line in page.lines
        ]

        img_metadata[key] = {
            "filename": filename,
            "size": size,
            "regions": regions,
            "file_attributes": {},
        }

    return {
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_region_shape": True,
                    "show_image_policy": "all",
                },
                "image": {
                    "region_label": "level",
                    "region_color": "level",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION",
                },
            },
            "core": {"buffer_size": 18, "filepath": {}, "default_filepath": ""},
            "project": {"name": project_name},
        },
        "_via_img_metadata": img_metadata,
        "_via_attributes": {
            "region": {
                "level": {
                    "type": "radio",
                    "description": "Annotation level",
                    "options": LEVEL_OPTIONS,
                    "default_options": {"word": True},
                },
                "text": {
                    "type": "text",
                    "description": (
                        "Word text (as written) or single character label. Leave empty for dot "
                        "regions."
                    ),
                },
            },
            "file": {},
        },
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": list(img_metadata.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a VIA2 (VGG Image Annotator, robots.ox.ac.uk/~vgg/software/via) project file "
            "for Tier-2 word/char/dot annotation, pre-loaded with each page's existing line "
            "boxes as reference. Open the generated .json in VIA (File > Load project), point "
            "VIA at the annotation_dir's images/ folder when prompted, annotate, then export via "
            "Annotation > Export Annotations (as json) and feed that file to "
            "ottoman_htr.data.via_import."
        )
    )
    parser.add_argument("annotation_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--project-name", default="tier2")
    args = parser.parse_args()

    project = build_via_project(args.annotation_dir, args.project_name)
    args.output_json.write_text(json.dumps(project, ensure_ascii=False, indent=2))
    print(f"wrote VIA project ({len(project['_via_img_metadata'])} images) to {args.output_json}")


if __name__ == "__main__":
    main()
