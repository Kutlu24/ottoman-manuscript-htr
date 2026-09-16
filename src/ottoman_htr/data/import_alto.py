from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .schema import BBox, LineAnnotation, PageAnnotation


def convert_alto_file(
    alto_path: str | Path,
    *,
    image_path: str | None = None,
    manuscript_id: str | None = None,
    folio: str | None = None,
    scribe: str | None = None,
) -> PageAnnotation:
    """Converts one ALTO XML page (as exported by eScriptorium/Kraken, e.g. OpenITI MAKHZAN) into
    a PageAnnotation. Only line-level geometry is populated -- `words` stays empty, since
    line-level-only annotated ALTO has no real word boxes to draw from (see Tier 1 vs. Tier 2 in
    docs/corpus_collection_plan.md). `manuscript_id`/`folio`/`scribe` aren't reliably encoded in
    ALTO itself; pass them explicitly when the source corpus's manifest has them, otherwise they
    fall back to guesses from the filename."""
    alto_path = Path(alto_path)
    root = ET.parse(alto_path).getroot()

    lines: list[LineAnnotation] = []
    texts: list[str] = []
    for line_el in root.findall(".//{*}TextLine"):
        text = _line_text(line_el)
        bbox = _line_bbox(line_el)
        angle = _baseline_angle_deg(_parse_points(line_el.get("BASELINE", "")))
        lines.append(LineAnnotation(bbox=bbox, text=text, baseline_angle_deg=angle))
        texts.append(text)

    stem = alto_path.stem
    return PageAnnotation(
        image_path=image_path or _resolve_image_path(root, alto_path),
        manuscript_id=manuscript_id or stem.split("_")[0],
        folio=folio or stem,
        transcription="\n".join(texts),
        scribe=scribe,
        lines=lines,
    )


def convert_directory(alto_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Converts every *.xml in alto_dir to a PageAnnotation JSON of the same stem in output_dir.
    Page images still need to be placed under output_dir/images/ by hand (or a corpus-specific
    script) before python -m ottoman_htr.data.validate will find them."""
    alto_dir = Path(alto_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for alto_path in sorted(alto_dir.glob("*.xml")):
        page = convert_alto_file(alto_path)
        out_path = output_dir / f"{alto_path.stem}.json"
        page.to_json(out_path)
        written.append(out_path)
    return written


def _line_text(line_el: ET.Element) -> str:
    strings = [s.get("CONTENT", "") for s in line_el.findall("{*}String")]
    if strings:
        return " ".join(s for s in strings if s)
    return (line_el.get("CONTENT") or "").strip()


def _line_bbox(line_el: ET.Element) -> BBox:
    hpos, vpos, width, height = (line_el.get(k) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
    if None not in (hpos, vpos, width, height):
        x0, y0 = float(hpos), float(vpos)
        return BBox(x0=x0, y0=y0, x1=x0 + float(width), y1=y0 + float(height))

    polygon = line_el.find("{*}Shape/{*}Polygon")
    if polygon is not None and polygon.get("POINTS"):
        points = _parse_points(polygon.get("POINTS"))
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    raise ValueError(f"TextLine {line_el.get('ID')!r} has neither HPOS/VPOS/WIDTH/HEIGHT nor a polygon")


def _parse_points(raw: str) -> list[tuple[float, float]]:
    """Parses an ALTO BASELINE/POINTS attribute. The ALTO spec allows "x1,y1 x2,y2 ..." pairs,
    but real eScriptorium/Kraken exports (confirmed against the OpenITI MAKHZAN corpus) instead
    emit a flat, comma-free "x1 y1 x2 y2 ..." list -- handle both."""
    if not raw:
        return []
    if "," in raw:
        points = []
        for token in raw.split():
            x_str, _, y_str = token.partition(",")
            if y_str:
                points.append((float(x_str), float(y_str)))
        return points
    values = raw.split()
    return [(float(values[i]), float(values[i + 1])) for i in range(0, len(values) - 1, 2)]


def _baseline_angle_deg(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    slope, _ = np.polyfit(xs, ys, 1)
    return math.degrees(math.atan(slope))


def _resolve_image_path(root: ET.Element, alto_path: Path) -> str:
    file_name_el = root.find(".//{*}sourceImageInformation/{*}fileName")
    if file_name_el is not None and file_name_el.text:
        return f"images/{Path(file_name_el.text.strip()).name}"
    for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        if alto_path.with_suffix(ext).exists():
            return f"images/{alto_path.with_suffix(ext).name}"
    return f"images/{alto_path.stem}.tif"  # best-effort guess -- verify against the real corpus manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a directory of ALTO XML files (e.g. OpenITI MAKHZAN, eScriptorium exports) "
            "into PageAnnotation JSON, one file per page. Word/char/dot geometry is left empty -- "
            "line-level-only ALTO has no real word boxes (see docs/corpus_collection_plan.md, Tier 2)."
        )
    )
    parser.add_argument("alto_dir", type=Path, help="Directory of *.xml ALTO files")
    parser.add_argument("output_dir", type=Path, help="Directory to write PageAnnotation JSON into")
    args = parser.parse_args()

    written = convert_directory(args.alto_dir, args.output_dir)
    print(f"wrote {len(written)} PageAnnotation file(s) to {args.output_dir}")
    print(f"remember to place the corresponding page images under {args.output_dir}/images/ "
          "before running ottoman_htr.data.validate")


if __name__ == "__main__":
    main()
