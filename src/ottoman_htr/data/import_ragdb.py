from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .schema import BBox, LineAnnotation, PageAnnotation


def convert_ragdb_export(
    export_dir: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    manuscript_id: str = "ragdb",
    use_corrected: bool = True,
    min_width: float = 20,
    min_height: float = 10,
    keep_empty: bool = False,
    exclude_pages: set[str] | frozenset[str] = frozenset(),
    dropped: list[tuple[str, int, str]] | None = None,
) -> list[Path]:
    """Converts a page/line export from ottoman-rag's correction tool (one JSON per document under
    `export_dir/pages/` and `export_dir/lines/`, as written by ArtifactData's `out_dir`) into
    PageAnnotation JSON + copied images, ready for `via_export`. Each line doc carries a Kraken
    `bbox` ([x0, y0, x1, y1]), a `draft_text` and a `corrected_text`; only line-level geometry
    exists, so words/dots stay empty for the Tier-2 pass to fill. Page images are looked up in
    `images_dir` by the page doc's `asset_id` (`<asset_id>.<ext>`).

    `use_corrected=False` keeps the raw Kraken draft even where a human correction exists. Lines
    still at status "pending" carry the draft text unchanged, i.e. it is NOT verified ground truth.

    Kraken's segmenter on non-manuscript or low-resolution images emits many junk "lines" (specks,
    stray strokes). Lines are dropped when their box is narrower than `min_width` or shorter than
    `min_height` pixels, or when they have no text (`keep_empty` keeps those); a line a human has
    already touched (status other than "pending") is never dropped. Pages named in `exclude_pages`
    are skipped outright (e.g. printed rather than handwritten material, which no size filter can
    detect). Each drop is appended to `dropped` as (page_id, line_index, reason) when given; a
    skipped page appears once with line_index -1 and reason "excluded_page". A page left with no
    lines is skipped."""
    export_dir = Path(export_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    lines_by_page: dict[str, list[dict]] = {}
    for path in sorted((export_dir / "lines").glob("*.json")):
        doc = json.loads(path.read_text())
        lines_by_page.setdefault(doc["page_id"], []).append(doc)

    written = []
    for page_path in sorted((export_dir / "pages").glob("*.json")):
        page_id = page_path.stem
        page_doc = json.loads(page_path.read_text())
        if page_id in exclude_pages:
            if dropped is not None:
                dropped.append((page_id, -1, "excluded_page"))
            continue
        lines = []
        for doc in sorted(lines_by_page.get(page_id, []), key=lambda d: d["line_index"]):
            x0, y0, x1, y1 = doc["bbox"]
            text = doc["corrected_text"] if use_corrected else doc["draft_text"]
            reason = None
            if doc.get("status", "pending") == "pending":
                reason = _junk_reason(x1 - x0, y1 - y0, text, min_width, min_height, keep_empty)
            if reason:
                if dropped is not None:
                    dropped.append((page_id, doc["line_index"], reason))
                continue
            lines.append(LineAnnotation(bbox=BBox(x0, y0, x1, y1), text=text))
        if not lines:
            continue

        image_src = _find_image(images_dir, page_doc["asset_id"])
        image_name = f"{page_id}{image_src.suffix}"
        (output_dir / "images" / image_name).write_bytes(image_src.read_bytes())

        page = PageAnnotation(
            image_path=f"images/{image_name}",
            manuscript_id=manuscript_id,
            folio=page_id,
            transcription="\n".join(line.text for line in lines),
            lines=lines,
        )
        out_path = output_dir / f"{page_id}.json"
        page.to_json(out_path)
        written.append(out_path)
    return written


def _junk_reason(
    width: float, height: float, text: str, min_width: float, min_height: float, keep_empty: bool
) -> str | None:
    if width < min_width:
        return "too_narrow"
    if height < min_height:
        return "too_short"
    if not keep_empty and not text.strip():
        return "empty_text"
    return None


def _find_image(images_dir: Path, asset_id: str) -> Path:
    matches = sorted(images_dir.glob(f"{asset_id}.*"))
    if not matches:
        raise FileNotFoundError(f"no image for asset {asset_id} in {images_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an ottoman-rag correction-tool export (pages/ + lines/ JSON docs) into "
            "PageAnnotation JSON + images/, ready for ottoman_htr.data.via_export."
        )
    )
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manuscript-id", default="ragdb")
    parser.add_argument("--draft-only", action="store_true", help="ignore corrected_text")
    parser.add_argument("--min-width", type=float, default=20)
    parser.add_argument("--min-height", type=float, default=10)
    parser.add_argument("--keep-empty", action="store_true", help="keep lines with no text")
    parser.add_argument(
        "--exclude-page", action="append", default=[], metavar="PAGE_ID",
        help="skip this page entirely (repeatable), e.g. printed rather than handwritten pages",
    )
    args = parser.parse_args()

    dropped: list[tuple[str, int, str]] = []
    written = convert_ragdb_export(
        args.export_dir,
        args.images_dir,
        args.output_dir,
        manuscript_id=args.manuscript_id,
        use_corrected=not args.draft_only,
        min_width=args.min_width,
        min_height=args.min_height,
        keep_empty=args.keep_empty,
        exclude_pages=frozenset(args.exclude_page),
        dropped=dropped,
    )
    reasons = Counter(reason for _, _, reason in dropped)
    print(f"wrote {len(written)} PageAnnotation files to {args.output_dir}")
    print(f"dropped {len(dropped)} lines/pages: {dict(reasons)}")


if __name__ == "__main__":
    main()
