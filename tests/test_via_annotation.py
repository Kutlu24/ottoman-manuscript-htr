import json

from ottoman_htr.data.schema import BBox, LineAnnotation, PageAnnotation
from ottoman_htr.data.validate import validate_directory
from ottoman_htr.data.via_export import build_via_project
from ottoman_htr.data.via_import import import_via_project


def _build_tier1_page(tmp_path):
    ann_dir = tmp_path / "tier1"
    images_dir = ann_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "page1.jpg").write_bytes(b"fake-jpeg-bytes")

    page = PageAnnotation(
        image_path="images/page1.jpg",
        manuscript_id="86",
        folio="p1",
        transcription="دولت عالیه",
        lines=[LineAnnotation(bbox=BBox(100, 200, 900, 280), text="دولت عالیه")],
    )
    page.to_json(ann_dir / "page1.json")
    return ann_dir


def test_build_via_project_includes_line_reference_region(tmp_path):
    ann_dir = _build_tier1_page(tmp_path)
    project = build_via_project(ann_dir)

    assert len(project["_via_img_metadata"]) == 1
    entry = next(iter(project["_via_img_metadata"].values()))
    assert entry["filename"] == "page1.jpg"
    assert len(entry["regions"]) == 1
    region = entry["regions"][0]
    assert region["region_attributes"] == {"level": "line", "text": "دولت عالیه"}
    assert region["shape_attributes"] == {"name": "rect", "x": 100, "y": 200, "width": 800, "height": 80}
    assert "level" in project["_via_attributes"]["region"]


def _via_export_for(ann_dir, word_region, char_regions, dot_regions=()):
    project = build_via_project(ann_dir)
    key = next(iter(project["_via_img_metadata"]))
    project["_via_img_metadata"][key]["regions"].append(word_region)
    project["_via_img_metadata"][key]["regions"].extend(char_regions)
    project["_via_img_metadata"][key]["regions"].extend(dot_regions)
    return project


def _rect(level, x, y, w, h, text=""):
    return {
        "shape_attributes": {"name": "rect", "x": x, "y": y, "width": w, "height": h},
        "region_attributes": {"level": level, "text": text},
    }


def _point(level, cx, cy):
    return {
        "shape_attributes": {"name": "point", "cx": cx, "cy": cy},
        "region_attributes": {"level": level, "text": ""},
    }


def test_import_via_project_reconstructs_word_char_dot_hierarchy(tmp_path):
    ann_dir = _build_tier1_page(tmp_path)

    # one word "بت" made of two chars: ب (no dot) and ت (two dots), inside the line bbox
    word = _rect("word", 150, 210, 100, 60, text="بت")
    char_te = _rect("char", 200, 210, 50, 60, text="ت")   # rightmost -> read first (RTL)
    char_be = _rect("char", 150, 210, 50, 60, text="ب")
    dots = [_point("dot", 210, 215), _point("dot", 220, 215)]  # both inside char_te's bbox

    via_json_path = tmp_path / "export.json"
    project = _via_export_for(ann_dir, word, [char_te, char_be], dots)
    via_json_path.write_text(json.dumps(project, ensure_ascii=False))

    out_dir = tmp_path / "tier2_out"
    written = import_via_project(via_json_path, ann_dir, out_dir)
    assert len(written) == 1

    page = PageAnnotation.from_json(out_dir / "page1.json")
    assert len(page.lines) == 1
    line = page.lines[0]
    assert len(line.words) == 1
    w = line.words[0]
    assert w.text == "بت"
    assert [c.label for c in w.chars] == ["ت", "ب"]  # RTL order preserved
    te, be = w.chars
    assert len(te.dot_points) == 2
    assert len(be.dot_points) == 0

    report = validate_directory(out_dir)
    assert report.errors == []
    assert report.n_chars_with_dots == 1


def test_import_via_project_nests_chars_directly_under_line_without_word_boxes(tmp_path):
    """Word boundaries are often not reliably determinable by eye in
    cursive Ottoman hands (docs/corpus_collection_plan.md) -- an
    annotator can skip word-level boxes entirely and draw char/dot
    regions straight against the line. import_via_project must fall
    back to one synthetic whole-line word rather than dropping the
    chars."""
    ann_dir = _build_tier1_page(tmp_path)

    # no word region at all -- two chars annotated directly inside the line bbox
    char_te = _rect("char", 200, 210, 50, 60, text="ت")
    char_be = _rect("char", 150, 210, 50, 60, text="ب")
    dots = [_point("dot", 210, 215)]

    project = build_via_project(ann_dir)
    key = next(iter(project["_via_img_metadata"]))
    project["_via_img_metadata"][key]["regions"].extend([char_te, char_be, *dots])
    via_json_path = tmp_path / "export.json"
    via_json_path.write_text(json.dumps(project, ensure_ascii=False))

    out_dir = tmp_path / "tier2_out"
    import_via_project(via_json_path, ann_dir, out_dir)

    page = PageAnnotation.from_json(out_dir / "page1.json")
    line = page.lines[0]
    assert len(line.words) == 1  # one synthetic whole-line word
    synthetic_word = line.words[0]
    assert synthetic_word.bbox == line.bbox
    assert [c.label for c in synthetic_word.chars] == ["ت", "ب"]  # RTL order preserved
    assert len(synthetic_word.chars[0].dot_points) == 1

    report = validate_directory(out_dir)
    assert report.errors == []


def test_import_via_project_attaches_dots_directly_to_line_without_word_or_char_boxes(tmp_path):
    """The actual chosen Tier-2 scope for this corpus: neither word nor
    character boundaries are reliably determinable by eye in this
    connected cursive hand (docs/corpus_collection_plan.md), so an
    annotator marks only the visible diacritical dots, directly against
    the line. import_via_project must attach these to
    LineAnnotation.dot_points, not drop them for lack of a parent char."""
    ann_dir = _build_tier1_page(tmp_path)

    # no word, no char regions -- three dots marked directly inside the line bbox
    dots = [_point("dot", 150, 215), _point("dot", 300, 220), _point("dot", 700, 210)]

    project = build_via_project(ann_dir)
    key = next(iter(project["_via_img_metadata"]))
    project["_via_img_metadata"][key]["regions"].extend(dots)
    via_json_path = tmp_path / "export.json"
    via_json_path.write_text(json.dumps(project, ensure_ascii=False))

    out_dir = tmp_path / "tier2_out"
    import_via_project(via_json_path, ann_dir, out_dir)

    page = PageAnnotation.from_json(out_dir / "page1.json")
    line = page.lines[0]
    assert line.words == []  # no word/char structure at all
    assert len(line.dot_points) == 3
    assert {(p.x, p.y) for p in line.dot_points} == {(150.0, 215.0), (300.0, 220.0), (700.0, 210.0)}

    report = validate_directory(out_dir)
    assert report.errors == []
    assert report.n_lines_with_direct_dots == 1
    assert report.n_dots_direct == 3


def test_import_via_project_skips_pages_not_in_export(tmp_path):
    ann_dir = _build_tier1_page(tmp_path)
    project = build_via_project(ann_dir)
    # drop the only image entry -> export "covers" no pages
    project["_via_img_metadata"] = {}
    via_json_path = tmp_path / "export.json"
    via_json_path.write_text(json.dumps(project))

    out_dir = tmp_path / "tier2_out"
    written = import_via_project(via_json_path, ann_dir, out_dir)
    assert written == []
