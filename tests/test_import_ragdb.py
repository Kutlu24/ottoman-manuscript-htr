import json

from ottoman_htr.data.import_ragdb import convert_ragdb_export
from ottoman_htr.data.schema import PageAnnotation
from ottoman_htr.data.via_export import build_via_project


def _write_export(tmp_path):
    export = tmp_path / "export"
    (export / "pages").mkdir(parents=True)
    (export / "lines").mkdir()
    images = tmp_path / "img"
    images.mkdir()
    (images / "abc123.jpg").write_bytes(b"fake-jpeg-bytes")
    (export / "pages" / "pg1.json").write_text(json.dumps({"asset_id": "abc123", "filename": "pg1.jpg"}))
    for idx, (draft, corrected) in enumerate([("اول", "اولى"), ("ایکنجی", "ایکنجی")]):
        (export / "lines" / f"pg1__{idx:03d}.json").write_text(json.dumps({
            "page_id": "pg1", "line_index": idx, "draft_text": draft,
            "corrected_text": corrected, "bbox": [10, 20 + idx * 40, 300, 50 + idx * 40], "status": "pending",
        }))
    return export, images


def test_convert_ragdb_export_feeds_via_export(tmp_path):
    export, images = _write_export(tmp_path)
    out = tmp_path / "out"

    written = convert_ragdb_export(export, images, out)

    assert [p.name for p in written] == ["pg1.json"]
    assert (out / "images" / "pg1.jpg").read_bytes() == b"fake-jpeg-bytes"
    page = PageAnnotation.from_json(written[0])
    assert page.image_path == "images/pg1.jpg"
    assert [l.text for l in page.lines] == ["اولى", "ایکنجی"]
    assert page.lines[0].bbox.x1 == 300

    project = build_via_project(out)
    entry = next(iter(project["_via_img_metadata"].values()))
    assert entry["filename"] == "pg1.jpg" and len(entry["regions"]) == 2


def test_draft_only_ignores_corrections(tmp_path):
    export, images = _write_export(tmp_path)
    written = convert_ragdb_export(export, images, tmp_path / "out", use_corrected=False)
    assert PageAnnotation.from_json(written[0]).lines[0].text == "اول"


def _write_junk_export(tmp_path):
    export = tmp_path / "export"
    (export / "pages").mkdir(parents=True)
    (export / "lines").mkdir()
    images = tmp_path / "img"
    images.mkdir()
    for pid in ("pg1", "pg2"):
        (images / f"{pid}.jpg").write_bytes(b"x")
        (export / "pages" / f"{pid}.json").write_text(json.dumps({"asset_id": pid, "filename": f"{pid}.jpg"}))
    rows = [  # page, idx, bbox, text, status
        ("pg1", 0, [0, 0, 300, 40], "keep", "pending"),
        ("pg1", 1, [0, 0, 5, 40], "narrow", "pending"),
        ("pg1", 2, [0, 0, 300, 4], "short", "pending"),
        ("pg1", 3, [0, 0, 300, 40], "  ", "pending"),
        ("pg1", 4, [0, 0, 5, 4], "", "corrected"),  # human-touched: never dropped
        ("pg2", 0, [0, 0, 300, 40], "", "pending"),  # page's only line is empty -> page skipped
    ]
    for pid, idx, bbox, text, status in rows:
        (export / "lines" / f"{pid}__{idx:03d}.json").write_text(json.dumps({
            "page_id": pid, "line_index": idx, "draft_text": text, "corrected_text": text,
            "bbox": bbox, "status": status,
        }))
    return export, images


def test_junk_lines_are_dropped_but_human_touched_lines_kept(tmp_path):
    export, images = _write_junk_export(tmp_path)
    dropped = []

    written = convert_ragdb_export(export, images, tmp_path / "out", dropped=dropped)

    assert [p.name for p in written] == ["pg1.json"]
    assert [l.text for l in PageAnnotation.from_json(written[0]).lines] == ["keep", ""]
    assert sorted(dropped) == [
        ("pg1", 1, "too_narrow"), ("pg1", 2, "too_short"), ("pg1", 3, "empty_text"), ("pg2", 0, "empty_text"),
    ]
    assert not (tmp_path / "out" / "images" / "pg2.jpg").exists()


def test_keep_empty_retains_textless_lines(tmp_path):
    export, images = _write_junk_export(tmp_path)
    written = convert_ragdb_export(export, images, tmp_path / "out", keep_empty=True)
    assert [p.name for p in written] == ["pg1.json", "pg2.json"]
