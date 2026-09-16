from pathlib import Path

from ottoman_htr.data.import_alto import _parse_points, convert_alto_file, convert_directory
from ottoman_htr.data.validate import validate_directory

FIXTURE = Path(__file__).parent / "fixtures" / "sample_line_level.xml"


def test_parse_points_handles_flat_and_comma_formats():
    # real eScriptorium/Kraken ALTO exports (confirmed against OpenITI MAKHZAN) use the flat form
    assert _parse_points("175 280 1215 270") == [(175.0, 280.0), (1215.0, 270.0)]
    # the ALTO spec's own "x,y x,y" form is also accepted, in case some source uses it
    assert _parse_points("175,280 1215,270") == [(175.0, 280.0), (1215.0, 270.0)]
    assert _parse_points("") == []


def test_convert_alto_file_extracts_lines_and_text():
    page = convert_alto_file(FIXTURE)

    assert page.image_path == "images/manuscript_0002_folio_014r.tif"
    assert len(page.lines) == 2
    assert page.lines[0].text == "دولت عالیه"
    assert page.lines[1].text == "علم و عرفان"
    assert page.transcription == "دولت عالیه\nعلم و عرفان"


def test_convert_alto_file_bbox_from_hpos_vpos():
    page = convert_alto_file(FIXTURE)
    bbox = page.lines[0].bbox
    assert (bbox.x0, bbox.y0, bbox.x1, bbox.y1) == (175, 222, 1215, 288)


def test_convert_alto_file_baseline_angle_is_slightly_negative():
    page = convert_alto_file(FIXTURE)
    # BASELINE "175,280 1215,270" slopes gently upward left-to-right -> small negative angle
    assert -2.0 < page.lines[0].baseline_angle_deg < 0.0


def test_convert_alto_file_no_words_populated():
    page = convert_alto_file(FIXTURE)
    assert all(line.words == [] for line in page.lines)


def test_convert_directory_writes_valid_annotations(tmp_path):
    output_dir = tmp_path / "converted"
    written = convert_directory(FIXTURE.parent, output_dir)

    assert len(written) == 1
    (output_dir / "images").mkdir()
    (output_dir / "images" / "manuscript_0002_folio_014r.tif").touch()

    report = validate_directory(output_dir)
    assert report.errors == []
    assert report.n_pages == 1
    assert report.n_lines == 2
