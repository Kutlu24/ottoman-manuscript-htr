from pathlib import Path

from ottoman_htr.data.validate import validate_directory
from ottoman_htr.geometry.features import line_geometry_vector

EXAMPLE_DIR = Path(__file__).parent.parent / "data" / "example"


def test_example_annotation_parses_cleanly():
    report = validate_directory(EXAMPLE_DIR)
    assert report.errors == []
    assert report.n_pages == 1
    assert report.n_lines == 2
    assert report.n_words == 5
    assert report.n_chars_with_dots == 4  # ت, ی, ن, ف


def test_example_line_geometry_is_well_formed():
    report_path = next(EXAMPLE_DIR.glob("*.json"))
    from ottoman_htr.data.schema import PageAnnotation

    page = PageAnnotation.from_json(report_path)
    for line in page.lines:
        vec = line_geometry_vector(line)
        assert vec.shape == (13,)
        assert vec[0] > 0  # line width
