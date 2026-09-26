import numpy as np
from PIL import Image

from ottoman_htr.data.dot_candidates import build_dot_candidate_project, find_dot_candidates
from ottoman_htr.data.schema import BBox, LineAnnotation, PageAnnotation


def _page_with_dot_and_stroke():
    gray = np.full((100, 300), 0.9, dtype=np.float32)
    gray[47:53, 50:56] = 0.1  # 6x6 blob, mid-line: a dot
    gray[48:52, 100:220] = 0.1  # long horizontal stroke: a ligature, not a dot
    gray[20:26, 150:156] = 0.1  # blob outside the line box
    return gray


def test_finds_dot_sized_blob_but_not_stroke_or_outside_blob():
    line = LineAnnotation(bbox=BBox(0, 30, 300, 70), text="x")  # height 40
    found = find_dot_candidates(_page_with_dot_and_stroke(), line)
    assert found == [(53.0, 50.0)]


def test_thin_specks_are_rejected_and_blob_near_edge_outside_band_is_kept():
    gray = np.full((100, 300), 0.9, dtype=np.float32)
    gray[31:37, 50:56] = 0.1  # centre 4px below the top of a 40px line: outside the 6% edge band, kept
    gray[30:32, 150:156] = 0.1  # too thin (height 2)
    line = LineAnnotation(bbox=BBox(0, 30, 300, 70), text="x")
    assert find_dot_candidates(gray, line) == [(53.0, 34.0)]


def test_project_contains_dot_point_regions_alongside_line_boxes(tmp_path):
    ann_dir = tmp_path / "ann"
    (ann_dir / "images").mkdir(parents=True)
    Image.fromarray((_page_with_dot_and_stroke() * 255).astype(np.uint8)).save(ann_dir / "images" / "p.png")
    PageAnnotation(
        image_path="images/p.png", manuscript_id="m", folio="p", transcription="x",
        lines=[LineAnnotation(bbox=BBox(0, 30, 300, 70), text="x")],
    ).to_json(ann_dir / "p.json")

    project = build_dot_candidate_project(ann_dir)

    regions = next(iter(project["_via_img_metadata"].values()))["regions"]
    assert [r["region_attributes"]["level"] for r in regions] == ["line", "dot"]
    assert regions[1]["shape_attributes"] == {"name": "point", "cx": 53, "cy": 50}


def test_low_resolution_pages_are_left_out_of_the_dot_project(tmp_path):
    ann_dir = tmp_path / "ann"
    (ann_dir / "images").mkdir(parents=True)
    Image.fromarray((_page_with_dot_and_stroke() * 255).astype(np.uint8)).save(ann_dir / "images" / "p.png")
    PageAnnotation(
        image_path="images/p.png", manuscript_id="m", folio="p", transcription="x",
        lines=[LineAnnotation(bbox=BBox(0, 30, 300, 70), text="x")],  # median line height 40
    ).to_json(ann_dir / "p.json")

    skipped = []
    kept = build_dot_candidate_project(ann_dir, min_median_line_height=40, skipped=skipped)
    assert len(kept["_via_img_metadata"]) == 1 and skipped == []

    dropped = build_dot_candidate_project(ann_dir, min_median_line_height=41, skipped=skipped)
    assert dropped["_via_img_metadata"] == {} and dropped["_via_image_id_list"] == []
    assert skipped == [("p.png", 40.0)]
