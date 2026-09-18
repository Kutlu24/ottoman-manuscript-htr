import numpy as np

from ottoman_htr.data.schema import BBox, LineAnnotation, Point
from ottoman_htr.geometry.features import (
    GEOMETRY_DIM,
    LINE_GEOMETRY_DIM,
    combined_geometry_vector,
    ink_density,
    line_geometry_vector,
)


def _sample_line() -> LineAnnotation:
    # Tier-2 for this corpus annotates dots directly on the line (word/char
    # boxes are not reliably determinable by eye in connected Ottoman
    # cursive hands -- see docs/corpus_collection_plan.md).
    return LineAnnotation(
        bbox=BBox(0, 0, 200, 30),
        text="بير",
        dot_points=[Point(x=20, y=10), Point(x=100, y=12), Point(x=180, y=8)],
    )


def test_line_geometry_vector_shape():
    vec = line_geometry_vector(_sample_line())
    assert vec.shape == (LINE_GEOMETRY_DIM,)
    assert vec[0] == 200  # line width
    assert vec[3] == 3  # n_chars, from len(line.text) -- no char annotation needed
    assert vec[4] == 3  # n_dots
    assert vec[5] == 1.0  # dots per char


def test_ink_density_bounds():
    gray = np.random.rand(32, 32).astype("float32")
    assert 0.0 <= ink_density(gray) <= 1.0


def test_combined_geometry_vector_matches_declared_dim():
    gray = np.random.rand(30, 200).astype("float32")
    vec = combined_geometry_vector(_sample_line(), gray)
    assert vec.shape == (GEOMETRY_DIM,)
