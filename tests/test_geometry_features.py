import numpy as np

from ottoman_htr.data.schema import BBox, CharAnnotation, LineAnnotation, WordAnnotation
from ottoman_htr.geometry.features import (
    GEOMETRY_DIM,
    combined_geometry_vector,
    ink_density,
    line_geometry_vector,
)


def _sample_line() -> LineAnnotation:
    char = CharAnnotation(bbox=BBox(0, 0, 10, 20), label="ب")
    word = WordAnnotation(bbox=BBox(0, 0, 40, 20), text="بير", chars=[char])
    return LineAnnotation(bbox=BBox(0, 0, 200, 30), text="بير", words=[word])


def test_line_geometry_vector_shape():
    vec = line_geometry_vector(_sample_line())
    assert vec.shape == (13,)
    assert vec[0] == 200  # line width


def test_ink_density_bounds():
    gray = np.random.rand(32, 32).astype("float32")
    assert 0.0 <= ink_density(gray) <= 1.0


def test_combined_geometry_vector_matches_declared_dim():
    gray = np.random.rand(30, 200).astype("float32")
    vec = combined_geometry_vector(_sample_line(), gray)
    assert vec.shape == (GEOMETRY_DIM,)
