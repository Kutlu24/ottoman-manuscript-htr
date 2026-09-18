from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..data.schema import LineAnnotation

LINE_GEOMETRY_DIM = 9
IMAGE_GEOMETRY_DIM = 12
GEOMETRY_DIM = LINE_GEOMETRY_DIM + IMAGE_GEOMETRY_DIM


def ink_mask(gray: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Soft ink mask from a [0, 1]-normalized grayscale crop (dark = ink)."""
    inv = 1.0 - gray
    if threshold is None:
        threshold = float(inv.mean() + inv.std())
    return inv > threshold


def ink_density(gray: np.ndarray) -> float:
    return float((1.0 - gray).mean())


def connected_component_stats(mask: np.ndarray) -> tuple[int, float]:
    """(component_count, mean_component_size) for a binary ink mask."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return 0, 0.0
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    return n, float(np.mean(sizes))


def stroke_direction_histogram(gray: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Gradient-orientation histogram, weighted by gradient magnitude -- a coarse stroke-direction proxy."""
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    hist, _ = np.histogram(angle, bins=n_bins, range=(-np.pi, np.pi), weights=magnitude)
    total = hist.sum()
    return (hist / total if total > 0 else hist).astype(np.float32)


def baseline_slope(mask: np.ndarray) -> float:
    """Least-squares slope of ink-pixel y vs. x -- a crude baseline-tilt estimate."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def _all_dot_points(line: LineAnnotation) -> list[tuple[float, float]]:
    """Dots annotated directly on the line (the real Tier-2 target -- see
    LineAnnotation.dot_points) plus any nested under word/char boxes, for
    pages where those were genuinely drawn. Word/char boundaries
    themselves are NOT used as features here -- see the redesign note
    below."""
    points = [(p.x, p.y) for p in line.dot_points]
    points += [(p.x, p.y) for w in line.words for c in w.chars for p in c.dot_points]
    return points


def line_geometry_vector(line: LineAnnotation) -> np.ndarray:
    """Layout-only summary from the annotation alone -- no pixels needed.

    Redesigned around what Tier-2 annotation actually produces for this
    corpus (docs/corpus_collection_plan.md): word and even character
    bounding boxes are not reliably determinable by eye in connected
    Ottoman cursive hands (this is a property of Arabic-script cursive
    writing -- medial letterforms are designed to connect via an
    unbroken ligature stroke -- not a defect specific to any one
    scribe), so this vector no longer depends on them. What is real and
    used:
      - line.bbox.width/height, baseline_angle_deg: from the line box
        itself (Tier-1, ALTO-derived, no manual annotation needed).
      - n_chars: from the line's own transcription text, not from a
        character count promised by visual char boxes -- text is
        already correct and available (Tier-1) and does not depend on
        segmentation at all.
      - n_dots and their distribution across the line: the one thing
        Tier-2 annotation actually adds here. Dots are visually
        discrete, unlike connected letterforms, so drawing a point on
        each visible one is a tractable annotation task -- point
        annotation, not a segmentation problem.
    """
    n_chars = len([c for c in line.text if not c.isspace()])
    dots = _all_dot_points(line)
    n_dots = len(dots)

    x0, x1 = line.bbox.x0, line.bbox.x1
    span = (x1 - x0) or 1.0
    thirds = [0, 0, 0]
    for x, _y in dots:
        idx = min(2, max(0, int(((x - x0) / span) * 3)))
        thirds[idx] += 1
    third_density = [t / n_dots if n_dots else 0.0 for t in thirds]

    return np.array(
        [
            line.bbox.width,
            line.bbox.height,
            line.baseline_angle_deg,
            float(n_chars),
            float(n_dots),
            n_dots / n_chars if n_chars else 0.0,
            *third_density,
        ],
        dtype=np.float32,
    )


def image_geometry_vector(gray: np.ndarray) -> np.ndarray:
    """Ink-only summary from a line crop's pixels -- usable even without word/char box annotations."""
    mask = ink_mask(gray)
    n_components, mean_component_size = connected_component_stats(mask)
    direction_hist = stroke_direction_histogram(gray)
    return np.concatenate(
        [
            np.array([ink_density(gray), float(n_components), mean_component_size, baseline_slope(mask)], dtype=np.float32),
            direction_hist,
        ]
    ).astype(np.float32)


def combined_geometry_vector(line: LineAnnotation, gray: np.ndarray) -> np.ndarray:
    return np.concatenate([line_geometry_vector(line), image_geometry_vector(gray)])
