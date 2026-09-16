from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..data.schema import LineAnnotation

LINE_GEOMETRY_DIM = 13
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


def line_geometry_vector(line: LineAnnotation) -> np.ndarray:
    """Layout-only summary from the annotation alone (word/char boxes, dot counts) -- no pixels needed."""
    word_widths = [w.bbox.width for w in line.words]
    word_gaps = [line.words[i + 1].bbox.x0 - line.words[i].bbox.x1 for i in range(len(line.words) - 1)]
    char_widths = [c.bbox.width for w in line.words for c in w.chars]
    n_chars = sum(len(w.chars) for w in line.words)
    n_dots = sum(len(c.dot_points) for w in line.words for c in w.chars)

    def _stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.asarray(values, dtype=np.float32)
        return float(arr.mean()), float(arr.std())

    word_w_mean, word_w_std = _stats(word_widths)
    gap_mean, gap_std = _stats(word_gaps)
    char_w_mean, char_w_std = _stats(char_widths)

    return np.array(
        [
            line.bbox.width,
            line.bbox.height,
            line.baseline_angle_deg,
            float(len(line.words)),
            word_w_mean,
            word_w_std,
            gap_mean,
            gap_std,
            float(n_chars),
            char_w_mean,
            char_w_std,
            float(n_dots),
            n_dots / n_chars if n_chars else 0.0,
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
