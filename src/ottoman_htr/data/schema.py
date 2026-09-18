from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Point:
    x: float
    y: float


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class CharAnnotation:
    bbox: BBox
    label: str | None = None  # ground-truth character, when available (needed for geometric_contrastive_loss)
    dot_points: list[Point] = field(default_factory=list)
    contour: list[Point] = field(default_factory=list)


@dataclass
class WordAnnotation:
    bbox: BBox
    text: str
    chars: list[CharAnnotation] = field(default_factory=list)


@dataclass
class LineAnnotation:
    bbox: BBox
    text: str
    baseline_angle_deg: float = 0.0
    words: list[WordAnnotation] = field(default_factory=list)
    # Diacritical dots annotated directly against the line, with no word/char
    # attribution -- see docs/corpus_collection_plan.md's Tier-2 note: word
    # and even character boundaries are often not reliably determinable by
    # eye in connected Ottoman cursive hands (letters share connecting
    # ligature strokes by the nature of the script, not as a defect of any
    # one scribe), so dot position is the annotation Tier-2 actually asks
    # for. `words` (and any char-level dot_points nested inside it) remains
    # for pages where word/char boxes genuinely were drawn.
    dot_points: list[Point] = field(default_factory=list)


@dataclass
class PageAnnotation:
    image_path: str
    manuscript_id: str
    folio: str
    transcription: str
    scribe: str | None = None
    lines: list[LineAnnotation] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> PageAnnotation:
        raw = json.loads(Path(path).read_text())
        return cls(
            image_path=raw["image_path"],
            manuscript_id=raw["manuscript_id"],
            folio=raw["folio"],
            transcription=raw["transcription"],
            scribe=raw.get("scribe"),
            lines=[_line_from_dict(d) for d in raw.get("lines", [])],
        )


def _point_from_dict(d: dict) -> Point:
    return Point(x=d["x"], y=d["y"])


def _bbox_from_dict(d: dict) -> BBox:
    return BBox(x0=d["x0"], y0=d["y0"], x1=d["x1"], y1=d["y1"])


def _char_from_dict(d: dict) -> CharAnnotation:
    return CharAnnotation(
        bbox=_bbox_from_dict(d["bbox"]),
        label=d.get("label"),
        dot_points=[_point_from_dict(p) for p in d.get("dot_points", [])],
        contour=[_point_from_dict(p) for p in d.get("contour", [])],
    )


def _word_from_dict(d: dict) -> WordAnnotation:
    return WordAnnotation(
        bbox=_bbox_from_dict(d["bbox"]),
        text=d["text"],
        chars=[_char_from_dict(c) for c in d.get("chars", [])],
    )


def _line_from_dict(d: dict) -> LineAnnotation:
    return LineAnnotation(
        bbox=_bbox_from_dict(d["bbox"]),
        text=d["text"],
        baseline_angle_deg=d.get("baseline_angle_deg", 0.0),
        words=[_word_from_dict(w) for w in d.get("words", [])],
        dot_points=[_point_from_dict(p) for p in d.get("dot_points", [])],
    )
