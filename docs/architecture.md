# Architecture notes

## Why fuzzy logic here, and why not instead of OCR

Fuzzy logic is not a replacement for the neural recognizer — it is the layer that combines
multiple *sources of evidence* (visual shape, geometric layout, lexical/grammatical fit) without
forcing any one of them to commit to a hard decision first. The distinction that matters:

- **Binary OCR**: `grayscale → threshold → OCR`. A pixel is ink or it isn't, before the model
  ever sees it.
- **This project**: `grayscale → continuous representation → geometric features → fuzzy
  inference over multiple evidence sources → character/word`.

A region can legitimately be "82% ink, 91% letter-shaped, 63% dot" — and a faded or damaged
region should carry that uncertainty forward rather than being forced to a `1` or `0` at the
preprocessing stage, only for later stages to have no way to recover the lost information.

## Three-way ablation (the actual experiment)

| Model | Pipeline |
|---|---|
| A — Traditional OCR | `scan → binarization → OCR` |
| B — Neural OCR | `raw image → CNN/Transformer → CTC` (no geometry, no fuzzy layer) |
| C — Geometric + Fuzzy (this repo) | `raw image → geometric features → fuzzy fusion → CTC` |

Compare on: Character Error Rate, Word Error Rate, performance on faded ink, performance across
different scribes/hands, performance on synthetically degraded copies of the same page (blur,
stain, contrast loss, dot removal). The interesting result isn't "C wins overall" — it's whether
C degrades *more slowly* than A/B as document quality drops. A negative result (no difference) is
still a defensible, measurable finding about where the error actually comes from.

## Why fusion happens at the membership level, not the embedding level

The conversation that motivated this project sketched a `Multimodal Fusion` block sitting after
separate `Geometric Encoder` and `Fuzzy Logic Layer` branches. In code
(`src/ottoman_htr/model.py`), the vision and geometry encoders each produce their own per-class
membership distribution (`ImageEncoder`/`GeometricEncoder` → linear head → softmax), and
`FuzzyFusionLayer` combines *those distributions*, not the raw embeddings. This keeps each
source's uncertainty visible and separately inspectable right up to the final decision — e.g. you
can log "geometry was confident this was ب, vision was split between ب and ت" — instead of
collapsing everything into one opaque fused vector before any character-level claim is made. This
is what "explainable, not black-box OCR" means concretely here.

## Data format

`PageAnnotation` (`src/ottoman_htr/data/schema.py`) mirrors the annotation triple discussed:

```
IMAGE:
  path to the page scan

TRANSKRİPSİYON:
  full page transcription (and per-line text)

GEOMETRİ:
  line bounding boxes + baseline angle
  word bounding boxes
  char bounding boxes + dot points (+ optional per-char label, for supervised geometry)
```

`ottoman_htr.geometry.features` derives two complementary vectors from this:

- `line_geometry_vector` — layout-only (word count, word/gap widths, char/dot counts), computed
  from the annotation alone, no pixels.
- `image_geometry_vector` — ink-only (density, connected components, stroke-direction histogram,
  baseline slope), computed from the line crop's pixels, no annotation needed.

`combined_geometry_vector` concatenates both (`GEOMETRY_DIM = 25`). A corpus with only
transcriptions and line boxes (no word/char boxes) can still train on `image_geometry_vector`
alone — the char/word-level geometry is what upgrades the model from "line-level HTR with some
hand-crafted features" to the letter-level geometric-embedding idea below.

## Geometric embedding (future work, not yet wired into `model.py`)

The longer-term idea from the original discussion: embed individual *characters* (not just
lines) into a shared geometric space, so the same letter written by different scribes clusters
together. `losses.geometric_contrastive_loss` implements the training signal for this (InfoNCE
over `GeometricEncoder` embeddings, positive pairs = same ground-truth character label) but needs
per-character crops and labels to run — i.e. a corpus with `CharAnnotation.label` filled in, which
is a heavier annotation cost than line-level transcription alone. Worth doing once the line-level
ablation shows the geometric signal helps at all; premature before that.

## What's deliberately not built yet

- No transformer-based Ottoman language model — `language/lm.py` is a character n-gram model,
  enough to test whether LM re-scoring helps at all before justifying a heavier model.
- No real geometric feature extraction from raw contours (no OpenCV dependency in this
  environment) — `geometry/features.py` uses numpy/scipy (gradients, connected components) as a
  first pass; swap in contour/curvature features once there's a labeled corpus to validate them
  against.
- No corpus. Everything here is validated against synthetic tensors/arrays in `tests/`, not real
  manuscript pages. The next concrete step is turning ~150 annotated pages into `PageAnnotation`
  JSON files and running the three-way ablation above.
