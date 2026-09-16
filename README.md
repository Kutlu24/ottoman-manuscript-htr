# Ottoman Manuscript HTR — Geometric & Fuzzy Representation

Research prototype for **University of Bern, Digital Humanities Center**. Not to be confused with
[`ottoman-rag`](../ottoman-rag) (a RAG system that *consumes* existing, pre-trained HTR/OCR
models). This project builds the recognition model itself, for **handwritten** Ottoman
manuscripts (el yazması eserler), not printed/modern Ottoman text.

## Research question

> Can uncertainty-preserving geometric representations improve the recognition and
> transcription of Ottoman handwritten manuscripts compared with conventional
> binarization-based OCR?

Classical OCR commits to a binary ink/background decision very early (`grayscale → threshold →
OCR`). For historical manuscripts — faded ink, stained paper, touching letters, missing dots,
scribe-specific style — that early commitment throws away exactly the evidence a human reader
relies on. The hypothesis here is that keeping that evidence as *fuzzy membership* (e.g. "82%
ink, 91% letter-shaped, 63% dot") for as long as possible, and only resolving it once visual,
geometric, and linguistic evidence are combined, degrades more gracefully on damaged pages than
a model that binarizes up front. Full framing, the three-way ablation design (binary OCR vs.
neural OCR vs. this geometric+fuzzy model), and the diagrams behind the code layout below are in
[`docs/architecture.md`](docs/architecture.md).

## Architecture

```
Manuscript line image
        │
        ├──────────────────────┐
        ▼                      ▼
  ImageEncoder (CNN)     GeometricEncoder (MLP)
  → visual class dist.   → geometric class dist.
        │                      │
        └─────────┬────────────┘
                   ▼
           FuzzyFusionLayer
        (learned evidence weighting)
                   │
                   ▼
        fused per-timestep membership
                   │
                   ▼
               CTC decoding
                   │
                   ▼
        (optional) CharNGramLanguageModel
             re-scores candidates
                   │
                   ▼
              Transcription
```

See `src/ottoman_htr/model.py` for the wiring and `docs/architecture.md` for why the fusion
happens at the class-membership level rather than immediately after the encoders.

## Data format

One `PageAnnotation` JSON per manuscript page, next to its image (`src/ottoman_htr/data/schema.py`):
image path + full transcription + per-line/word/char bounding boxes + dot positions. This is the
"IMAGE + TRANSKRİPSİYON + GEOMETRİ" triple needed to train anything here — without paired
geometry annotations, only `image_geometry_vector` (pixel-only features) is available, not the
full `line_geometry_vector` (layout features).

A worked example — one page, annotated down to per-character boxes and dot positions, with the
annotation conventions (coordinate system, RTL ordering, when to fill in `dot_points`/`contour`)
— is in [`data/example/`](data/example/). For the real ~150-page corpus, drop `PageAnnotation`
JSON files + an `images/` folder into a directory and check they all parse with:

```bash
python -m ottoman_htr.data.validate path/to/annotation_dir
```

Source pages typically arrive as line-level ALTO XML (e.g. from OpenITI MAKHZAN or an
eScriptorium export) rather than this JSON directly — convert a directory of `*.xml` files with:

```bash
python -m ottoman_htr.data.import_alto path/to/alto_dir path/to/output_dir
```

This fills in line bounding boxes, baseline angle, and text; `words` is left empty (line-level
ALTO has no real word boxes) until a Tier-2 manual annotation pass adds them — see
[`docs/corpus_collection_plan.md`](docs/corpus_collection_plan.md).

Once a larger converted directory exists, pick a capped, manuscript-diverse ~150-page
train/val/test split from it with:

```bash
python -m ottoman_htr.data.select_corpus path/to/converted_dir path/to/split_dir \
    --metadata path/to/makhzan_metadata.tsv
```

This caps any single dominant manuscript's contribution, then assigns whole manuscripts (never
splitting one scribe's pages across sets) to train/val/test, preferring at least one non-Naskh
manuscript in val and test when available.

Tier-1 (line-level) is enough to train the model as-is; the geometric layout features in
`line_geometry_vector` and the letter-level embedding idea need word/char/dot boxes, which
line-level ALTO doesn't have. For a manuscript-diverse subset, generate a
[VIA](https://www.robots.ox.ac.uk/~vgg/software/via/) annotation project pre-loaded with the
existing line boxes as reference:

```bash
python -m ottoman_htr.data.via_export path/to/annotation_dir path/to/project.json
```

and merge the exported word/char/dot regions back once annotated:

```bash
python -m ottoman_htr.data.via_import path/to/export.json path/to/annotation_dir path/to/output_dir
```

Full annotation-tool workflow (nesting-by-geometry rules, what `level`/`text` values to use) is
in [`docs/corpus_collection_plan.md`](docs/corpus_collection_plan.md#tier-2-annotation-workflow).

## Status

Trained against real data, not just tested against it — architecture, data schema, losses, and
tests pass on synthetic data, and the whole pipeline (ALTO import → corpus selection/split →
`OttomanLineDataset` → model forward/backward) has been run end to end against the real
[OpenITI MAKHZAN](https://doi.org/10.5281/zenodo.19861912) Ottoman Turkish subset (CC BY-NC-SA;
241 real manuscript pages, 26 manuscripts) — see
[`docs/corpus_collection_plan.md`](docs/corpus_collection_plan.md) for what was verified and
what's still open.

A real learning-rate sweep + extended run on the Tier-1 split (105 train / 23 val pages) found a
working reference config — `lr=3e-3`, ~130–150 epochs — reaching **val CER ≈ 0.79**, with val CER
plateauing (not improving) past ~150 epochs. Full numbers, and the dataset-caching fix that took
one epoch from 213s down to ~6s on GPU, are in
[`docs/training_notes.md`](docs/training_notes.md). 0.79 CER is a real learning signal, not a
usable transcription accuracy — the next lever is more/better data (Tier-2 annotation), not more
epochs.

The three-way ablation from `docs/architecture.md` (binary OCR vs. neural vs. geometric+fuzzy)
now has a runner, smoke-tested end to end on real data:

```bash
python -m ottoman_htr.ablation path/to/train_dir path/to/val_dir \
    --epochs 150 --lr 3e-3 --output-dir runs/ablation
```

Trains all three arms (`binary`/`neural`/`fuzzy` — see `model.build_model`) on identical
data/hyperparameters and writes a `best_val_cer`/`best_val_wer` comparison to
`ablation_summary.json`. Not yet run for the full 150 epochs per arm (~3× a single training
run's time) — that's the next real result to produce.

## Running

```bash
pip install -e ".[dev]"
pytest
python -m ottoman_htr.train /path/to/annotation_dir --epochs 10
```
