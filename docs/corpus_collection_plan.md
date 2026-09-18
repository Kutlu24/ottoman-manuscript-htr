# Corpus collection plan

Goal: ~150 real, annotated manuscript pages to run the three-way ablation in
`docs/architecture.md` (binary OCR vs. neural OCR vs. geometric+fuzzy) and, on a smaller subset,
test the letter-level geometric-embedding hypothesis (`losses.geometric_contrastive_loss`).

## Key decision: bootstrap from OpenITI MAKHZAN, don't digitize from scratch

[**OpenITI MAKHZAN**](https://doi.org/10.5281/zenodo.19861912) (Miller et al., *Journal of Open
Humanities Data* vol. 12, 2026, [10.5334/johd.465](https://doi.org/10.5334/johd.465)) already
contains:

- **26 Ottoman Turkish manuscripts, 241 handwritten pages**, real images + line-level
  segmentation + transcription, exported as **ALTO XML**, built with eScriptorium/Kraken.
- Script breakdown: 229 pages Naskh, 5 Ta'liq, 5 Divani, 2 other — Naskh-dominated, as expected
  for the genre, but with enough non-Naskh pages to sanity-check generalization.
- Some pages have "comprehensive segmentation of all text regions" (main text + margins/glosses);
  most have "segmentation only of the main text body."
- **License: CC BY-NC-SA.** Usable for this non-commercial research prototype; **cannot** be
  redistributed as part of any commercial product or relicensed without separate permission from
  the depositors. Must cite the JOHD paper (14 authors) and the Zenodo DOI wherever the corpus
  or a model trained on it is described. Add a `NOTICE`/citation file once data lands in the
  repo — do not just note it here and forget.

Starting from this instead of commissioning fresh digitization + transcription turns the corpus
problem from "get manuscript images and pay someone to transcribe 150 pages" into "select 150 of
241 existing pages and add the geometry MAKHZAN doesn't have." That's the single biggest lever
on both timeline and cost, so it's the plan's starting assumption — re-verify the live Zenodo
record (page counts, license) before committing further work on top of it, since data papers can
lag the actual deposit.

## What MAKHZAN gives us vs. what we still have to annotate

| Field in `PageAnnotation` | In MAKHZAN? |
|---|---|
| `image_path` | Yes |
| `transcription` (page/line text) | Yes |
| `lines[].bbox`, `lines[].text` | Yes (ALTO XML line elements) |
| `lines[].baseline_angle_deg` | Derivable from ALTO baseline points |
| `lines[].words[]` (word boxes) | **No** |
| `words[].chars[]` (char boxes, labels, dot_points) | **No** |

So MAKHZAN alone is enough for the **core ablation** (Model A/B/C all operate at line level;
`image_geometry_vector` is pixel-only and needs no word/char boxes). It is **not** enough for
`line_geometry_vector`'s layout features (word count, gaps, char widths, dot ratio) or for
`geometric_contrastive_loss`, which need real word/char/dot annotations. Hence a two-tier plan:

- **Tier 1 — all 150 pages, line-level only** (from MAKHZAN via a converter, see below). Drives
  the main A/B/C ablation and the synthetic-degradation robustness test.
- **Tier 2 — a ~20–30 page subset, full word/char/dot annotation** (manual, on top of Tier 1).
  Drives the layout-feature ablation and the geometric-embedding experiment. Start here, not
  with all 150 — it's the expensive part, and the Tier-1 result (does geometry+fuzzy help at
  all?) should gate whether Tier 2 is worth expanding.

## Steps

1. **Download & filter.** Pull OpenITI MAKHZAN from Zenodo, filter to the Ottoman Turkish subset
   (26 manuscripts / 241 pages).
2. **Convert with `python -m ottoman_htr.data.import_alto <alto_dir> <output_dir>`**
   (`src/ottoman_htr/data/import_alto.py`): parses each ALTO XML page's `TextLine` elements
   (`HPOS`/`VPOS`/`WIDTH`/`HEIGHT` or polygon fallback for bbox, `BASELINE` points for
   `baseline_angle_deg`, `String`/`SP` children for text) into a `PageAnnotation` JSON,
   `words: []` for now — real MAKHZAN `String` positions are Kraken-interpolated, not
   hand-annotated word boxes, so they're deliberately not trusted as geometry. `manuscript_id`
   and `scribe` aren't reliably encoded in ALTO itself; pass them explicitly per-file once the
   real MAKHZAN manifest is in hand rather than relying on the filename-stem fallback. Then run
   `python -m ottoman_htr.data.validate <output_dir>` — it only checks JSON parses, not that the
   referenced images actually exist yet, so still eyeball a few `image_path` guesses by hand.
3. **Select 150 of 241 pages, split by manuscript, not randomly** — same reasoning as the
   train/val/test split in `docs/architecture.md`: an entire manuscript (one scribe, one
   hand) goes to one split, or the model just memorizes the scribe.
   `python -m ottoman_htr.data.select_corpus` (`src/ottoman_htr/data/select_corpus.py`) does
   this: caps any single manuscript's contribution (real data has one manuscript, Doc 3263, with
   116 of the 241 pages — capped to 30 by default, then trimmed further to hit the page target),
   then assigns whole manuscripts to train/val/test via largest-first greedy balancing, making
   sure val and test each get a non-Naskh manuscript when one is available. Verified against the
   real downloaded corpus: 150 pages selected across all 26 manuscripts (cap only binds on Doc
   3263), split **105 train / 23 val / 22 test** across 11/9/6 manuscripts, with 1/4/2 non-Naskh
   manuscripts respectively — train, val, test, and the model's forward/backward pass on the
   materialized train split all checked out end to end.
4. **Prefer "main text body only" pages first** for Tier 1, to avoid marginalia/gloss noise in
   the initial ablation; comprehensively-segmented pages become useful later if margin text
   turns out to matter.
5. **Pick the Tier-2 subset** (~20–30 pages) spanning multiple manuscripts and, if available,
   more than one script type — the point is generalization, not just raw count. Done: **24
   pages** in `data/raw/makhzan/tier2_subset` — all 12 non-Naskh pages available in the 150-page
   split (riqa ×1, taliq ×5 across 2 manuscripts, naskh_shikaste_mixed ×1, divani ×4 across 2
   manuscripts) plus 12 Naskh pages spread across 10 different manuscripts (1–2 pages each, for
   scribal variety rather than depth on any one hand). 446 lines total.
6. **Annotate Tier 2 word/char/dot boxes.** eScriptorium doesn't do word/char segmentation, so
   this needs a separate lightweight tool: [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/)
   (single offline HTML file, no install, exports JSON). `src/ottoman_htr/data/via_export.py`
   generates a VIA2 project pre-loaded with each page's existing line boxes as a reference layer
   (labelled `level=line`, not meant to be edited), and `via_import.py` merges the exported
   word/char/dot regions back into `PageAnnotation` JSON — see "Tier-2 annotation workflow"
   below for the concrete steps.
7. **Synthetic degradation pass** (per `docs/architecture.md`'s robustness test) is applied to
   clean Tier-1 images programmatically — no need to source naturally damaged pages separately.

## Resolved

- **Per-manuscript page counts and script distribution**: confirmed from the real metadata
  (`data/raw/makhzan/metadata.tsv`, gitignored): 241 pages / 26 manuscripts, one of which (Doc
  3263, Uppsala University Library, naskh, 1600) alone holds 116 pages; the other 25 range from
  1–17 pages each, across 9 repositories (Berlin Staatsbibliothek, Koç University, Leipzig,
  Dresden SLUB, Tübingen, Leiden, BnF, Princeton, Forschungsbibliothek Gotha). `select_corpus.py`
  handles this imbalance automatically (see above) rather than needing a manually-tuned split.
- **ALTO format assumptions**: the real MAKHZAN ALTO export uses a flat, comma-free
  `"x1 y1 x2 y2 ..."` point list for `BASELINE`/`POINTS` (not the ALTO spec's `"x1,y1 x2,y2"`
  form `import_alto.py` originally assumed) — fixed and covered by a regression test. Confirmed
  clean parse across all 241 real pages (0 errors, 3760 lines, no lines silently left at a
  bogus 0.0 baseline angle). Also confirmed: exactly one `<String>` per `<TextLine>` (i.e. no
  fake word segmentation to accidentally trust), and every page has a matching image (52 `.jpg`
  + 189 `.png`, mixed extensions — `import_alto.py`'s extension-probing fallback handles this).

## Tier-2 annotation workflow

**Revised 2026-09-18, after a real pilot.** The original plan below (word →
char → dot, all hand-drawn) turned out not to match this corpus. Piloting it
on the smallest page (`89_12037`, 7 lines) found, with real evidence (pixel
density analysis, not just impression):

- **Word boundaries are not reliably determinable by eye.** Binarizing a
  line and looking for full-height ink gaps found only one genuine gap in a
  570px, 5-word line — the rest connect via a running baseline ligature.
  This is standard for this style of Ottoman cursive, not a defect of this
  one scribe.
- **Character boundaries aren't either, for the same reason.** Medial
  Arabic-script letterforms are designed to connect via an unbroken stroke.
  This is also *why the model uses CTC loss* — CTC exists precisely to avoid
  needing pre-segmented characters, which the geometric-embedding feature
  vector had been implicitly assuming were annotatable.
- **Dots are the one thing that stayed tractable** — visually discrete,
  small, blob-shaped, distinguishable from connecting strokes at ~8–10×
  zoom. Confirmed against real examples (e.g. the two dots of ت in
  "دستانى", found independently by both eye and a pixel connected-component
  check).

**Schema change to match:** `LineAnnotation` gained `dot_points` (dots
attached directly to the line, no word/char attribution — see
`src/ottoman_htr/data/schema.py`). `line_geometry_vector` was redesigned
around this (`GEOMETRY_DIM` 25 → 21, `LINE_GEOMETRY_DIM` 13 → 9): word/char
box-derived dimensions are gone; `n_chars` now comes from the line's own
transcription text (already correct, Tier-1, no annotation needed) instead
of a promised-but-unreliable character count; `n_dots` and its distribution
across line-thirds are the real new signal. `via_import.py` falls back
word → char-on-line → dot-on-line depending on what an annotator actually
drew, so nothing here forces a rewrite if a future page *does* get clean
word/char boxes.

**Semi-automated candidate generation, not blind hand-drawing.** Hand-
placing ~6,000+ dots across 446 lines was estimated (from real per-line
pacing) at 40–80+ hours either way (human or AI-assisted). Built instead:
a connected-component blob detector over each line (`scipy.ndimage.label`
on an adaptively-binarized crop), filtered by shape/position **relative to
line height** (dot-like candidates are small, roughly square, and not in
the top/bottom ~6% of the line — that edge zone is dominated by ascender
tips and neighbouring-line bleed). Relative, not absolute-pixel, thresholds
matter: this corpus's 24 pages come from 9 different repositories with
visibly different scan resolutions, and an absolute-pixel filter calibrated
on one page silently starved higher-resolution pages of real detections.

Result: 28,108 raw connected components across 24 pages → 5,909 filtered
candidates (`data/raw/makhzan/tier2_dot_candidates_via_project.json`, a
real VIA2 project with every candidate as a `level=dot` point region
alongside the existing `level=line` reference boxes).

**Honest accuracy, from real sampling (not assumed):** spot-checked against
several lines across 3 different pages. Precision is inconsistent,
page-dependent — roughly 25–80% depending on scribal density and scan
quality — and recall has real gaps, especially for lines using
Ottoman-specific letters that carry more/less common dot patterns (e.g. چ,
ق). **This is a draft, not ground truth.** It cuts the task from
"draw ~6,000 points from nothing" to "review ~6,000 pre-placed points and
delete the wrong ones" — materially faster, but a human confirmation pass
is still required before trusting it.

1. **Open** `data/raw/makhzan/tier2_dot_candidates_via_project.json` in VIA
   ([robots.ox.ac.uk/~vgg/software/via](https://www.robots.ox.ac.uk/~vgg/software/via/)),
   pointing it at `data/raw/makhzan/tier2_subset/images/` when asked.
2. **Review, don't draw from scratch.** Each line's yellow reference box is
   still there; red dot points are the candidates. Delete false positives
   (common near line edges), add any obviously missed dots. Zoom in — at
   normal view many of these are too small to judge confidently.
3. **Export → merge**, same as before:
   ```bash
   python -m ottoman_htr.data.via_import path/to/export.json data/raw/makhzan/tier2_subset data/raw/makhzan/tier2_annotated
   python -m ottoman_htr.data.validate data/raw/makhzan/tier2_annotated
   ```
   `validate` now also reports `lines_with_direct_dots`/`dots_direct` for
   this annotation shape.

The candidate-generation pipeline itself lives in
`scripts/tier2_dot_review.py` (not part of the installed package — a
working tool for this one task, kept for whoever does the confirmation
pass, or for regenerating candidates if the filter is retuned).

## Open questions

- Whether any MAKHZAN Ottoman Turkish manuscripts overlap with material already used elsewhere
  in the Bern DH Center's or `ottoman-rag`'s pipelines — check before treating this as "new" data.
- Who does the Tier-2 manual annotation (DH Center student time vs. own time) — sizes the
  realistic timeline; ~20–30 pages of char+dot-level annotation is real, non-trivial labor even
  with a lightweight tool.

## Not pursuing (for now)

- **Fresh digitization from Süleymaniye Library**: fully digitized but bulk image access needs
  an individual permission request, not an open bulk-download — too slow to be the primary
  source for a 150-page prototype corpus, revisit only if MAKHZAN's diversity proves
  insufficient.
- **Leiden University Digital Collections** (CC-BY, openly reusable, ~6,500 Middle East
  manuscripts including Ottoman Turkish): a plausible supplemental image source if more scribe
  diversity is needed later, but images alone still require in-house transcription — higher
  marginal cost per page than MAKHZAN, so treat as a stretch goal, not part of the core 150.
- **Austrian National Library (ÖNB)**: broad IIIF support but per-item rights not verified —
  don't depend on it without checking the specific manuscript's rights statement first.
- **Transcriptiones (Basel)**: a good citable Swiss-DH precedent for framing/funding narratives
  (2024 Swiss Reproducibility Award, 2023 Open Research Data Prize bronze), but it's a
  general-purpose crowdsourced transcription+metadata platform with no published geometry-level
  annotation methodology — not a workflow to copy for this project's line/word/char/dot format.
