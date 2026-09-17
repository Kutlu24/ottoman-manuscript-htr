# Tier-1 training notes

First real training runs against the MAKHZAN-derived Tier-1 split (`data/raw/makhzan/split_150`:
105 train pages / 1678 lines, 23 val pages / 436 lines — see `docs/corpus_collection_plan.md`).
Run on Google Colab (free T4 GPU), `batch_size=8`.

## Fix: dataset was redoing image decode + geometry extraction every epoch

`OttomanLineDataset.__getitem__` originally re-opened and re-decoded the *page* image and
recomputed `combined_geometry_vector` from scratch on every access — and since a page has ~16
lines on average, that meant re-decoding the same page image up to 16× per epoch, every epoch.
This, not the training device, was the actual bottleneck:

| Config | Epoch time (train loop) |
|---|---|
| CPU, no fix | 213s |
| GPU (T4), no fix | 180s (barely faster — the bottleneck was CPU-bound regardless of device) |
| GPU + `num_workers=2` (parallel loading, still no caching) | ~149s |
| CPU, with fix (decode each page once, precompute all samples in `__init__`) | 93s |
| **GPU + fix** | **~5–9s** |

Fix: decode each page image once per `OttomanLineDataset` construction (cached by path) and
precompute every sample's crop + geometry vector up front, so `__getitem__` is a plain list
lookup. See `src/ottoman_htr/data/dataset.py`. This is what actually unlocked fast iteration —
not the GPU move itself, which only paid off once the redundant CPU work was gone.

## Learning-rate sweep (150 epochs each, otherwise identical config)

| LR | Best val CER | At epoch |
|---|---|---|
| 1e-3 (default) | 0.8818 | ~149 |
| 3e-4 | 0.849 | 150 |
| **3e-3** | **0.793** | 132 |

`3e-3` was the clear winner — reached a lower CER, and reached it faster (val CER only starts
moving off 1.0 — i.e. the model escaping the CTC "predict nothing" collapse — around epoch
40–55 for all three LRs; `3e-3` also had the steepest subsequent descent).

## Extended run: does more than 150 epochs help?

Ran `lr=3e-3` for up to 500 epochs. Reached **epoch 415/500** before the Colab session lost its
GPU allocation mid-run (free-tier usage limit) and the runtime — and with it the checkpoint and
the rest of the log — was unrecoverable; nothing past what was visible in the notebook at
disconnect survived.

What was visible (epochs 401–415): val CER oscillating in the **0.85–0.88** range — no better
than, and generally worse than, the 150-epoch run's best of **0.793 at epoch 132**. Read
together, this is a real (if informal) finding: on this small a training set (105 pages), val
CER at `lr=3e-3` plateaus and gets noisier past roughly epoch 130–150, rather than continuing to
improve. Throwing more epochs at it is not the lever that matters next.

**Working reference config:** `lr=3e-3`, ~130–150 epochs, `batch_size=8`, `num_workers=2`. Best
observed val CER ≈ **0.79**. Val WER never meaningfully dropped below 1.0 in any run — word-level
output isn't usable yet at this scale; only a character-level learning signal is present so far.

## What this does and doesn't tell us

- It confirms the pipeline trains end-to-end on real data and the model learns something
  non-trivial (train loss drops steadily; val CER moves well off the CTC-collapse ceiling of 1.0).
- 0.79 CER is far from a usable transcription — expected, given only 105 real training pages and
  no pretraining. This is a baseline signal check, not a claim of a working reader.
- Given the plateau past ~150 epochs, the next real lever is more/better data (Tier-2 word/char
  annotation for the geometric-embedding path, or simply more Tier-1 pages) rather than longer
  training runs — see `docs/corpus_collection_plan.md` and `docs/architecture.md`.
- **No checkpoint from any of these runs was saved locally** — Colab sessions are ephemeral and
  none were downloaded before disconnecting. These numbers are the evidence trail; the trained
  weights themselves do not exist anymore. Future long runs should periodically download
  `best_model.pt`/`train_log.jsonl` (or mount Drive) rather than relying on the live session.

## Three-way ablation: binary vs. neural vs. geometric+fuzzy (150 epochs each)

Ran locally on CPU (background, not Colab), `lr=3e-3`, `batch_size=8`, same Tier-1 split (105
train / 23 val pages). All three arms share the identical `ImageEncoder` backbone
(`src/ottoman_htr/model.py:build_model`) so the architectural variable is isolated per
`docs/architecture.md`'s design. Unlike the Colab runs above, checkpoints this time were saved
for real: `runs/ablation_150ep/{binary,neural,fuzzy}/{best_model.pt,last_model.pt,train_log.jsonl}`
(gitignored, local only — not lost this time, but also not committed).

| Model | best val CER | best val WER | best epoch | final train loss |
|---|---|---|---|---|
| **binary** (A: scan → binarize → OCR) | 0.8294 | 1.0006 | 125 | 3.1219 |
| **neural** (B: raw grayscale → CNN → CTC) | **0.8265** | 1.0794 | 98 | 2.6798 |
| **fuzzy** (C: image + geometry → fuzzy fusion, this project's proposal) | 0.8403 | **0.9994** | 133 | **1.9266** |

Full log: `runs/ablation_150ep.log`. Machine-readable summary: `runs/ablation_150ep/ablation_summary.json`.

### Reading this honestly: the core hypothesis is neither confirmed nor refuted here

**On CER, fuzzy is the worst of the three** (0.8403 vs. 0.8265–0.8294) — a small but real gap in
the wrong direction for this project's own thesis. Two things push back against reading that as a
clean refutation:

- **Fuzzy is the only arm to get WER under 1.0** (0.9994) — meaning at least one full word came
  out exactly right somewhere in validation; binary and neural never did that in 150 epochs.
- **Fuzzy's final train loss (1.9266) is well below the other two** (2.68, 3.12) at the *same*
  epoch budget — it was still fitting harder when the run was cut off, while binary/neural's
  train losses had largely flattened. Consistent with "fuzzy needed more epochs to show an
  advantage," though not proof of it — could equally be overfitting starting to show up as lower
  train loss without a matching val-CER improvement.

**A more important caveat found while writing this up:** `combined_geometry_vector` (used by the
`fuzzy` arm) concatenates `line_geometry_vector` (13 dims, layout-derived) with
`image_geometry_vector` (12 dims, pixel-derived) — see `src/ottoman_htr/geometry/features.py`.
`line_geometry_vector` reads `line.bbox`/`baseline_angle_deg` (present in Tier-1 line-level ALTO —
3 of its 13 dims carry real signal here) **and** word widths/gaps, char widths, and dot counts
(`line.words[...]`) — but Tier-1 data has `words=[]` on every line (see
`docs/corpus_collection_plan.md`), so **roughly 10 of the fuzzy model's 25 geometry input
dimensions were constant zero for this entire run.** Those are exactly the letter/dot-level
features the original fuzzy-logic hypothesis (faded ink, missing dots, letter-shape confidence)
was about. This ablation tested a fuzzy model running on roughly half its intended input space,
not the real hypothesis — **fair comparison against the geometry Tier-1 actually offers, but not
yet a fair test of the full geometric-embedding idea.**

### What this does and doesn't tell us

- All three arms are still far from usable transcription (CER ≈ 0.83–0.84) at this data scale (105
  training pages, no pretraining) — consistent with the single-model plateau found above (≈0.79 at
  best, with different hyperparameters and without the other two arms competing for the same
  150-epoch budget; not directly comparable number-for-number).
- The three-way comparison itself works end to end and produced a real, if inconclusive, result —
  the pipeline, the shared backbone, and the ablation runner (`src/ottoman_htr/ablation.py`) are
  validated on real data now, twice.
- The honest verdict: **this ablation cannot yet distinguish "the fuzzy hypothesis is wrong" from
  "the fuzzy model hasn't been given the data it needs to prove itself."** Tier-2 annotation
  (word/char/dot boxes via VIA — already in progress, see `docs/corpus_collection_plan.md`) is a
  precondition for a real test of this project's central claim, not just a nice-to-have for later.
