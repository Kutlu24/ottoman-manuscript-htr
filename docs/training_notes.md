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
