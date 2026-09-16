from __future__ import annotations

import torch

from .data.vocab import CharVocab
from .text_utils import levenshtein


def greedy_ctc_decode(log_probs: torch.Tensor, vocab: CharVocab) -> list[str]:
    """CTC greedy decoding: argmax per timestep, collapse consecutive repeats, drop blanks."""
    predictions = log_probs.argmax(dim=-1)  # (B, T)
    texts = []
    for seq in predictions.tolist():
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev:
                collapsed.append(idx)
            prev = idx
        texts.append(vocab.decode(collapsed))
    return texts


def character_error_rate(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, target) / len(target)


def word_error_rate(pred: str, target: str) -> float:
    target_words = target.split()
    if not target_words:
        return 0.0 if pred.split() else 1.0
    return levenshtein(pred.split(), target_words) / len(target_words)


def corpus_cer(preds: list[str], targets: list[str]) -> float:
    """Micro-averaged CER: total edit distance over total target characters, across a corpus
    (not the mean of per-line CERs, which would over-weight short lines)."""
    total_edits = sum(levenshtein(p, t) for p, t in zip(preds, targets))
    total_chars = sum(len(t) for t in targets)
    return total_edits / total_chars if total_chars else 0.0


def corpus_wer(preds: list[str], targets: list[str]) -> float:
    total_edits = sum(levenshtein(p.split(), t.split()) for p, t in zip(preds, targets))
    total_words = sum(len(t.split()) for t in targets)
    return total_edits / total_words if total_words else 0.0
