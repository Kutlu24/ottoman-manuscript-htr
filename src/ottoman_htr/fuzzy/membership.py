from __future__ import annotations

import torch

from ..text_utils import levenshtein


def evidence_to_membership(logits: torch.Tensor) -> torch.Tensor:
    """Raw per-class evidence (from vision or geometry) -> a membership distribution over
    character classes, so different sources are combined in the same [0, 1] space."""
    return torch.softmax(logits, dim=-1)


def dictionary_membership(candidate: str, lexicon: set[str], max_edits: int = 2) -> float:
    """Fuzzy lexicon-agreement score in [0, 1]: 1.0 for an exact match, decaying with edit
    distance, 0.0 once farther than `max_edits` from every lexicon entry."""
    if not lexicon:
        return 0.0
    if candidate in lexicon:
        return 1.0
    best = min(levenshtein(candidate, word) for word in lexicon)
    if best > max_edits:
        return 0.0
    return 1.0 - best / (max_edits + 1)
