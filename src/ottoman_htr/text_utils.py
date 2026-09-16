from __future__ import annotations

from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (strings or token lists). Shared by the fuzzy
    dictionary-membership score and the CER/WER metrics, which both need it."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
