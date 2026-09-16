from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

_PAD = ""  # start-of-sequence padding, outside any real Ottoman/Turkish text


class CharNGramLanguageModel:
    """Character-level n-gram model for re-scoring CTC candidate transcriptions with textual
    context -- a cheap stand-in for the "Ottoman Language Model" stage until a corpus large
    enough to justify a transformer LM exists (see docs/architecture.md)."""

    def __init__(self, order: int = 3):
        self.order = order
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.totals: Counter = Counter()
        self.vocab: set[str] = set()

    def fit(self, texts: Iterable[str]) -> None:
        for text in texts:
            padded = (_PAD * (self.order - 1)) + text
            for i in range(len(text)):
                context = padded[i : i + self.order - 1]
                char = padded[i + self.order - 1]
                self.counts[context][char] += 1
                self.totals[context] += 1
                self.vocab.add(char)

    def score(self, text: str) -> float:
        """Average log-probability per character, add-one smoothed over the seen vocabulary."""
        if not self.vocab:
            raise RuntimeError("call fit() before score()")
        padded = (_PAD * (self.order - 1)) + text
        v = len(self.vocab)
        log_prob = 0.0
        for i in range(len(text)):
            context = padded[i : i + self.order - 1]
            char = padded[i + self.order - 1]
            count = self.counts[context][char]
            total = self.totals[context]
            log_prob += math.log((count + 1) / (total + v))
        return log_prob / max(len(text), 1)

    def rescore_candidates(self, candidates: list[str]) -> list[tuple[str, float]]:
        """Candidates sorted best-first by language-model score."""
        scored = [(c, self.score(c)) for c in candidates]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)
