from __future__ import annotations

import json
from pathlib import Path

BLANK = "<blank>"
UNK = "<unk>"


class CharVocab:
    """Character<->index mapping for CTC targets/decoding. Index 0 is always the CTC blank."""

    def __init__(self, characters: list[str]):
        self.itos = [BLANK, UNK, *dict.fromkeys(characters)]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> CharVocab:
        chars = sorted({c for text in texts for c in text})
        return cls(chars)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.stoi[UNK]) for c in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.itos[i] for i in indices if i != 0)

    def to_json(self, path: str | Path) -> None:
        """Saves the character inventory (itos[2:], i.e. without BLANK/UNK) so a checkpoint can
        be reloaded with the exact vocabulary it was trained against."""
        Path(path).write_text(json.dumps(self.itos[2:], ensure_ascii=False, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> CharVocab:
        return cls(json.loads(Path(path).read_text()))
