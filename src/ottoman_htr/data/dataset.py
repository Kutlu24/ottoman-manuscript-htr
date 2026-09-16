from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..geometry.features import combined_geometry_vector
from .schema import LineAnnotation, PageAnnotation
from .vocab import CharVocab


@dataclass
class LineSample:
    image: torch.Tensor  # (1, H, W) grayscale crop, normalized to [0, 1]
    geometry: torch.Tensor  # (GEOMETRY_DIM,) hand-crafted summary features
    target: torch.Tensor  # (L,) character indices for CTC
    text: str


class OttomanLineDataset(Dataset):
    """One sample per manuscript line: image crop + geometric summary + CTC target.

    Expects a directory of PageAnnotation JSON files (see data.schema) whose `image_path` is
    relative to that same directory."""

    def __init__(self, annotation_dir: str | Path, vocab: CharVocab, line_height: int = 64):
        self.vocab = vocab
        self.line_height = line_height
        self._lines: list[tuple[Path, LineAnnotation]] = []
        for ann_path in sorted(Path(annotation_dir).glob("*.json")):
            page = PageAnnotation.from_json(ann_path)
            image_path = (Path(annotation_dir) / page.image_path).resolve()
            for line in page.lines:
                self._lines.append((image_path, line))

        # Decode each page image once and reuse it across its lines, then precompute every
        # sample's crop + geometry features up front. Otherwise __getitem__ redoes both the
        # PIL decode and the scipy/numpy geometry extraction on every access -- with
        # persistent_workers=True that means every epoch, not just once, and PIL/scipy are
        # CPU-bound regardless of training device (see docs/architecture.md).
        page_cache: dict[Path, Image.Image] = {}
        self._samples: list[LineSample] = []
        for image_path, line in self._lines:
            page_img = page_cache.get(image_path)
            if page_img is None:
                with Image.open(image_path) as opened:
                    page_img = opened.convert("L")
                page_cache[image_path] = page_img
            self._samples.append(self._build_sample(page_img, line))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> LineSample:
        return self._samples[idx]

    def _build_sample(self, page_img: Image.Image, line: LineAnnotation) -> LineSample:
        arr = self._crop_line(page_img, line)
        geometry = torch.from_numpy(combined_geometry_vector(line, arr)).float()
        image = torch.from_numpy(arr).unsqueeze(0)
        target = torch.tensor(self.vocab.encode(line.text), dtype=torch.long)
        return LineSample(image=image, geometry=geometry, target=target, text=line.text)

    def _crop_line(self, page_img: Image.Image, line: LineAnnotation) -> np.ndarray:
        box = (line.bbox.x0, line.bbox.y0, line.bbox.x1, line.bbox.y1)
        crop = page_img.crop(box)
        scale = self.line_height / max(crop.height, 1)
        crop = crop.resize((max(int(crop.width * scale), 1), self.line_height))
        return np.asarray(crop, dtype=np.float32) / 255.0


def collate_lines(batch: list[LineSample]) -> dict[str, torch.Tensor]:
    """Pads variable-width line images and variable-length targets for a CTC batch."""
    widths = [s.image.shape[-1] for s in batch]
    max_w = max(widths)
    height = batch[0].image.shape[-2]
    images = torch.zeros(len(batch), 1, height, max_w)
    for i, s in enumerate(batch):
        images[i, :, :, : s.image.shape[-1]] = s.image
    return {
        "images": images,
        "geometry": torch.stack([s.geometry for s in batch]),
        "targets": torch.cat([s.target for s in batch]),
        "target_lengths": torch.tensor([len(s.target) for s in batch], dtype=torch.long),
        "input_lengths": torch.tensor(widths, dtype=torch.long),
    }
