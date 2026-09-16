from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data.dataset import OttomanLineDataset, collate_lines
from .data.vocab import CharVocab
from .losses import recognition_loss
from .metrics import corpus_cer, corpus_wer, greedy_ctc_decode
from .model import build_model


def build_vocab(annotation_dir: Path) -> CharVocab:
    """Built from the raw annotation JSON text fields directly, since the dataset itself needs a
    vocab to construct its CTC targets. Always built from the training split -- val/test
    characters unseen in training legitimately fall back to <unk>, matching real deployment."""
    texts = []
    for ann_path in sorted(annotation_dir.glob("*.json")):
        raw = json.loads(ann_path.read_text())
        texts.extend(line["text"] for line in raw.get("lines", []))
    return CharVocab.from_texts(texts)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, vocab: CharVocab, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for batch in loader:
        images = batch["images"].to(device)
        geometry = batch["geometry"].to(device)
        batch_targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        log_probs = model(images, geometry)
        input_lengths = model.vision_encoder.output_length(batch["input_lengths"]).to(device)
        loss = recognition_loss(log_probs, batch_targets, input_lengths, target_lengths)
        total_loss += loss.item()

        preds.extend(greedy_ctc_decode(log_probs, vocab))
        offset = 0
        for length in batch["target_lengths"].tolist():
            targets.append(vocab.decode(batch_targets[offset : offset + length].tolist()))
            offset += length

    return {
        "loss": total_loss / max(len(loader), 1),
        "cer": corpus_cer(preds, targets),
        "wer": corpus_wer(preds, targets),
    }


def train(
    train_dir: Path,
    val_dir: Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: Path,
    num_workers: int = 0,
    model_kind: str = "fuzzy",
) -> None:
    # Each sample decodes a JPEG/PNG and runs scipy/numpy geometry feature extraction on CPU --
    # that dominates step time regardless of device, so parallel workers matter more than the GPU.
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    vocab = build_vocab(train_dir)
    train_dataset = OttomanLineDataset(train_dir, vocab)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_lines, **loader_kwargs
    )

    val_loader = None
    if val_dir is not None:
        val_dataset = OttomanLineDataset(val_dir, vocab)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_lines, **loader_kwargs
        )

    config = ModelConfig()
    model = build_model(model_kind, config, vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab.to_json(output_dir / "vocab.json")
    log_path = output_dir / "train_log.jsonl"
    best_cer = float("inf")

    print(f"train: {len(train_dataset)} lines, vocab size {len(vocab)}, device {device}, model {model_kind}")
    if val_loader is not None:
        print(f"val: {len(val_dataset)} lines")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            images = batch["images"].to(device)
            geometry = batch["geometry"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            log_probs = model(images, geometry)
            input_lengths = model.vision_encoder.output_length(batch["input_lengths"]).to(device)

            loss = recognition_loss(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        record = {
            "epoch": epoch + 1,
            "train_loss": round(total_loss / max(len(train_loader), 1), 4),
            "elapsed_s": round(time.time() - t0, 1),
        }

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, vocab, device)
            record["val_loss"] = round(val_metrics["loss"], 4)
            record["val_cer"] = round(val_metrics["cer"], 4)
            record["val_wer"] = round(val_metrics["wer"], 4)
            if val_metrics["cer"] < best_cer:
                best_cer = val_metrics["cer"]
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                record["checkpoint"] = "best_model.pt"

        print(json.dumps(record, ensure_ascii=False))
        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    torch.save(model.state_dict(), output_dir / "last_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Ottoman manuscript HTR model.")
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2, help="Parallel DataLoader workers (0 disables)")
    parser.add_argument(
        "--model",
        dest="model_kind",
        choices=["binary", "neural", "fuzzy"],
        default="fuzzy",
        help="Which ablation arm to train: 'binary' (A), 'neural' (B), 'fuzzy' (C, default) -- see docs/architecture.md",
    )
    args = parser.parse_args()
    train(
        args.train_dir,
        args.val_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.device,
        args.output_dir,
        args.num_workers,
        args.model_kind,
    )


if __name__ == "__main__":
    main()
