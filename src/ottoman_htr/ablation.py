from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .train import train

MODEL_KINDS = ["binary", "neural", "fuzzy"]
MODEL_LABELS = {
    "binary": "A -- scan -> binarization -> OCR",
    "neural": "B -- raw image -> CNN -> CTC (continuous, no geometry/fuzzy)",
    "fuzzy": "C -- image + geometry -> fuzzy fusion -> CTC (this project's proposal)",
}


def run_ablation(
    train_dir: Path,
    val_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    num_workers: int,
    output_dir: Path,
) -> dict:
    """Trains all three ablation arms (docs/architecture.md) on identical data/hyperparameters,
    so any outcome difference is attributable to architecture, not tuning generosity."""
    output_dir = Path(output_dir)
    summary: dict[str, dict] = {}

    for kind in MODEL_KINDS:
        run_dir = output_dir / kind
        print(f"\n=== model '{kind}' ({MODEL_LABELS[kind]}) ===")
        train(train_dir, val_dir, epochs, batch_size, lr, device, run_dir, num_workers, model_kind=kind)

        records = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text().splitlines()]
        val_records = [r for r in records if "val_cer" in r]
        best = min(val_records, key=lambda r: r["val_cer"]) if val_records else None
        last = records[-1] if records else None
        summary[kind] = {
            "label": MODEL_LABELS[kind],
            "best_val_cer": best["val_cer"] if best else None,
            "best_val_wer": best["val_wer"] if best else None,
            "best_epoch": best["epoch"] if best else None,
            "final_train_loss": last["train_loss"] if last else None,
        }

    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== ablation summary ===")
    print(f"{'model':<8} {'best_val_cer':>12} {'best_val_wer':>12} {'best_epoch':>10}")
    for kind in MODEL_KINDS:
        s = summary[kind]
        print(f"{kind:<8} {s['best_val_cer']:>12} {s['best_val_wer']:>12} {s['best_epoch']:>10}")
    print(f"\nwritten to {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the three-way ablation from docs/architecture.md: binary OCR (A) vs. neural "
            "OCR (B) vs. this project's geometric+fuzzy model (C), on identical data and "
            "hyperparameters."
        )
    )
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("val_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3, help="see docs/training_notes.md for why 3e-3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    run_ablation(
        args.train_dir,
        args.val_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.device,
        args.num_workers,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
