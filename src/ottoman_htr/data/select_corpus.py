from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .schema import PageAnnotation


@dataclass
class ManuscriptPages:
    manuscript_id: str
    pages: list[str] = field(default_factory=list)  # PageAnnotation JSON stems
    scripts: set[str] = field(default_factory=set)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def has_non_naskh(self) -> bool:
        return any(s not in ("naskh", "unknown") for s in self.scripts)


def load_page_scripts(metadata_tsv: str | Path) -> dict[str, str]:
    """Doc Part ID -> script, from the MAKHZAN metadata TSV. Page stems are "{Doc ID}_{Doc Part
    ID}" (confirmed against the real corpus), so this is keyed on Doc Part ID alone."""
    scripts: dict[str, str] = {}
    with open(metadata_tsv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            part_id = row.get("Doc Part ID", "").strip()
            if part_id:
                scripts[part_id] = row.get("Script", "").strip().lower() or "unknown"
    return scripts


def group_by_manuscript(annotation_dir: str | Path, page_scripts: dict[str, str]) -> dict[str, ManuscriptPages]:
    manuscripts: dict[str, ManuscriptPages] = {}
    for json_path in sorted(Path(annotation_dir).glob("*.json")):
        stem = json_path.stem
        page = PageAnnotation.from_json(json_path)
        manuscript = manuscripts.setdefault(page.manuscript_id, ManuscriptPages(page.manuscript_id))
        manuscript.pages.append(stem)
        _, _, part_id = stem.partition("_")
        manuscript.scripts.add(page_scripts.get(part_id, "unknown"))
    return manuscripts


def cap_and_select(
    manuscripts: dict[str, ManuscriptPages],
    target_pages: int,
    max_per_manuscript: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    """manuscript_id -> selected page stems, capping any single manuscript's contribution so it
    can't dominate the corpus (one manuscript in the real MAKHZAN Ottoman Turkish subset has 116
    of 241 pages -- see docs/corpus_collection_plan.md), then trimming the largest selections
    down further if the cap alone still leaves more than target_pages."""
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for ms_id, ms in manuscripts.items():
        pages = list(ms.pages)
        rng.shuffle(pages)
        selected[ms_id] = pages[:max_per_manuscript]

    total = sum(len(pages) for pages in selected.values())
    while total > target_pages:
        biggest = max(selected, key=lambda k: len(selected[k]))
        selected[biggest].pop()
        total -= 1
    return selected


def split_by_manuscript(
    selected: dict[str, list[str]],
    manuscripts: dict[str, ManuscriptPages],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, list[str]]:
    """Assigns whole manuscripts to train/val/test -- never splits one manuscript's pages across
    sets, so a scribe's hand never leaks between them -- via largest-first greedy balancing
    against each split's target share, after first making sure val and test each get one
    non-Naskh manuscript (if the selection has any at all)."""
    grand_total = sum(len(pages) for pages in selected.values())
    order = sorted(selected, key=lambda k: len(selected[k]), reverse=True)
    totals = {"train": 0, "val": 0, "test": 0}
    target_frac = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    assignment: dict[str, str] = {}

    non_naskh_ids = [ms_id for ms_id in order if manuscripts[ms_id].has_non_naskh]
    for split in ("val", "test"):
        for ms_id in non_naskh_ids:
            if ms_id not in assignment:
                assignment[ms_id] = split
                totals[split] += len(selected[ms_id])
                break

    for ms_id in order:
        if ms_id in assignment:
            continue
        split = min(totals, key=lambda s: totals[s] - target_frac[s] * grand_total)
        assignment[ms_id] = split
        totals[split] += len(selected[ms_id])

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for ms_id, split in assignment.items():
        out[split].extend(selected[ms_id])
    return out


def materialize_split(annotation_dir: str | Path, stems: list[str], output_dir: str | Path, copy: bool = False) -> None:
    """Writes the given pages' JSON + images into output_dir as a self-contained annotation_dir
    (usable directly by OttomanLineDataset), via symlinks by default to avoid tripling disk use."""
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        src_json = annotation_dir / f"{stem}.json"
        page = PageAnnotation.from_json(src_json)
        _link(src_json, output_dir / f"{stem}.json", copy)
        src_image = annotation_dir / page.image_path
        _link(src_image, images_dir / Path(page.image_path).name, copy)


def _link(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select ~150 pages from a larger converted annotation_dir (e.g. all 241 real MAKHZAN "
            "Ottoman Turkish pages), capping any single manuscript's share, then split the "
            "selection into train/val/test by whole manuscript (see docs/corpus_collection_plan.md)."
        )
    )
    parser.add_argument("annotation_dir", type=Path, help="Directory of *.json PageAnnotation + images/")
    parser.add_argument("output_dir", type=Path, help="Where to write train/, val/, test/ subdirectories")
    parser.add_argument("--metadata", type=Path, default=None, help="MAKHZAN metadata TSV, for script-aware splitting")
    parser.add_argument("--target-pages", type=int, default=150)
    parser.add_argument("--max-per-manuscript", type=int, default=30)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = parser.parse_args()

    page_scripts = load_page_scripts(args.metadata) if args.metadata else {}
    manuscripts = group_by_manuscript(args.annotation_dir, page_scripts)
    selected = cap_and_select(manuscripts, args.target_pages, args.max_per_manuscript, args.seed)
    splits = split_by_manuscript(selected, manuscripts, args.val_frac, args.test_frac)

    for split_name, stems in splits.items():
        materialize_split(args.annotation_dir, stems, args.output_dir / split_name, args.copy)

    total_selected = sum(len(pages) for pages in selected.values())
    print(f"{len(manuscripts)} manuscripts available, {sum(m.n_pages for m in manuscripts.values())} pages total")
    print(f"selected {total_selected} pages across {len(selected)} manuscripts (cap={args.max_per_manuscript}/manuscript)")
    for split_name in ("train", "val", "test"):
        split_manuscripts = [ms for ms in selected if any(s in splits[split_name] for s in selected[ms])]
        non_naskh = sum(1 for ms in split_manuscripts if manuscripts[ms].has_non_naskh)
        print(f"  {split_name}: {len(splits[split_name])} pages, {len(split_manuscripts)} manuscripts, {non_naskh} non-Naskh")
    print("\nCC BY-NC-SA: cite the OpenITI MAKHZAN paper (JOHD vol. 12, 2026) and the Zenodo DOI "
          "wherever this selection or a model trained on it is described -- see "
          "docs/corpus_collection_plan.md.")


if __name__ == "__main__":
    main()
