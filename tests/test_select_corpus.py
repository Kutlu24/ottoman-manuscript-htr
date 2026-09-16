import csv

import pytest

from ottoman_htr.data.schema import BBox, LineAnnotation, PageAnnotation
from ottoman_htr.data.select_corpus import (
    cap_and_select,
    group_by_manuscript,
    materialize_split,
    split_by_manuscript,
)
from ottoman_htr.data.validate import validate_directory

# A synthetic corpus mirroring the real shape: one big manuscript (A, naskh) that dominates,
# several small ones, and one non-Naskh manuscript to exercise the split constraint.
MANUSCRIPT_SPEC = {
    "A": ("naskh", 20),
    "B": ("naskh", 5),
    "C": ("naskh", 4),
    "D": ("taliq", 2),
    "E": ("naskh", 1),
}


def _build_corpus(tmp_path):
    ann_dir = tmp_path / "annotations"
    images_dir = ann_dir / "images"
    images_dir.mkdir(parents=True)
    metadata_path = tmp_path / "metadata.tsv"

    rows = []
    for ms_id, (script, n_pages) in MANUSCRIPT_SPEC.items():
        for i in range(n_pages):
            part_id = f"{ms_id}p{i}"
            stem = f"{ms_id}_{part_id}"
            image_name = f"{stem}.jpg"
            (images_dir / image_name).touch()
            page = PageAnnotation(
                image_path=f"images/{image_name}",
                manuscript_id=ms_id,
                folio=part_id,
                transcription="متن",
                lines=[LineAnnotation(bbox=BBox(0, 0, 100, 20), text="متن")],
            )
            page.to_json(ann_dir / f"{stem}.json")
            rows.append({"Doc Part ID": part_id, "Script": script})

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Doc Part ID", "Script"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return ann_dir, metadata_path


def test_group_by_manuscript_counts_and_scripts(tmp_path):
    ann_dir, metadata_path = _build_corpus(tmp_path)
    from ottoman_htr.data.select_corpus import load_page_scripts

    scripts = load_page_scripts(metadata_path)
    manuscripts = group_by_manuscript(ann_dir, scripts)

    assert {ms_id: m.n_pages for ms_id, m in manuscripts.items()} == {"A": 20, "B": 5, "C": 4, "D": 2, "E": 1}
    assert manuscripts["D"].has_non_naskh
    assert not manuscripts["A"].has_non_naskh


def test_cap_and_select_limits_dominant_manuscript_and_hits_target(tmp_path):
    ann_dir, metadata_path = _build_corpus(tmp_path)
    from ottoman_htr.data.select_corpus import load_page_scripts

    manuscripts = group_by_manuscript(ann_dir, load_page_scripts(metadata_path))
    selected = cap_and_select(manuscripts, target_pages=15, max_per_manuscript=6, seed=0)

    assert len(selected["A"]) <= 6  # capped, even though A has 20 pages available
    assert sum(len(pages) for pages in selected.values()) == 15


def test_split_by_manuscript_never_splits_one_manuscript(tmp_path):
    ann_dir, metadata_path = _build_corpus(tmp_path)
    from ottoman_htr.data.select_corpus import load_page_scripts

    manuscripts = group_by_manuscript(ann_dir, load_page_scripts(metadata_path))
    selected = cap_and_select(manuscripts, target_pages=32, max_per_manuscript=20, seed=0)
    splits = split_by_manuscript(selected, manuscripts, val_frac=0.2, test_frac=0.2)

    for ms_id, pages in selected.items():
        containing_splits = {name for name, stems in splits.items() if set(pages) & set(stems)}
        assert len(containing_splits) == 1, f"manuscript {ms_id} leaked across {containing_splits}"


def test_split_by_manuscript_gives_val_and_test_a_non_naskh_manuscript(tmp_path):
    ann_dir, metadata_path = _build_corpus(tmp_path)
    from ottoman_htr.data.select_corpus import load_page_scripts

    manuscripts = group_by_manuscript(ann_dir, load_page_scripts(metadata_path))
    selected = cap_and_select(manuscripts, target_pages=32, max_per_manuscript=20, seed=0)
    splits = split_by_manuscript(selected, manuscripts, val_frac=0.2, test_frac=0.2)

    # only one non-Naskh manuscript (D) exists, so it can satisfy at most one of val/test --
    # the constraint is "give one if available", not "guarantee both when supply is this thin"
    non_naskh_splits = [
        name for name in ("val", "test")
        if any(manuscripts[ms].has_non_naskh and set(selected[ms]) & set(splits[name]) for ms in selected)
    ]
    assert len(non_naskh_splits) >= 1


def test_materialize_split_produces_valid_annotation_dir(tmp_path):
    ann_dir, metadata_path = _build_corpus(tmp_path)
    from ottoman_htr.data.select_corpus import load_page_scripts

    manuscripts = group_by_manuscript(ann_dir, load_page_scripts(metadata_path))
    selected = cap_and_select(manuscripts, target_pages=32, max_per_manuscript=20, seed=0)
    splits = split_by_manuscript(selected, manuscripts, val_frac=0.2, test_frac=0.2)

    out_dir = tmp_path / "train_split"
    materialize_split(ann_dir, splits["train"], out_dir)

    report = validate_directory(out_dir)
    assert report.errors == []
    assert report.n_pages == len(splits["train"])
    for image in (out_dir / "images").iterdir():
        assert image.is_symlink()
