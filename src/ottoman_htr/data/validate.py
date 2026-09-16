from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .schema import PageAnnotation


@dataclass
class ValidationReport:
    n_pages: int = 0
    n_lines: int = 0
    n_words: int = 0
    n_chars: int = 0
    n_chars_with_label: int = 0
    n_chars_with_dots: int = 0
    errors: list[str] | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def validate_directory(annotation_dir: str | Path) -> ValidationReport:
    """Parses every *.json in annotation_dir as a PageAnnotation and tallies its contents.
    Malformed files are recorded in `errors` rather than raising, so one bad page doesn't stop
    a batch check of the other 149."""
    report = ValidationReport()
    for path in sorted(Path(annotation_dir).glob("*.json")):
        try:
            page = PageAnnotation.from_json(path)
        except Exception as exc:  # noqa: BLE001 - collecting every malformed file, not just the first
            report.errors.append(f"{path.name}: {exc}")
            continue
        report.n_pages += 1
        for line in page.lines:
            report.n_lines += 1
            for word in line.words:
                report.n_words += 1
                for char in word.chars:
                    report.n_chars += 1
                    if char.label is not None:
                        report.n_chars_with_label += 1
                    if char.dot_points:
                        report.n_chars_with_dots += 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a directory of PageAnnotation JSON files.")
    parser.add_argument("annotation_dir", type=Path)
    args = parser.parse_args()

    report = validate_directory(args.annotation_dir)
    print(
        f"pages={report.n_pages} lines={report.n_lines} words={report.n_words} "
        f"chars={report.n_chars} labeled_chars={report.n_chars_with_label} "
        f"chars_with_dots={report.n_chars_with_dots}"
    )
    if report.errors:
        print(f"\n{len(report.errors)} file(s) failed to parse:", file=sys.stderr)
        for error in report.errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
