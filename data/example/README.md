# Example annotation

`manuscript_0001_folio_002r.json` is one illustrative `PageAnnotation` (schema in
`src/ottoman_htr/data/schema.py`), not real transcribed manuscript data — the boxes are
plausible but hand-invented, meant to show annotators/tooling the exact shape expected from each
of the ~150 real pages.

## Conventions

- **Coordinates**: pixels in the scan's native resolution, origin top-left, `x` increasing
  rightward — same for every box regardless of script reading direction.
- **Reading order**: Arabic-script words/chars are read right-to-left, so within a line's
  `words` array (and a word's `chars` array), entries go from *highest x to lowest x* — the
  first word in the array is the rightmost one on the page, matching how a reader's eye moves.
- **`dot_points`**: only for characters whose dots are visually separable ink marks (e.g. `ت`,
  `ن`, `ف`); omit (`[]`) for characters without dots rather than guessing a position.
- **`contour`**: optional per character; leave `[]` until contour-tracing tooling exists
  (`docs/architecture.md` notes this isn't wired up yet). Filled in here for one character only,
  to show the expected `{x, y}` point-list shape.
- **`image_path`**: relative to the annotation directory itself (here, `images/...`), not to the
  JSON file's own location or the repo root — `OttomanLineDataset` resolves it that way.

## Validating a batch of annotations

Once real pages start landing in a directory (JSON files + an `images/` subfolder), check they
all parse and get a quick summary with:

```bash
python -m ottoman_htr.data.validate path/to/annotation_dir
```
