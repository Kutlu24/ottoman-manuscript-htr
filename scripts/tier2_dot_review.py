"""Working script for the manual (AI-verified) Tier-2 dot review pass.
Not part of the package -- a throwaway tool for this one annotation task.

Usage:
  python scripts/tier2_dot_review.py sheet <page_name> <line_start> <line_end> <out_png>
      Renders lines [line_start, line_end) of a page, stacked, with every
      'kept' candidate numbered, for visual review.
  python scripts/tier2_dot_review.py record <page_name> <line_index> <confirmed_indices_csv>
      Marks a line's candidates at the given indices (0-based, into that
      line's full candidate list, matching the numbers drawn by `sheet`)
      as confirmed dots, and the line's status as done.
  python scripts/tier2_dot_review.py add <page_name> <line_index> <cx> <cy>
      Adds a dot the automatic candidate list missed (a recall fix),
      found by direct visual inspection.
  python scripts/tier2_dot_review.py status
      Prints how many of the 446 lines are done.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FULL_CANDIDATES = json.loads(Path('/tmp/via_pilot/full_candidates_v2.json').read_text())
PROGRESS_PATH = ROOT / 'data/raw/makhzan/tier2_dot_review/progress.json'
SUBSET_DIR = ROOT / 'data/raw/makhzan/tier2_subset'


def load_progress() -> dict:
    return json.loads(PROGRESS_PATH.read_text())


def save_progress(p: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(p, ensure_ascii=False, indent=1))


def cmd_sheet(page_name: str, line_start: int, line_end: int, out_path: str) -> None:
    entry = FULL_CANDIDATES[page_name]
    img = Image.open(SUBSET_DIR / entry['image_path']).convert('L')
    scale = 6
    pad = 8
    tiles = []
    for li in range(line_start, line_end):
        line = entry['lines'][li]
        x, y, w, h = line['bbox']
        crop = img.crop((x - pad, y - pad, x + w + pad, y + h + pad)).convert('RGB')
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        draw = ImageDraw.Draw(crop)
        kept_idx = 0
        for c in line['candidates']:
            if not c['kept']:
                continue
            px = (c['cx'] + pad) * scale
            py = (c['cy'] + pad) * scale
            draw.ellipse([px - 12, py - 12, px + 12, py + 12], outline=(255, 0, 0), width=2)
            draw.text((px - 6, py - 28), str(kept_idx), fill=(255, 0, 0))
            kept_idx += 1
        label = f'line {li}: {line["text"]}'
        label_h = 28
        labeled = Image.new('RGB', (crop.width, crop.height + label_h), (255, 255, 255))
        ld = ImageDraw.Draw(labeled)
        ld.text((4, 4), label, fill=(0, 0, 0))
        labeled.paste(crop, (0, label_h))
        tiles.append(labeled)

    total_h = sum(t.height for t in tiles) + 10 * (len(tiles) - 1)
    max_w = max(t.width for t in tiles)
    sheet = Image.new('RGB', (max_w, total_h), (200, 200, 200))
    yoff = 0
    for t in tiles:
        sheet.paste(t, (0, yoff))
        yoff += t.height + 10
    sheet.save(out_path)
    print(f'wrote {out_path} ({sheet.size}), lines {line_start}-{line_end - 1}')
    for li in range(line_start, line_end):
        n_kept = sum(1 for c in entry['lines'][li]['candidates'] if c['kept'])
        print(f'  line {li}: {n_kept} candidates -- {entry["lines"][li]["text"]}')


def cmd_record(page_name: str, line_index: int, confirmed_csv: str) -> None:
    entry = FULL_CANDIDATES[page_name]
    line = entry['lines'][line_index]
    kept = [c for c in line['candidates'] if c['kept']]
    confirmed_indices = [int(x) for x in confirmed_csv.split(',') if x.strip() != ''] if confirmed_csv.strip() else []
    confirmed_points = [[kept[i]['cx'], kept[i]['cy']] for i in confirmed_indices]

    progress = load_progress()
    rec = progress[page_name][str(line_index)]
    rec['status'] = 'done'
    rec['confirmed'] = confirmed_points
    save_progress(progress)
    print(f'{page_name} line {line_index}: confirmed {len(confirmed_points)} dot(s)')


def cmd_add(page_name: str, line_index: int, cx: float, cy: float) -> None:
    progress = load_progress()
    rec = progress[page_name][str(line_index)]
    rec['added'].append([cx, cy])
    save_progress(progress)
    print(f'{page_name} line {line_index}: added missed dot at ({cx},{cy})')


def cmd_status() -> None:
    progress = load_progress()
    total = 0
    done = 0
    total_dots = 0
    for page, lines in progress.items():
        for li, rec in lines.items():
            total += 1
            if rec['status'] == 'done':
                done += 1
                total_dots += len(rec['confirmed']) + len(rec['added'])
    print(f'{done}/{total} lines done ({100*done/total:.1f}%), {total_dots} confirmed dots so far')
    for page, lines in progress.items():
        page_done = sum(1 for r in lines.values() if r['status'] == 'done')
        if 0 < page_done < len(lines):
            print(f'  in progress: {page} {page_done}/{len(lines)}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'sheet':
        cmd_sheet(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif cmd == 'record':
        cmd_record(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else '')
    elif cmd == 'add':
        cmd_add(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    elif cmd == 'status':
        cmd_status()
    else:
        raise SystemExit(f'unknown command {cmd}')
