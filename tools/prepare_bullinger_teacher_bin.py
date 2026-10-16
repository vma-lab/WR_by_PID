#!/usr/bin/env python3
"""BullingerDB teacher training data (Table 4, BullingerDB block, Stage 1).

Takes the provided binarised pages of the writer-retrieval TRAIN split and
converts them to the white-ink-on-black convention of the reference
AttMask trainer (the convention of the provided Historical-WI binarised
pages the HisIR19 teacher trains on). Polarity is detected: black-on-white
sources are inverted (255 - x), white-on-black sources are copied as they
are. The output is one flat directory (label-free training).

NAMING: the reference loader parses the writer id as
int(name.split('-')[0]). BullingerDB names its files
WRITER_LETTER_PAGE.png with underscores, so the copy replaces the FIRST
underscore by a hyphen: 8018_11448_0001.png -> 8018-11448_0001.png. The
reference code stays untouched.

SCALE: the scans are about 2.4x larger than the Historical-WI pages the
recipe operates on (median long side 3,982 vs 1,644 px), which makes the
trainer's per-sample full-page morphological augmentation CPU-bound. All
pages are therefore downscaled by a fixed factor of 0.5 (median long side
about 1,990 px, within the Historical-WI spread) and re-thresholded to
bilevel (> 127). A fixed factor keeps the relative scales of the
consistently scanned corpus; the colour pages use the same factor
(tools/prepare_bullinger_scale05.py).

Also copies letters.csv to data/bullinger_letters.csv for
tools/bullinger_ndcg.py.

    data/BullingerDB/bullingerdb-wr/**/train/*.png
        -> data/bullinger_teacher_bin/*.png   (4,396 files)

    python tools/prepare_bullinger_teacher_bin.py    # idempotent; verifies
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BULL = Path('data/BullingerDB')
DST = Path('data/bullinger_teacher_bin')
LETTERS_SRC = BULL / 'letters.csv'
LETTERS_DST = Path('data/bullinger_letters.csv')
SCALE = 0.5

wr = BULL / 'bullingerdb-wr'
hits = [d for d in wr.rglob('train') if d.is_dir()]
if len(hits) != 1:
    sys.exit(f"!! expected exactly one train dir under {wr}, found {len(hits)}")
src = hits[0]
files = sorted(p for p in src.iterdir() if p.suffix.lower() == '.png')
if len(files) != 4396:
    sys.exit(f"!! {src} has {len(files)} PNGs, expected 4,396")

means = [float(np.asarray(Image.open(p).convert('L')).mean())
         for p in files[:: len(files) // 12][:12]]
light_bg = sum(m > 127 for m in means)
if light_bg not in (0, 12):
    sys.exit(f"!! mixed polarity in sample ({light_bg}/12 light) - investigate")
invert = light_bg == 12
print(f"source polarity: {'black-on-white -> INVERTING' if invert else 'white-on-black -> copying as-is'}"
      f"  (sample means {[f'{m:.0f}' for m in means[:4]]}...)")

DST.mkdir(parents=True, exist_ok=True)

done = 0
for p in files:
    if '_' not in p.name or not p.name.split('_')[0].isdigit():
        sys.exit(f"!! unexpected source name {p.name!r} - cannot derive writer id")
    out = DST / p.name.replace('_', '-', 1)
    if out.exists():
        done += 1
        continue
    g = np.asarray(Image.open(p).convert('L'))
    if invert:
        g = 255 - g
    im = Image.fromarray(g.astype(np.uint8))
    w, h = im.size
    im = im.resize((max(1, round(w * SCALE)), max(1, round(h * SCALE))),
                   Image.LANCZOS)
    arr = (np.asarray(im) > 127).astype(np.uint8) * 255
    Image.fromarray(arr).save(out)
    done += 1
    if done % 1000 == 0:
        print(f"  {done}/{len(files)}")

if LETTERS_SRC.exists() and not LETTERS_DST.exists():
    LETTERS_DST.write_bytes(LETTERS_SRC.read_bytes())

outs = sorted(DST.glob('*.png'))
vmeans = [float(np.asarray(Image.open(p).convert('L')).mean())
          for p in outs[:: max(1, len(outs) // 20)][:20]]
dark_bg = sum(1 for m in vmeans if m <= 127)
try:
    writers = {int(p.name.split('-')[0]) for p in outs}
    parse_ok = True
except ValueError:
    writers, parse_ok = set(), False
sides = sorted(max(Image.open(p).size) for p in outs[:: max(1, len(outs) // 20)][:20])
med_side = sides[len(sides) // 2] if sides else 0
print("=== VERIFICATION ===")
print(f"  sample median long side={med_side}px "
      f"(target ~1,990; ICDAR17 median 1,644)")
print(f"  files={len(outs)} (expected 4,396)  letters.csv copied: {LETTERS_DST.exists()}")
print(f"  polarity sample: {dark_bg}/{len(vmeans)} dark-background "
      f"(MUST be all - white ink on black, the trainer's convention)")
print(f"  reference-loader name parse: {'OK' if parse_ok else 'FAILED'}  "
      f"writers={len(writers)} (expected 164)  e.g. {outs[0].name if outs else '-'}")
ok = (len(outs) == 4396 and dark_bg == len(vmeans) and parse_ok
      and len(writers) == 164)
print("  OVERALL:", "OK - teacher training data ready" if ok else "FAILED - do not train")
sys.exit(0 if ok else 1)
