#!/usr/bin/env python3
"""Polarity conversion for the provided Historical-WI binarised pages.

The provided binarised sets are white ink on black (the convention of the
reference trainer's loader), but the evaluation's BINARISATION=input path
expects dark ink on light paper. This script writes black-ink-on-white
copies (255 - x, saved as lossless PNG, same writer folders and file
stems):

    data/wi_icdar_17_test_bin  -> data/wi_icdar_17_test_bin_inv
    data/wi_icdar_17_train_bin -> data/wi_icdar_17_train_bin_inv

    python tools/invert_histwi_bins.py     # idempotent; ends with verification

The evaluation re-binarises these pages with the same Sauvola step as
every binarised-input page (window 51, k 0.2). That step leaves strokes
unchanged, re-thresholds compression grey levels, and only clears the
interior of solid black regions wider than the 51-pixel window; the
binarised view then reaches the network as white ink on black, the
convention the ViT-S/16 models were trained on.

The original white-ink copies are kept: Stage 1 trains on them (see
data/README.md, data/histwi_train_bin), and the reference trainer's loader
detects polarity itself.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PAIRS = [
    (Path('data/wi_icdar_17_test_bin'),
     Path('data/wi_icdar_17_test_bin_inv'), 3600, 720),
    (Path('data/wi_icdar_17_train_bin'),
     Path('data/wi_icdar_17_train_bin_inv'), 1182, 394),
]
EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

for src, dst, exp_n, exp_w in PAIRS:
    if not src.exists():
        sys.exit(f"!! {src} missing")
    files = [p for p in sorted(src.rglob('*')) if p.suffix.lower() in EXTS]
    done = 0
    for p in files:
        out = dst / p.relative_to(src)
        out = out.with_suffix('.png')
        if out.exists():
            done += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        g = np.asarray(Image.open(p).convert('L'))
        Image.fromarray((255 - g).astype(np.uint8)).save(out)
        done += 1
        if done % 500 == 0:
            print(f"  {src.name}: {done}/{len(files)}")

    outs = [p for p in dst.rglob('*.png')]
    folders = {p.parent.name for p in outs}
    means = [float(np.asarray(Image.open(p).convert('L')).mean())
             for p in outs[:: max(1, len(outs) // 40)]]
    pol = sum(1 for m in means if m > 127)
    print(f"{dst.name}: images={len(outs)} (expected {exp_n})  "
          f"folders={len(folders)} (expected {exp_w})  "
          f"polarity sample: {pol}/{len(means)} white-bg (MUST be all)")
    if len(outs) != exp_n or pol != len(means):
        sys.exit(f"!! {dst.name} FAILED verification - check the source folder")
print("\nOK - inverted copies ready in data/wi_icdar_17_*_bin_inv")
