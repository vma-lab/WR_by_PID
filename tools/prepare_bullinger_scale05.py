#!/usr/bin/env python3
"""BullingerDB colour splits downscaled by a fixed factor of 0.5.

BullingerDB scans are about 2.4x larger than the Historical-WI pages the
recipe operates on (median long side 3,982 vs 1,644 px); the factor 0.5
gives a median long side of about 1,990 px, within the Historical-WI
spread. Every BullingerDB row uses this one scale: the teacher's binarised
training pages (tools/prepare_bullinger_teacher_bin.py, the Stage-2
teacher view of configs/stage2_bullinger_student.yaml), the student's
colour training input and the 16,502-page test set. The script writes
fixed-factor copies of both colour splits (same writer folders and file
stems; LANCZOS; JPEG quality 92):

    data/bullinger_wr_train_colour -> data/bullinger_wr_train_colour05
    data/bullinger_wr_test_colour  -> data/bullinger_wr_test_colour05

The two source folders are described in data/README.md. A fixed factor
(not a long-side cap) keeps the relative scales of the consistently
scanned corpus intact. Idempotent per file; a version marker per output
directory.

    python tools/prepare_bullinger_scale05.py
"""
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DL = Path('data')
SCALE = 0.5
VERSION = 'colour-scale0.5-v1'
EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
PAIRS = [('bullinger_wr_train_colour', 'bullinger_wr_train_colour05', 4396),
         ('bullinger_wr_test_colour', 'bullinger_wr_test_colour05', 16502)]

for src_name, dst_name, expect in PAIRS:
    src, dst = DL / src_name, DL / dst_name
    if not src.is_dir():
        sys.exit(f"!! {src} missing - see data/README.md (BullingerDB)")
    files = [p for p in sorted(src.rglob('*')) if p.suffix.lower() in EXTS]
    if len(files) != expect:
        sys.exit(f"!! {src} has {len(files)} images, expected {expect}")
    marker = dst / '_prep_version.txt'
    have = marker.read_text().strip() if marker.exists() else None
    if dst.exists() and have != VERSION:
        sys.exit(f"!! {dst} holds outputs of {have!r}, this script is "
                 f"{VERSION!r} - rm -rf {dst} and re-run")
    dst.mkdir(parents=True, exist_ok=True)
    marker.write_text(VERSION + '\n')
    done = skip = 0
    for p in files:
        out = (dst / p.relative_to(src)).with_suffix('.jpg')
        if out.exists():
            skip += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(p).convert('RGB')
        w, h = im.size
        im = im.resize((max(1, round(w * SCALE)), max(1, round(h * SCALE))),
                       Image.LANCZOS)
        im.save(out, quality=92)
        done += 1
        if done % 2000 == 0:
            print(f"  {src_name}: {done}/{len(files)}")
    outs = [p for p in dst.rglob('*') if p.suffix.lower() == '.jpg']
    sides = sorted(max(Image.open(p).size)
                   for p in outs[:: max(1, len(outs) // 12)][:12])
    med = sides[len(sides) // 2] if sides else 0
    writers = {p.parent.name for p in outs}
    print(f"=== {dst_name}: files={len(outs)} (expected {expect})  "
          f"writers={len(writers)}  new={done} reused={skip}  "
          f"sample median long side={med}px (target ~1,990)")
print("=== VERIFICATION done ===")
