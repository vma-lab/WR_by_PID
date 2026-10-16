"""Reorganise an image dataset into writer-ID subfolders.

Reads a CSV ground-truth file (e.g. the HisIR19 ground truth) that maps
filenames to writer IDs, then copies every matched image into::

    <output_dir>/<writer_id>/<original_filename>

The resulting layout is the one ``torchvision.ImageFolder``,
``data_loader.py`` and ``leave_one_out.py`` read (see data/README.md).

Usage (from the repository root)
--------------------------------
python tools/reorganize_by_writer.py --input_dir <raw> --csv <gt.csv> \
    --output_dir data/wi_comp_19_test
"""

import argparse
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

SEP = "=" * 60


def _parse_csv(csv_path):
    """Return a dict mapping filename -> writer_id (both as strings), plus raw df."""
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding_errors="replace")
    raw_columns = list(df.columns)
    df.columns = [str(c).strip().lower() for c in df.columns]

    fname_col = next(
        (c for c in ("filename", "filenema", "file") if c in df.columns), None
    )
    label_col = next(
        (c for c in ("label", "id", "writer", "writer_id") if c in df.columns), None
    )

    if fname_col is None or label_col is None:
        raise ValueError(
            f"CSV must contain a filename column (filename/filenema/file) and a "
            f"writer column (label/id/writer/writer_id). Got: {raw_columns}"
        )

    mapping = {}
    for _, row in df.iterrows():
        mapping[str(row[fname_col]).strip()] = str(row[label_col]).strip()
    return mapping, raw_columns, fname_col, label_col


def _norm(p):
    """Normalise a path string for safe comparison (UNC-safe, no Path.resolve)."""
    return os.path.normpath(str(p))


def _find_images(input_dir):
    """Recursively find all image files under *input_dir*."""
    root = Path(os.path.normpath(input_dir))
    imgs = sorted(
        f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTS
    )
    return imgs


def _match_images(img_paths, csv_map):
    """Match each image to a writer ID via the CSV mapping.

    Tries basename then stem (no extension).
    Returns (matched, unmatched) lists.
    """
    matched = []
    unmatched = []

    for p in img_paths:
        writer = None
        for candidate in (p.name, p.stem):
            if candidate in csv_map:
                writer = csv_map[candidate]
                break
        if writer is not None:
            matched.append((p, writer))
        else:
            unmatched.append(p)

    return matched, unmatched


def _validate_image(path):
    """Try to open an image with PIL. Returns None on success, error string on failure."""
    try:
        with Image.open(path) as img:
            img.verify()
        return None
    except Exception as exc:
        return str(exc)


def reorganize(input_dir, csv_path, output_dir, *, symlink=False, verify=False):
    """Copy (or symlink) images into writer-ID subfolders and print a report."""
    t0 = time.time()

    csv_map, raw_columns, fname_col, label_col = _parse_csv(csv_path)
    csv_writers = set(csv_map.values())

    img_paths = _find_images(input_dir)
    if not img_paths:
        print("[ERROR] No images found. Check --input_dir.", file=sys.stderr)
        sys.exit(1)

    ext_counts = Counter(p.suffix.lower() for p in img_paths)
    source_dirs = set(str(p.parent) for p in img_paths)

    matched, unmatched = _match_images(img_paths, csv_map)
    if not matched:
        print("[ERROR] No images matched the CSV. Check filenames.", file=sys.stderr)
        sys.exit(1)

    matched_keys = set()
    for p, _ in matched:
        matched_keys.add(p.name)
        matched_keys.add(p.stem)
    csv_only = [k for k in csv_map if k not in matched_keys]

    out = Path(os.path.normpath(output_dir))
    copied = 0
    skipped_exists = 0
    copy_errors = []
    corrupt_images = []
    total_bytes = 0

    for src, writer_id in matched:
        dst_dir = out / writer_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if dst.exists():
            skipped_exists += 1
            continue

        if verify:
            err = _validate_image(src)
            if err is not None:
                corrupt_images.append((src, err))
                continue

        try:
            if symlink:
                os.symlink(os.path.abspath(str(src)), str(dst))
            else:
                shutil.copy2(src, dst)
            copied += 1
            try:
                total_bytes += src.stat().st_size
            except OSError:
                pass
        except Exception as exc:
            copy_errors.append((src, str(exc)))

    elapsed = time.time() - t0

    writer_counts = Counter(wid for _, wid in matched)
    n_writers = len(writer_counts)
    counts = sorted(writer_counts.values())
    single_image_writers = sum(1 for c in counts if c == 1)
    min_imgs = counts[0] if counts else 0
    max_imgs = counts[-1] if counts else 0
    median_imgs = counts[len(counts) // 2] if counts else 0
    mean_imgs = sum(counts) / len(counts) if counts else 0

    usable_writers = sum(1 for c in counts if c >= 2)
    usable_images = sum(c for c in counts if c >= 2)

    print(f"\n{SEP}")
    print(f"  DATASET REORGANIZATION REPORT")
    print(SEP)

    print(f"\n--- Input ---")
    print(f"  Input directory:    {input_dir}")
    print(f"  CSV file:           {csv_path}")
    print(f"  CSV columns:        {raw_columns}")
    print(f"  CSV columns used:   filename='{fname_col}', writer='{label_col}'")
    print(f"  Output directory:   {output_dir}")
    print(f"  Mode:               {'symlink' if symlink else 'copy'}")
    if verify:
        print(f"  Image verification: enabled")

    print(f"\n--- Source Statistics ---")
    print(f"  Images found on disk:   {len(img_paths)}")
    print(f"  Source subdirectories:   {len(source_dirs)}")
    print(f"  File types:             {dict(ext_counts)}")
    print(f"  Entries in CSV:         {len(csv_map)}")
    print(f"  Writers in CSV:         {len(csv_writers)}")

    print(f"\n--- Matching ---")
    print(f"  Images matched to CSV:  {len(matched)}")
    print(f"  Images NOT in CSV:      {len(unmatched)}")
    if unmatched:
        sample = [p.name for p in unmatched[:5]]
        print(f"    Sample unmatched:     {sample}")
    print(f"  CSV entries without image: {len(csv_only)}")
    if csv_only:
        print(f"    Sample CSV-only:      {csv_only[:5]}")

    print(f"\n--- Copy Results ---")
    print(f"  Successfully copied:    {copied}")
    if skipped_exists:
        print(f"  Skipped (already exist):{skipped_exists}")
    if copy_errors:
        print(f"  Copy errors:            {len(copy_errors)}")
        for src, err in copy_errors[:5]:
            print(f"    {src.name}: {err}")
    if corrupt_images:
        print(f"  Corrupt/unreadable:     {len(corrupt_images)}")
        for src, err in corrupt_images[:5]:
            print(f"    {src.name}: {err}")
    if total_bytes > 0:
        if total_bytes >= 1 << 30:
            size_str = f"{total_bytes / (1 << 30):.2f} GB"
        else:
            size_str = f"{total_bytes / (1 << 20):.1f} MB"
        print(f"  Total size copied:      {size_str}")

    print(f"\n--- Writer Statistics ---")
    print(f"  Total writers:          {n_writers}")
    print(f"  Images per writer:")
    print(f"    Min:                  {min_imgs}")
    print(f"    Max:                  {max_imgs}")
    print(f"    Median:               {median_imgs}")
    print(f"    Mean:                 {mean_imgs:.1f}")
    print(f"  Writers with 1 image:   {single_image_writers}")
    print(f"  Writers with >=2 images:{usable_writers}")
    print(f"  Usable images (>=2/wr): {usable_images}")

    if single_image_writers:
        print(
            f"\n  [!] {single_image_writers} writers have only 1 image and cannot\n"
            f"      contribute positive pairs for metric learning. They will be\n"
            f"      excluded from Recall@K and contrastive/triplet evaluation."
        )

    top5 = writer_counts.most_common(5)
    bottom5 = writer_counts.most_common()[:-6:-1]
    print(f"\n  Top-5 writers by image count:")
    for wid, cnt in top5:
        print(f"    Writer {wid}: {cnt} images")
    print(f"  Bottom-5 writers by image count:")
    for wid, cnt in bottom5:
        print(f"    Writer {wid}: {cnt} images")

    has_problems = bool(copy_errors or corrupt_images)
    print(f"\n--- Status ---")
    print(f"  Elapsed time:           {elapsed:.1f}s")
    if has_problems:
        print(f"  Result:                 COMPLETED WITH WARNINGS")
        print(f"  Action required:        Review errors above before training.")
    elif len(unmatched) > 0 or len(csv_only) > 0:
        print(f"  Result:                 COMPLETED (minor mismatches)")
        print(f"  Note:                   Some images/CSV entries could not be paired.")
        print(f"                          This is normal if the CSV covers a superset")
        print(f"                          or the directory has extra files.")
    else:
        print(f"  Result:                 SUCCESS (all images matched and copied)")

    print(SEP)
    print(f"  Output ready at: {out}")
    print(f"  Use as: --input_dir {out}")
    print(SEP)


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize images into writer-ID subfolders using a CSV ground truth."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root directory containing the original images (searched recursively).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV ground truth file (columns: filename + writer id).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination root. Writer-ID subfolders will be created here.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying files (saves disk space).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify each image can be opened by PIL before copying. "
        "Corrupt images are skipped and reported.",
    )
    args = parser.parse_args()

    norm_out = _norm(os.path.abspath(args.output_dir))
    norm_in = _norm(os.path.abspath(args.input_dir))
    if norm_out.startswith(norm_in + os.sep) or norm_out == norm_in:
        print(
            "[ERROR] output_dir is inside input_dir - this would cause recursion.",
            file=sys.stderr,
        )
        sys.exit(1)

    reorganize(
        args.input_dir, args.csv, args.output_dir,
        symlink=args.symlink, verify=args.verify,
    )


if __name__ == "__main__":
    main()
