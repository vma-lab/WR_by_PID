#!/usr/bin/env python3
"""Paired bootstrap between two leave-one-out per-query CSVs.

The arbiter for same-configuration comparisons, as a standalone tool:
mean per-query AP difference over the shared queries, 20,000 resamples,
percentile 95% CI, two-sided p, fixed seed.

  python tools/paired_bootstrap.py A_per_query.csv B_per_query.csv \
      [--label-a NAME] [--label-b NAME] [--cluster {query,writer}]

Reports A - B in percentage points. Rows with an empty `ap` (unscored
singleton queries) are skipped; the scored-query sets must be identical.
Queries are matched by file stem, so the colour (.jpg) and binarised
(.png) copies of a page pair up; a stem that occurs twice in one file is
an error.

--cluster query (default) resamples queries with replacement.
--cluster writer resamples writers with replacement instead: all scored
queries of a sampled writer enter the resample together, and its mean is
the sum of their AP differences divided by their number (ratio of sums).
Writers are read from the `writer` column and compared by their trailing
number when they have one, so the colour and binarised folder names of a
writer pair up; both files must assign every query to the same writer.

Example (run from the repository root):
  python tools/paired_bootstrap.py results/hisir19_student_base.csv \
    results/hisir19_reference_base.csv --label-a student --label-b reference
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

N_BOOT = 20_000
SEED = 42
CHUNK = 2_000
_TRAILING_NUMBER = re.compile(r'(\d+)$')


def writer_key(label):
    """Trailing number of a writer label ('icdar17-test_31' and
    'wi_icdar_17_test_bin_31' both give '31'); the label itself if it
    does not end in a number."""
    m = _TRAILING_NUMBER.search(label)
    return str(int(m.group(1))) if m else label


def load(path):
    """Scored queries of one CSV as {stem: (ap, top1_hit, writer label)}."""
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('ap', '') != '':
                key = Path(r['query']).stem
                if key in d:
                    sys.exit(f"duplicate query stem {key} in {path}")
                d[key] = (float(r['ap']), int(r.get('top1_hit', 0) or 0),
                          r.get('writer'))
    if not d:
        sys.exit(f"no scored queries in {path}")
    return d


def writer_ids(rows, path):
    """Normalised writer of every scored query, checked to be unambiguous."""
    labels = {k: v[2] for k, v in rows.items()}
    if any(w is None or w == '' for w in labels.values()):
        sys.exit(f"--cluster writer needs a filled 'writer' column in {path}")
    keys = {w: writer_key(w) for w in set(labels.values())}
    if len(set(keys.values())) != len(keys):
        sys.exit(f"writer labels in {path} are ambiguous after taking their "
                 f"trailing number - cannot cluster by writer")
    return {k: keys[w] for k, w in labels.items()}


def writer_resamples(d, writers, rng):
    """Cluster bootstrap over writers (ratio-of-sums mean per resample)."""
    _, inv = np.unique(writers, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv).astype(float)
    g = len(sums)
    boots = np.empty(N_BOOT)
    for s in range(0, N_BOOT, CHUNK):
        n = min(CHUNK, N_BOOT - s)
        idx = rng.integers(0, g, size=(n, g))
        boots[s:s + n] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return boots, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_a')
    ap.add_argument('csv_b')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--cluster', choices=['query', 'writer'], default='query',
                    help='resampling unit (default: query)')
    args = ap.parse_args()

    a, b = load(args.csv_a), load(args.csv_b)
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        sys.exit(f"scored-query sets differ ({len(a)} vs {len(b)}; "
                 f"{len(set(a) ^ set(b))} mismatched, e.g. "
                 f"{(only_a or only_b)[:3]}) - not a valid pair")
    keys = sorted(a)
    d = np.array([a[k][0] for k in keys]) - np.array([b[k][0] for k in keys])
    t1 = (np.mean([a[k][1] for k in keys]) - np.mean([b[k][1] for k in keys]))

    rng = np.random.default_rng(SEED)
    if args.cluster == 'query':
        boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
        unit = ''
    else:
        wa, wb = writer_ids(a, args.csv_a), writer_ids(b, args.csv_b)
        bad = [k for k in keys if wa[k] != wb[k]]
        if bad:
            sys.exit(f"writer labels differ between the two files ({len(bad)} "
                     f"in total, e.g. {bad[:3]}) - not a valid pair")
        boots, g = writer_resamples(d, [wa[k] for k in keys], rng)
        unit = f" from {g} writers, writer-cluster resampling"
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    ci = np.percentile(boots, [2.5, 97.5])
    wins, losses = int((d > 0).sum()), int((d < 0).sum())

    print(f"{args.label_a} - {args.label_b}  ({len(d)} shared queries{unit}, "
          f"{N_BOOT} resamples, seed {SEED})")
    print(f"  mAP delta: {d.mean()*100:+.3f} pt   "
          f"95% CI [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}]   "
          f"p = {'<1e-4' if p == 0 else f'{p:.4f}'}")
    print(f"  query wins/losses: {wins}/{losses}   "
          f"Top-1 delta: {t1*100:+.3f} pt (descriptive)")


if __name__ == '__main__':
    main()
