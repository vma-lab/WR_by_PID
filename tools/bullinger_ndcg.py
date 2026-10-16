#!/usr/bin/env python3
"""Temporal nDCG for Bullinger rows, reported under BOTH rankings:

  * OURS   - the reported pipeline's ranking: the variant's final
    descriptors (PCA or ESVM per the post-cache) + the harness's own
    kRNN query expansion (imported, not re-implemented; QE k=2,
    alpha=1, reciprocal - the frozen reported values), then cosine.
    The mAP printed for this ranking should closely match the harness CSV
    for the same variant - a free cross-validation of the script.
  * COSINE - plain cosine over the same descriptors: the BullingerDB
    paper's metric protocol, the like-for-like column next to their
    published nDCG.

Graded relevance per their definition:
    rel_q(k) = 1 - |year_k - year_q| / T_max   if writer_k == writer_q else 0
DCG over the full gallery, normalised by the ideal ranking, mean over
queries.

Reads the tag's descriptor cache: post_*.pt when the requested variant
applied PCA/ESVM (run this RIGHT AFTER that variant - the post cache holds
the LAST variant computed), else eval_*.pt. Verifies the cache meta matches
the variant and refuses otherwise.

Prints the aggregates of both rankings and writes their per-query rows
(query, writer, year, ap, ndcg, top1_hit; empty for a query without a
same-writer page) to
results/scenario1_<tag>/bullinger_<variant>_{ours,cosine}_per_query.csv.

    python tools/bullinger_ndcg.py --tag bullinger_trained --variant esvm_nopca
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leave_one_out import apply_query_expansion

T_MAX = 51.0
QE_K, QE_ALPHA = 2, 1.0

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--variant', default='std', choices=['std', 'esvm_nopca'])
ap.add_argument('--letters', default='data/bullinger_letters.csv')
args = ap.parse_args()

cache_dir = Path(f'results/scenario1_{args.tag}/desc_cache')
post = cache_dir / 'post_vlad100_patch.pt'
evalf = cache_dir / 'eval_vlad100_patch.pt'
src = None
if post.exists():
    d = torch.load(post, map_location='cpu', weights_only=False)
    m = d.get('meta', {})
    want_esvm = args.variant == 'esvm_nopca'
    if bool(m.get('esvm')) == want_esvm and \
       ((int(m.get('pca_dim', 0)) == 0) == want_esvm):
        src = d
    else:
        print(f"[warn] post cache holds a different variant "
              f"(pca_dim={m.get('pca_dim')}, esvm={m.get('esvm')}) - ", end='')
if src is None:
    if args.variant == 'esvm_nopca':
        sys.exit("!! no matching post cache for esvm_nopca - re-run that "
                 "variant, then this script immediately after")
    print("falling back to raw eval descriptors (pre-PCA) - the std nDCG "
          "then reflects the un-whitened space; prefer running right after "
          "the std variant")
    if not evalf.exists():
        sys.exit(f"!! {evalf} missing")
    src = torch.load(evalf, map_location='cpu', weights_only=False)

imgs = [Path(p) for p in src['valid_imgs']]
E = src['E'].float()
E = torch.nn.functional.normalize(E, dim=1)
N = E.shape[0]
print(f"tag={args.tag} variant={args.variant}  descriptors: {N} x {E.shape[1]}")

writers = np.array([p.parent.name for p in imgs])
letters = [p.stem.split('_')[0] for p in imgs]
year_by_letter = {r['letter_id']: float(r['year'])
                  for r in csv.DictReader(open(args.letters))}
missing = [l for l in set(letters) if l not in year_by_letter]
if missing:
    sys.exit(f"!! {len(missing)} letters missing from letters.csv "
             f"(first: {missing[:5]})")
years = np.array([year_by_letter[l] for l in letters])
span = years.max() - years.min()
print(f"writers={len(set(writers))}  year span={years.min():.0f}-"
      f"{years.max():.0f} ({span:.0f}; T_max constant = {T_MAX:.0f})")

wid = {w: i for i, w in enumerate(sorted(set(writers)))}
w_idx = np.array([wid[w] for w in writers])
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log2 = np.log2(np.arange(2, N + 1))


def score(Emat, label, name):
    Eg = Emat.to(dev)
    ndcgs, aps, top1 = [], [], 0
    rows = []
    CH = 256
    for s in range(0, N, CH):
        e = min(s + CH, N)
        sims = (Eg[s:e] @ Eg.T).cpu().numpy()
        for qi in range(s, e):
            row = sims[qi - s].copy()
            row[qi] = -np.inf
            order = np.argsort(-row)[: N - 1]
            same = w_idx[order] == w_idx[qi]
            if not same.any():
                rows.append([imgs[qi].name, writers[qi], f"{years[qi]:g}",
                             "", "", ""])
                continue
            rel = np.where(same,
                           1.0 - np.abs(years[order] - years[qi]) / T_MAX, 0.0)
            rel = np.clip(rel, 0.0, 1.0)
            dcg = float((((2.0 ** rel) - 1.0) / log2).sum())
            ideal = np.sort(rel)[::-1]
            idcg = float((((2.0 ** ideal) - 1.0) / log2).sum())
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            hits = np.flatnonzero(same)
            aps.append(float(np.mean(np.arange(1, len(hits) + 1) / (hits + 1))))
            top1 += int(same[0])
            rows.append([imgs[qi].name, writers[qi], f"{years[qi]:g}",
                         f"{aps[-1]:.6f}", f"{ndcgs[-1]:.6f}", int(same[0])])
    print(f"[NDCG:{label}] queries={len(ndcgs)}  "
          f"temporal nDCG={np.mean(ndcgs)*100:.2f}  "
          f"mAP={np.mean(aps)*100:.2f}  Top-1={top1/len(ndcgs)*100:.2f}")
    out = cache_dir.parent / f"bullinger_{args.variant}_{name}_per_query.csv"
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query', 'writer', 'year', 'ap', 'ndcg', 'top1_hit'])
        w.writerows(rows)
    print(f"[NDCG:{label}] per-query rows -> {out}")


E_qe = apply_query_expansion(E.clone(), QE_K, QE_ALPHA, dev, reciprocal=True)
E_qe = torch.nn.functional.normalize(E_qe.float().cpu(), dim=1)
score(E_qe, f'ours(QE k{QE_K})', 'ours')

score(E, 'cosine(their protocol)', 'cosine')
