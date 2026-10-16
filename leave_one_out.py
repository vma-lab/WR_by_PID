import argparse, csv
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from checkpoint_utils import load_backbone
from inference_utils import (
    tile_image,
    embed_tiles,
    aggregate_embeddings,
    build_agg_kwargs,
    resolve_aggregation_config,
    resolve_use_patch_tokens,
    ensure_salad_pooling_available,
    resolve_image_size,
    apply_esvm_normalization,
    build_ink_mask_index,
    external_ink_scores,
)
import os
import numpy as np
from tqdm import tqdm

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}



def _passthrough_collate(batch):
    return batch[0]


class _TileImageDataset(torch.utils.data.Dataset):
    """Per-page CPU tiling, run inside DataLoader workers so it overlaps the GPU
    forward.  ``__getitem__`` returns one page's
    ``(tiles, masks, ext_ink, path, label)``; ``tiles``/``masks`` are ``None``
    when nothing survives filtering (path/label are still returned so the skip
    can be logged).
    """

    def __init__(self, imgs, labels, *, image_size, tile_size,
                 overlap, max_image_size, mean, std, config_path,
                 ink_mask_dir=None, patch_size=16):
        self.imgs = imgs
        self.labels = labels
        self.image_size = image_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_image_size = max_image_size
        self.mean = mean
        self.std = std
        self.config_path = config_path
        self.patch_size = int(patch_size)
        self.ink_mask_dir = ink_mask_dir
        self._mask_by_stem = (build_ink_mask_index(ink_mask_dir)
                              if ink_mask_dir else None)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        lab = self.labels[i]
        size = resolve_image_size(
            str(p), self.image_size, self.tile_size,
            self.overlap, self.max_image_size,
        )
        mask_path = None
        if self._mask_by_stem is not None:
            mask_path = self._mask_by_stem.get(Path(p).stem.lower())
            if mask_path is None:
                raise RuntimeError(
                    f"[ink-mask] no mask for {Path(p).name} in "
                    f"{self.ink_mask_dir}")
        tk = dict(
            image_size=size,
            tile_size=self.tile_size,
            mean=self.mean,
            std=self.std,
            overlap=self.overlap,
            config_path=self.config_path,
        )
        t, m = tile_image(str(p), **tk)
        if t is None or t.numel() == 0:
            return None, None, None, str(p), lab
        ext_ink = None
        if mask_path is not None:
            ext_ink = external_ink_scores(
                mask_path, patch_size=self.patch_size, tile_kwargs=tk)
            if ext_ink is None or ext_ink.shape[0] != t.shape[0]:
                raise RuntimeError(
                    f"[ink-mask] tile-grid mismatch for "
                    f"{Path(p).name} at size {size}: page has "
                    f"{t.shape[0]} tiles, mask "
                    f"{0 if ext_ink is None else ext_ink.shape[0]} - the mask "
                    f"is not in the same pixel frame as the page")
        return t, m, ext_ink, str(p), lab


def _save_confusion(y_true, y_pred, out_prefix):
    labels = sorted(set(y_true) | set(y_pred))
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    import pandas as pd
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(f"{out_prefix}_confusion_counts.csv")
    with np.errstate(divide='ignore', invalid='ignore'):
        cmn = cm / cm.sum(axis=1, keepdims=True)
        cmn[np.isnan(cmn)] = 0
    pd.DataFrame(cmn, index=labels, columns=labels).to_csv(f"{out_prefix}_confusion_norm.csv")


def _collect_images(input_dir):
    p = Path(input_dir)
    imgs = [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in VALID_EXTS]
    print(f"[INFO] input_dir: {input_dir}")
    print(f"[INFO] Found {len(imgs)} images")
    if imgs:
        print(f"[INFO] Sample image path: {imgs[0]}")
        print(f"[INFO] Sample parent:     {imgs[0].parent}")
    return imgs


def _load_labels(img_paths, labels_csv=None, input_dir=None):
    import os
    import pandas as pd
    root_str = os.path.normpath(input_dir)
    has_subdirs = any(os.path.normpath(str(Path(p).parent)) != root_str for p in img_paths)
    labels = {}
    if labels_csv:
        df = pd.read_csv(labels_csv, sep=None, engine="python")
        df.columns = [str(c).strip().lower() for c in df.columns]
        fname_col = next((c for c in ("filename", "filenema", "file") if c in df.columns), None)
        label_col = next((c for c in ("label", "id", "writer") if c in df.columns), None)
        if fname_col is None or label_col is None:
            raise ValueError(
                f"labels_csv must contain a filename column (filename/filenema/file) and a "
                f"label column (label/id/writer). Got columns: {list(df.columns)}"
            )
        print(f"[INFO] Using labels_csv column '{fname_col}' as filename and '{label_col}' as writer label")
        for _, row in df.iterrows():
            labels[str(row[fname_col])] = str(row[label_col])
        print(f"[INFO] Loaded {len(labels)} label entries from {labels_csv}")
    if has_subdirs and not labels:
        for p in img_paths:
            labels[str(p)] = p.parent.name
    elif not labels:
        raise ValueError("Flat folder detected. Provide --labels_csv with columns filename,label.")
    out = []
    missing_samples = []
    for p in img_paths:
        key = None
        for candidate in (str(p),
                          (str(Path(p).relative_to(input_dir)) if input_dir and str(p).startswith(str(input_dir)) else None),
                          p.name,
                          p.stem):
            if candidate is not None and candidate in labels:
                key = candidate
                break
        if key is None:
            if len(missing_samples) < 3:
                missing_samples.append(p.name)
            continue
        out.append(labels[key])
    if missing_samples:
        csv_samples = list(labels.keys())[:5]
        raise ValueError(
            f"{len(img_paths) - len(out)} of {len(img_paths)} images have no label match in the CSV.\n"
            f"  Sample missing image filenames: {missing_samples}\n"
            f"  Sample CSV filename entries:    {csv_samples}\n"
            f"  Tried matching on: full path, relative path, basename, and stem (no extension)."
        )
    print(f"[INFO] Matched {len(out)} images to writer labels ({len(set(out))} unique writers)")
    return out


def _compute_metrics(sim_row_sorted_labels, query_label, ks=(1, 3, 5)):
    positives_total = sum(1 for l in sim_row_sorted_labels if l == query_label)
    top_hits = {k: 0 for k in ks}
    ap = 0.0
    if positives_total == 0:
        return top_hits, None
    correct_seen = 0
    for rank, lab in enumerate(sim_row_sorted_labels, start=1):
        if lab == query_label:
            correct_seen += 1
            ap += correct_seen / rank
        for k in ks:
            if rank == k and correct_seen > 0:
                top_hits[k] = 1
    ap /= positives_total
    return top_hits, ap


def _pca_whiten_fit(E_fit: torch.Tensor, n_components: int, *, device: torch.device,
                    niter: int = 6, eps: float = 1e-8, exact: bool = False) -> dict:
    """Fit PCA-whitening on ``E_fit`` and return reusable transform stats.

    Equivalent to ``sklearn.decomposition.PCA(n_components, whiten=True).fit``:
    centre, take the top principal axes, record per-component std.  Uses a
    randomized SVD (``torch.pca_lowrank``) on the GPU when available (orders of
    magnitude faster than sklearn's CPU solver for the wide VLAD/SALAD
    descriptors AND sidesteps the multi-threaded OpenBLAS deadlock that hangs
    the CPU SVD on large matrices); retries on CPU torch otherwise.

    Fitting on a held-out, label-free set (e.g. the validation pages) keeps the
    protocol non-transductive: no statistic of the test collection enters the
    transform.  ``n_components`` is clamped to ``min(n, N_fit - 1, D)`` - the
    rank of the centred fit data.
    """
    N, D = int(E_fit.shape[0]), int(E_fit.shape[1])
    k = max(1, min(int(n_components), max(N - 1, 1), D))

    def _fit_on(dev: torch.device) -> dict:
        torch.manual_seed(0)
        Eg = E_fit.to(device=dev, dtype=torch.float32)
        mean = Eg.mean(dim=0, keepdim=True)
        Eg = Eg - mean
        if exact:
            _, S, Vh = torch.linalg.svd(Eg, full_matrices=False)
            V = Vh.transpose(0, 1)[:, :k]
            S = S[:k]
        else:
            q = min(k + 16, N, D)
            _, S, V = torch.pca_lowrank(Eg, q=q, center=False, niter=niter)
            V = V[:, :k]
            S = S[:k]
        std = (S / (float(max(N - 1, 1)) ** 0.5)).clamp_min(eps)
        stats = {
            'mean': mean.detach().to('cpu', dtype=torch.float32),
            'components': V.detach().to('cpu', dtype=torch.float32),
            'std': std.detach().to('cpu', dtype=torch.float32),
            'n_fit': N,
        }
        del Eg, V, S
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        return stats

    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            return _fit_on(device)
        except Exception as exc:
            print(f"[WARN] GPU PCA fit failed ({exc!r}); retrying on CPU.")
            torch.cuda.empty_cache()
    return _fit_on(torch.device('cpu'))


def _pca_whiten_apply(E: torch.Tensor, stats: dict, *, device: torch.device) -> torch.Tensor:
    """Apply previously fitted PCA-whitening stats to the rows of ``E``."""

    def _apply_on(dev: torch.device) -> torch.Tensor:
        Eg = E.to(device=dev, dtype=torch.float32)
        out = (Eg - stats['mean'].to(dev)) @ stats['components'].to(dev)
        out = out / stats['std'].to(dev).unsqueeze(0)
        out = out.detach().to('cpu', dtype=torch.float32).contiguous()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        return out

    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            return _apply_on(device)
        except Exception as exc:
            print(f"[WARN] GPU PCA apply failed ({exc!r}); retrying on CPU.")
            torch.cuda.empty_cache()
    return _apply_on(torch.device('cpu'))


def _pca_whiten(E: torch.Tensor, n_components: int, *, device: torch.device,
                niter: int = 6, eps: float = 1e-8) -> torch.Tensor:
    """Fit-and-apply PCA-whitening on the SAME matrix (transductive).

    Legacy path, kept for ``--pca_dim`` without ``--pca_fit_dir``: the whitening
    statistics then come from the evaluation collection itself - label-free but
    transductive.  Prefer ``--pca_fit_dir`` (fit on validation pages) for a
    protocol directly comparable to published train-fit numbers.
    """
    stats = _pca_whiten_fit(E, n_components, device=device, niter=niter, eps=eps)
    return _pca_whiten_apply(E, stats, device=device)


def _embed_collection(
    imgs,
    labels,
    *,
    desc: str,
    image_size,
    tile_size,
    overlap,
    max_image_size,
    mean,
    std,
    config_path: str,
    embed_workers: int,
    backbone,
    device: torch.device,
    batch_size: int,
    use_patch_tokens: bool,
    patch_size: int,
    ink_threshold,
    ink_mask_dir=None,
    agg_kwargs: dict,
    salad_cache,
    max_tokens: int = 0,
    token_l2: bool = True,
    token_proj=None,
):
    """Embed a collection of pages into one descriptor per page.

    This is the single embedding path for the evaluation corpus AND for any
    auxiliary, label-free collections (PCA-whitening fit set, ESVM negative
    pool), so every descriptor is produced by the identical pipeline.

    Returns ``(E [N, D], valid_imgs, valid_labels)``; ``E`` is ``None`` when
    nothing could be embedded.
    """
    embs = []
    valid_imgs = []
    valid_labels = []
    tile_ds = _TileImageDataset(
        imgs, labels,
        image_size=image_size, tile_size=tile_size,
        overlap=overlap, max_image_size=max_image_size,
        mean=mean, std=std, config_path=config_path,
        ink_mask_dir=ink_mask_dir, patch_size=patch_size,
    )
    if embed_workers > 0:
        embed_iter: Any = torch.utils.data.DataLoader(
            tile_ds, batch_size=1, shuffle=False,
            num_workers=embed_workers, collate_fn=_passthrough_collate,
            pin_memory=False, persistent_workers=False,
        )
    else:
        embed_iter = (tile_ds[i] for i in range(len(tile_ds)))
    for item in tqdm(embed_iter, total=len(tile_ds), desc=desc):
        tiles, masks, ext_ink, p_str, lab = item
        if tiles is None:
            print(f"[skip] {Path(p_str).name}: no tiles after filtering")
            continue
        edge_mask = None
        if use_patch_tokens:
            if masks is not None and masks.numel() > 0:
                edge_mask = masks.reshape(-1)
        else:
            edge_mask = masks
        te, cls_desc, ink_scores = embed_tiles(
            tiles,
            backbone,
            device,
            batch_size,
            use_patch_tokens=use_patch_tokens,
            token_l2=token_l2,
            token_proj=token_proj,
        )
        if ext_ink is not None:
            ink_scores = ext_ink.reshape(-1)
        agg_mask = edge_mask
        if (use_patch_tokens
                and ink_threshold is not None
                and not (ink_scores is not None
                         and ink_scores.numel() == te.shape[0])):
            raise RuntimeError(
                f"[eval] ink gate requested (ink_threshold={ink_threshold}) but "
                f"per-patch ink scores are "
                f"{'missing' if ink_scores is None else 'length-mismatched'} "
                f"({0 if ink_scores is None else ink_scores.numel()} vs "
                f"{te.shape[0]} tokens) - refusing the silent Canny fallback")
        if (
            use_patch_tokens
            and ink_threshold is not None
            and ink_scores is not None
            and ink_scores.numel() == te.shape[0]
        ):
            agg_mask = ink_scores.reshape(-1) >= float(ink_threshold)

        if max_tokens and int(max_tokens) > 0:
            m = (agg_mask.reshape(-1).bool()
                 if agg_mask is not None
                 else torch.ones(te.shape[0], dtype=torch.bool))
            n_fg = int(m.sum())
            if n_fg > int(max_tokens):
                idx = m.nonzero(as_tuple=True)[0]
                gtor = torch.Generator().manual_seed(0)
                keep = idx[torch.randperm(idx.numel(), generator=gtor)[:int(max_tokens)]]
                capped = torch.zeros_like(m)
                capped[keep] = True
                agg_mask = capped
                print(f"[CAP] {Path(p_str).name}: {n_fg} foreground tokens → "
                      f"capped to {int(max_tokens)} (per-page memory guard).")
        vec = aggregate_embeddings(
            te,
            mask=agg_mask,
            fallback=cls_desc,
            fallback_mode='auto',
            salad_cache_key=p_str if salad_cache is not None else None,
            **agg_kwargs,
        )
        embs.append(vec.unsqueeze(0))
        valid_imgs.append(p_str)
        valid_labels.append(lab)
    if not embs:
        return None, valid_imgs, valid_labels
    return torch.cat(embs, 0), valid_imgs, valid_labels


def _load_descriptor_cache(path: Path, src_imgs, meta: dict):
    """Load cached descriptors; return ``(E, valid_imgs, valid_labels)`` or None.

    A cache entry is only honoured when the stored embedding-geometry ``meta``
    AND the source image list match exactly - anything else recomputes (and
    overwrites), so a stale entry can never silently change a result.
    """
    if not Path(path).exists():
        return None
    try:
        blob = torch.load(path, map_location='cpu')
    except Exception as exc:
        print(f"[CACHE][warn] unreadable cache file {path} ({exc!r}); recomputing.")
        return None
    stored_meta = blob.get('meta') or {}
    _compat_defaults = {
        'token_l2': True,
        'vlad_pre_l2': True,
        'vlad_residual_norm': True,
        'codebook_overlap': stored_meta.get('overlap'),
        'pca_exact': False,
        'simple_window': False,
        'grayscale_p': 0.0,
        'distill_proj': False,
        'ink_max': None,
        'ink_method': 'sauvola',
        'ink_mask_dir': None,
        'codebook_algorithm': 'minibatch',
        'codebook_kmeans_n_init': 10,
        'ink_k': meta.get('ink_k'),
        'ink_window': meta.get('ink_window'),
        'cfg_fingerprint': meta.get('cfg_fingerprint'),
    }
    for _k, _v in _compat_defaults.items():
        if _k in meta:
            stored_meta.setdefault(_k, _v)
    if stored_meta != meta:
        diffs = [
            f"{k}: {stored_meta.get(k)!r} -> {meta.get(k)!r}"
            for k in sorted(set(stored_meta) | set(meta))
            if stored_meta.get(k) != meta.get(k)
        ]
        print(f"[CACHE][warn] {Path(path).name}: embedding key changed, recomputing. "
              f"Differing field(s): {'; '.join(diffs) if diffs else '(unknown)'}")
        return None
    if blob.get('src_imgs') != [str(p) for p in src_imgs]:
        n_stored = len(blob.get('src_imgs') or [])
        print(f"[CACHE][warn] {Path(path).name}: source image list changed "
              f"({n_stored} stored vs {len(src_imgs)} now); recomputing.")
        return None
    E = blob.get('E')
    if not isinstance(E, torch.Tensor) or E.ndim != 2:
        return None
    return E, list(blob.get('valid_imgs', [])), list(blob.get('valid_labels', []))


def _store_descriptor_cache(path: Path, src_imgs, valid_imgs, valid_labels,
                            E: torch.Tensor, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'src_imgs': [str(p) for p in src_imgs],
            'valid_imgs': list(valid_imgs),
            'valid_labels': list(valid_labels),
            'E': E.detach().cpu().to(torch.float32),
            'meta': meta,
        },
        path,
    )
    print(f"[CACHE] Stored {E.shape[0]} descriptors → {path}")


def apply_query_expansion(E: torch.Tensor, k: int, alpha: float, device: torch.device,
                          *, reciprocal: bool = True, iters: int = 1) -> torch.Tensor:
    """Average query expansion over k (reciprocal) nearest neighbours.

    Re-embeds each page descriptor as the L2-normalised weighted mean of itself
    and its ``k`` nearest neighbours, then recomputes from the updated set.
    With ``reciprocal=True`` only *mutual* neighbours reinforce (j helps i only
    if i is also among j's top-k) - the reference system's kRNN query expansion,
    robust to impostors because a one-sided hub neighbour is ignored.

    This directly targets TAIL DISPERSION: when a writer's pages form a chain
    (a-b close, b-c close, a-c far), expanding through the shared neighbour b
    pulls a and c together, lifting the 2nd/3rd/4th same-writer pages up the
    ranking without touching a correct rank-1.  Operates on whatever descriptor
    space it is handed (raw, PCA-whitened, or ESVM normals).
    """
    def _run(dev: torch.device) -> torch.Tensor:
        X = F.normalize(E.to(device=dev, dtype=torch.float32), dim=1)
        N = X.shape[0]
        kk = max(1, min(int(k), N - 1))
        idx_rows = torch.arange(N, device=dev).unsqueeze(1)
        for _ in range(max(1, int(iters))):
            sim = X @ X.t()
            sim.fill_diagonal_(-1e9)
            knn = sim.topk(kk, dim=1).indices
            membership = torch.zeros(N, N, dtype=torch.bool, device=dev)
            membership[idx_rows.expand_as(knn), knn] = True
            adj = (membership & membership.t()) if reciprocal else membership
            neigh_sum = adj.float() @ X
            X = F.normalize(X + float(alpha) * neigh_sum, dim=1)
        return X.detach().to('cpu', dtype=torch.float32).contiguous()

    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            return _run(device)
        except Exception as exc:
            print(f"[WARN] GPU query expansion failed ({exc!r}); retrying on CPU.")
            torch.cuda.empty_cache()
    return _run(torch.device('cpu'))


def _cosine_sim_matrix(E: torch.Tensor, device: torch.device) -> torch.Tensor:
    """L2-normalise rows and return the full N×N cosine-similarity matrix.

    Runs the (N×D)·(D×N) matmul on the GPU when available - essential when the
    descriptor stays high-dimensional (e.g. PCA disabled on a 49152-d VLAD/SALAD
    descriptor), where the CPU matmul is a multi-hour bottleneck.  Returns a CPU
    tensor (the argsort consumer expects CPU).
    """
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            Eg = F.normalize(E.to(device=device, dtype=torch.float32), dim=1)
            sims = (Eg @ Eg.t()).to('cpu')
            del Eg
            torch.cuda.empty_cache()
            return sims
        except Exception as exc:
            print(f"[WARN] GPU similarity matmul failed ({exc!r}); using CPU.")
            torch.cuda.empty_cache()
    En = F.normalize(E, dim=1)
    return En @ En.t()


def _score_queries(order, valid_labels, ks=(1, 3, 5), progress=False,
                   per_query=None, top_n=5):
    """Compute retrieval metrics for each query.

    When ``per_query`` is a list, one record per scoreable query is appended -
    the raw material of the error audit (which queries lose AP, to what):
    ``{'idx', 'label', 'num_pos', 'ap', 'first_hit_rank', 'top': [(j, hit)…]}``.
    Singleton-writer queries (no positives) are recorded with ``ap=None``.
    """

    iterator = range(len(valid_labels))
    if progress:
        iterator = tqdm(iterator, desc="Scoring queries")

    from collections import Counter
    label_counts = Counter(valid_labels)

    topk_hits = {k: 0 for k in ks}
    ap_list = []
    skipped = 0
    y_true, y_pred = [], []

    for i in iterator:
        row = order[i]
        ranked_idx = row.tolist() if hasattr(row, 'tolist') else list(row)
        ranked_idx = [j for j in ranked_idx if j != i]
        if not ranked_idx:
            skipped += 1
            continue

        ranked_labels = [valid_labels[j] for j in ranked_idx]

        hits, apv = _compute_metrics(ranked_labels, valid_labels[i], ks)

        if per_query is not None:
            q_label = valid_labels[i]
            first_hit = next(
                (r + 1 for r, l in enumerate(ranked_labels) if l == q_label), None
            )
            per_query.append({
                'idx': i,
                'label': q_label,
                'num_pos': label_counts[q_label] - 1,
                'ap': apv,
                'first_hit_rank': first_hit,
                'top': [(j, valid_labels[j] == q_label) for j in ranked_idx[:top_n]],
            })

        if apv is None:
            skipped += 1
            continue

        y_true.append(valid_labels[i])
        y_pred.append(ranked_labels[0])

        for k in ks:
            topk_hits[k] += hits[k]
        ap_list.append(apv)

    return topk_hits, ap_list, skipped, y_true, y_pred


def main():
    ap = argparse.ArgumentParser(description="Leave-one-out retrieval (mAP, top-k)")
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--input_dir', type=str, required=True)
    ap.add_argument('--labels_csv', type=str, default=None,
                    help="CSV with columns filename,label (required if folder is flat)")
    ap.add_argument('--output_csv', type=str, required=True)
    ap.add_argument('--method', choices=['mean', 'max', 'gem', 'vlad', 'salad'], default=None)
    ap.add_argument('--vlad_clusters', type=int, default=None)
    ap.add_argument('--metric', choices=['cosine', 'euclidean'], default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--embed_workers', type=int, default=None,
                    help='CPU worker processes for per-image tiling during embedding, '
                         'so a page is tiled while the GPU embeds the previous one. '
                         'Default: None = use training.num_workers from the config; '
                         '0 = single-process (original behaviour). Affects wall-clock '
                         'only, not the descriptors.')
    ap.add_argument('--max_tokens_per_page', type=int, default=0,
                    help='Cap foreground patch tokens per page before aggregation '
                         '(0 = unbounded). A guard against a single huge scan blowing up '
                         'the SALAD/VLAD GPU memory (which can hard-fault the driver on '
                         'dense native-resolution tiling). Deterministic subsample; no-op '
                         'below the cap. Recommended ~131072 for OVERLAP=0.75 at native '
                         'resolution.')
    ap.add_argument('--codebook_dir', type=str, default=None,
                    help='Directory of images used to AUTO-FIT a shared VLAD/SALAD '
                         'codebook (label-free). '
                         'Default: the eval --input_dir. To match the reference system, '
                         'point this at the HisIR19 VALIDATION set (the paper '
                         'fits the k-means vocabulary on the validation images, never '
                         'on labels).')
    ap.add_argument('--codebook_sample_images', type=int, default=2000,
                    help='Auto-fit codebook: sample at most this many pages for the '
                         'vocabulary (0 = use all).')
    ap.add_argument('--codebook_max_tokens', type=int, default=500000,
                    help='Auto-fit codebook: reservoir cap on tokens used for k-means.')
    ap.add_argument('--codebook_max_per_image', type=int, default=256,
                    help='Auto-fit codebook: cap tokens contributed per page.')
    ap.add_argument('--codebook_algorithm', choices=('minibatch', 'kmeans'),
                    default='minibatch',
                    help='Auto-fit codebook: k-means solver. minibatch (default, '
                         'the base configuration) or kmeans (full-batch Lloyd, the '
                         "reference system's fit). Cache-keyed: a switch re-embeds.")
    ap.add_argument('--codebook_kmeans_n_init', type=int, default=10,
                    help='Auto-fit codebook: n_init for the full-batch solver '
                         '(ignored by minibatch). Cache-keyed.')
    ap.add_argument('--vlad_power', dest='vlad_power', action='store_const', const=True,
                    default=None,
                    help='Enable signed power normalisation on VLAD residuals')
    ap.add_argument('--no-vlad_power', dest='vlad_power', action='store_const', const=False,
                    help='Disable signed power normalisation')
    ap.add_argument('--vlad_power_gamma', type=float, default=None,
                    help='Exponent for signed power normalisation (default: 0.5)')
    ap.add_argument('--vlad_power_eps', type=float, default=None,
                    help='Epsilon added before the power transform (default: 1e-12)')
    ap.add_argument('--disable_intra_norm', action='store_true',
                    help='Disable intra-cluster L2 normalisation')
    ap.add_argument('--disable_final_norm', action='store_true',
                    help='Skip the final global L2 normalisation')
    ap.add_argument('--salad_support_points', type=int, default=None,
                    help='Number of prototypes for SALAD pooling')
    ap.add_argument('--salad_reg', type=float, default=None,
                    help='Sinkhorn regularisation strength for SALAD')
    ap.add_argument('--salad_sinkhorn_iters', type=int, default=None,
                    help='Maximum Sinkhorn iterations per SALAD refinement step')
    ap.add_argument('--salad_sinkhorn_tol', type=float, default=None,
                    help='Convergence tolerance for the Sinkhorn updates')
    ap.add_argument('--salad_update_iters', type=int, default=None,
                    help='Number of outer refinement steps for SALAD')
    ap.add_argument('--salad_init', type=str, default=None,
                    choices=['kmeans++', 'random', 'identity'],
                    help='Prototype initialisation for SALAD pooling')
    ap.add_argument('--salad_cache', dest='salad_cache', action='store_true',
                    help='Enable caching of SALAD OT prototypes')
    ap.add_argument('--no-salad_cache', dest='salad_cache', action='store_false',
                    help='Disable caching of SALAD OT prototypes')
    ap.set_defaults(salad_cache=None)
    ap.add_argument('--salad_sort_by_mass', dest='salad_sort_by_mass', action='store_true',
                    help='Sort SALAD prototypes by transported mass (default)')
    ap.add_argument('--no-salad_sort_by_mass', dest='salad_sort_by_mass', action='store_false',
                    help='Preserve solver order for SALAD prototypes')
    ap.set_defaults(salad_sort_by_mass=None)
    ap.add_argument('--salad_dustbin', dest='salad_dustbin', action='store_true',
                    help='Enable SALAD dustbin dimension')
    ap.add_argument('--no-salad_dustbin', dest='salad_dustbin', action='store_false',
                    help='Disable SALAD dustbin dimension')
    ap.set_defaults(salad_dustbin=None)
    ap.add_argument('--esvm', dest='esvm', action='store_true',
                    help='Apply Exemplar-SVM re-normalisation to page descriptors before '
                         'scoring (codebook-free writer-retrieval post-processing).')
    ap.add_argument('--no-esvm', dest='esvm', action='store_false',
                    help='Disable ESVM re-normalisation.')
    ap.set_defaults(esvm=False)
    ap.add_argument('--esvm_num_neg', type=int, default=1000,
                    help='Generic negatives per exemplar SVM (<=0 = all others).')
    ap.add_argument('--esvm_C', type=float, default=1.0,
                    help='ESVM SVM regularisation strength C.')
    ap.add_argument('--esvm_n_jobs', type=int, default=1,
                    help='Parallel workers for ESVM fitting (joblib threads; 1 = serial).')
    ap.add_argument('--eval_overlap', type=float, default=-1.0,
                    help='Tile overlap to use AT INFERENCE only (overrides tiling.overlap); '
                         '-1 = use the config value. Denser (0.5, 0.75) samples more windows/page '
                         '-> more patches for the aggregator (denser tiling), at '
                         'higher LOO cost. Options: 0.0, 0.25, 0.5, 0.75.')
    ap.add_argument('--distill_proj', type=str, default='none',
                    choices=['none', 'auto'],
                    help="'auto' loads the distill_proj state stored in "
                         '--model_path (the affine map W trained during '
                         'distillation) and applies it to the RAW patch tokens '
                         'before normalisation/aggregation, so retrieval runs in '
                         "the teacher-shaped subspace; 'none' aggregates the "
                         'full-width tokens (the representation ablation of '
                         'Table 3). Part of the desc-cache key -> use a fresh '
                         'RUN_TAG.')
    ap.add_argument('--simple_window', action='store_true',
                    help="Window rule of the reference system's released code: "
                         "null out the tile keep-filter (max_padding_pct / "
                         "min_stroke_pct) so the moving window keeps EVERY "
                         "full-stride tile, and window content selection is left "
                         "entirely to the per-patch ink gate (--ink_threshold) at "
                         "aggregation. Without it the config's keep-filter applies "
                         "(bg-percentile content box + padding + Canny edges, or "
                         "the ink fraction for binary input). "
                         "Changes the descriptors -> use a fresh RUN_TAG (keyed).")
    ap.add_argument('--pca_dim', type=int, default=0,
                    help='Apply PCA-whitening and reduce descriptor to this dimensionality (0 to disable)')
    ap.add_argument('--pca_fit_dir', type=str, default=None,
                    help='Directory of images whose descriptors FIT the PCA-whitening '
                         '(label-free, test-disjoint - e.g. the validation set). '
                         'Without it the whitening is fitted on the evaluation '
                         'collection itself (label-free but transductive).')
    ap.add_argument('--esvm_neg_dir', type=str, default=None,
                    help='Directory of images whose descriptors form the ESVM negative '
                         'pool (writer-disjoint, Christlein protocol - e.g. the '
                         'validation set). Without it negatives are sampled from the '
                         'evaluation collection itself, where a query\'s own '
                         'same-writer pages can become its negatives.')
    ap.add_argument('--image_size', type=int, default=None,
                    help='Override dataset.image_size for this evaluation only '
                         '(0 = native resolution, canonical-clamped per page; '
                         '-1 = NATIVE-RAW: no resize/pad/clamp at all, sliding '
                         'window drops sub-stride edge leftovers - the reference '
                         "system's geometry).")
    ap.add_argument('--max_image_size', type=int, default=None,
                    help='Override dataset.max_image_size (long-side cap at '
                         'image_size=0; 0 disables the cap).')
    ap.add_argument('--binarisation', choices=['config', 'filter', 'input', 'off'], default='config',
                    help="Override dataset.binarisation for this evaluation only: "
                         "'filter' = the colour image as loaded to the backbone "
                         "+ Sauvola ink-count "
                         "token gate (the t_fg gate); 'input' = feed the BINARY image "
                         "to the backbone (symmetric input-norm + ink gate auto-applied; "
                         "use only with a binary-trained backbone); 'off' = Canny "
                         "edge-density gate; 'config' = leave the YAML untouched.")
    ap.add_argument('--ink_threshold', type=float, default=None,
                    help='Per-patch ink-fraction threshold for the ink-count gate '
                         '(t_fg analogue; default: tiling.binary_ink_threshold). '
                         'Active with patch tokens in BOTH binarisation filter '
                         '(image as loaded to the backbone) AND input (binary to '
                         'backbone) modes - configure_ink_foreground enables it '
                         'for either.')
    ap.add_argument('--ink_mask_dir', type=str, default=None,
                    help='Build the foreground gate from EXTERNAL per-page '
                         'binary masks in this directory (one file per page, '
                         'matched by filename stem, SAME pixel frame as the '
                         'photograph) instead of thresholding the tile itself. '
                         'The backbone input is UNCHANGED - only token '
                         'SELECTION moves. Intended for adopting a published '
                         'papyri binariser as the selector while the model '
                         'still reads the raw image. Requires --simple_window '
                         '(a content-dependent tile keep-filter would give the '
                         'page and the mask different tile grids). A missing '
                         'mask or a tile-count mismatch is a HARD FAILURE, '
                         'never a silent fallback. Part of the desc-cache key.')
    ap.add_argument('--token_l2', type=int, default=1,
                    help='1 (default) = L2-normalise patch tokens at extraction '
                         '(the base configuration). 0 = keep RAW token magnitudes '
                         '(the reference pipeline; affects codebook k-means, VLAD '
                         'assignment and residuals).')
    ap.add_argument('--vlad_pre_l2', type=int, default=1,
                    help='1 (default) = L2-normalise descriptors before codebook '
                         'k-means / VLAD assignment (the base configuration). '
                         '0 = raw space (the reference pipeline).')
    ap.add_argument('--vlad_residual_norm', type=int, default=1,
                    help='1 (default) = L2-normalise each residual before the '
                         'per-cluster sum (RN-VLAD, the base configuration). '
                         '0 = raw residual sums (Eq. 3 of the reference system).')
    ap.add_argument('--pca_exact', action='store_true',
                    help='Fit PCA-whitening with an EXACT economy SVD (sklearn '
                         'PCA(whiten=True) equivalence, the reference system\'s solver) instead of '
                         'the randomized torch.pca_lowrank. Post-descriptor: works '
                         'on cached descriptors, no re-embed.')
    ap.add_argument('--codebook_overlap', type=float, default=None,
                    help='Tile overlap for the auto-fit codebook source images '
                         '(default: the eval overlap). The reference system fits its codebook on '
                         'NON-overlapping stride-224 windows -> pass 0.0 to match.')
    ap.add_argument('--descriptor_cache', type=str, default=None,
                    help='Directory for cached page descriptors (pre-PCA). Re-runs that '
                         'only change PCA/ESVM/QE skip the hours of re-embedding. '
                         'Keyed on backbone mtime + embedding geometry (image size, '
                         'overlap, gate, aggregator, codebook source); a key '
                         'mismatch recomputes and overwrites. DELETE the directory '
                         'after changing embedding CODE - that is not detectable.')
    ap.add_argument('--qe_k', type=int, default=0,
                    help='Query expansion: average each descriptor with its k '
                         '(reciprocal) nearest neighbours before ranking (0 = off). '
                         'This is the reference system\'s kRNN; targets tail dispersion '
                         '(scattered same-writer pages). Applied after PCA/ESVM, '
                         'works on the cached descriptors (no re-embed).')
    ap.add_argument('--qe_alpha', type=float, default=1.0,
                    help='Weight on the neighbour sum in query expansion (default 1.0).')
    ap.add_argument('--qe_iters', type=int, default=1,
                    help='Query-expansion iterations (default 1; 2 spreads further, risks drift).')
    ap.add_argument('--qe_non_reciprocal', dest='qe_reciprocal', action='store_false',
                    help='Use plain (non-reciprocal) kNN for QE; default restricts to '
                         'mutual neighbours, which is more impostor-robust.')
    ap.set_defaults(qe_reciprocal=True)
    ap.add_argument('--use_patch_tokens', dest='use_patch_tokens', action='store_const', const=True,
                    default=None,
                    help='Aggregate over dense ViT patch tokens (~256 per tile) instead of '
                         'a single CLS per tile. Required for VLAD/SALAD to operate on the '
                         'descriptor counts they were designed for. Overrides aggregation.use_patch_tokens.')
    ap.add_argument('--no-use_patch_tokens', dest='use_patch_tokens', action='store_const', const=False,
                    help='Force CLS-only aggregation.')
    args = ap.parse_args()

    if args.esvm_neg_dir and args.pca_dim > 0 and not args.pca_fit_dir:
        args.pca_fit_dir = args.esvm_neg_dir
        print(f"[INFO] --pca_fit_dir defaulted to --esvm_neg_dir ({args.esvm_neg_dir}) "
              "so PCA-whitened queries and ESVM negatives share one space.")

    _bin_injected = False
    if args.binarisation != 'config' or args.simple_window:
        _base_cfg = OmegaConf.load(args.config)
        if args.binarisation != 'config':
            if OmegaConf.select(_base_cfg, 'dataset.binarisation') is None:
                _bin_injected = True
                OmegaConf.update(_base_cfg, 'dataset.binarisation',
                                 {'method': 'sauvola', 'window': 25, 'k': 0.2,
                                  'morph_open_kernel': 3}, force_add=True)
            OmegaConf.update(_base_cfg, 'dataset.binarisation.enable',
                             args.binarisation in ('filter', 'input'))
            if args.binarisation in ('filter', 'input'):
                OmegaConf.update(_base_cfg, 'dataset.binarisation.mode',
                                 args.binarisation, force_add=True)
        if args.simple_window:
            OmegaConf.update(_base_cfg, 'tiling.max_padding_pct', None,
                             force_add=True)
            OmegaConf.update(_base_cfg, 'tiling.min_stroke_pct', None,
                             force_add=True)
        _tag = (f"bin-{args.binarisation}"
                + ("_simplewin" if args.simple_window else ""))
        _patched = Path(args.output_csv).parent / (
            f"_eval_cfg_{Path(args.config).stem}_{_tag}.yaml"
        )
        _patched.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(_base_cfg, _patched)
        print(f"[INFO] eval overrides (binarisation '{args.binarisation}'"
              f"{', simple_window' if args.simple_window else ''}"
              f"): using patched config {_patched}")
        if args.simple_window:
            print("[eval] --simple_window: tile keep-filter OFF "
                  "(moving window + per-patch ink gate only)", flush=True)
        args.config = str(_patched)

    cfg = OmegaConf.load(args.config)
    from runtime_env import resolve_device
    device = resolve_device(cfg.training.get('device', 'cuda'), tag="[eval]")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    backbone = load_backbone(cfg, device, args.model_path)

    token_proj = None
    if str(getattr(args, 'distill_proj', 'none')) == 'auto':
        _ck = torch.load(args.model_path, map_location='cpu')
        _pw = _ck.get('distill_proj') if isinstance(_ck, dict) else None
        if _pw is None or 'weight' not in _pw:
            raise SystemExit('[distill_proj] --model_path carries no distill_proj '
                             'state (only distillation checkpoints do).')
        token_proj = torch.nn.Linear(_pw['weight'].shape[1], _pw['weight'].shape[0],
                                     bias='bias' in _pw)
        token_proj.load_state_dict(_pw)
        token_proj.to(device).eval()
        for _p in token_proj.parameters():
            _p.requires_grad_(False)
        print(f"[eval] distill_proj ACTIVE: raw patch tokens projected "
              f"{_pw['weight'].shape[1]} -> {_pw['weight'].shape[0]} (the "
              f"teacher-shaped subspace) before aggregation", flush=True)

    from inference_utils import configure_ink_foreground
    configure_ink_foreground(backbone, cfg)
    ink_mask_dir = getattr(args, 'ink_mask_dir', None)
    if ink_mask_dir:
        if not os.path.isdir(ink_mask_dir):
            raise SystemExit(f"!! --ink_mask_dir not a directory: {ink_mask_dir}")
        if not bool(args.simple_window):
            raise SystemExit(
                "!! --ink_mask_dir requires --simple_window (SIMPLE_WINDOW=1): "
                "a content-dependent tile keep-filter gives the page and the "
                "mask different tile grids.")
        ink_mask_dir = os.path.abspath(ink_mask_dir)
        print(f"[INK] gate from EXTERNAL masks: {ink_mask_dir} - token "
              f"SELECTION only; the backbone still reads the raw image.",
              flush=True)
    ink_k_eff = float(getattr(backbone, '_ink_k', 0.2))
    ink_window_eff = int(getattr(backbone, '_ink_window', 51))

    imgs = _collect_images(args.input_dir)
    labels = _load_labels(imgs, args.labels_csv, args.input_dir)

    tile_size = cfg.tiling.tile_size
    overlap = cfg.tiling.overlap
    if args.eval_overlap is not None and float(args.eval_overlap) >= 0.0:
        overlap = float(args.eval_overlap)
    batch_size = args.batch_size or cfg.training.batch_size
    agg_cfg = resolve_aggregation_config(getattr(cfg, 'aggregation', None))
    agg_method = args.method or agg_cfg['method']
    agg_method, _ = ensure_salad_pooling_available(
        agg_method,
        user_override=args.method is not None,
    )
    vlad_clusters = args.vlad_clusters if args.vlad_clusters is not None else agg_cfg['vlad_clusters']
    vlad_clusters = int(vlad_clusters) if vlad_clusters is not None else None
    metric = args.metric or 'cosine'
    image_size = cfg.dataset.image_size
    if args.image_size is not None:
        image_size = int(args.image_size)
    mean = None
    std = None

    ink_threshold = args.ink_threshold
    if ink_threshold is None:
        ink_threshold = float(getattr(cfg.tiling, 'binary_ink_threshold', 0.05) or 0.05)

    agg_kwargs, salad_cache = build_agg_kwargs(args, agg_cfg, agg_method, vlad_clusters)
    vlad_codebook = None

    if args.use_patch_tokens is not None:
        use_patch_tokens = bool(args.use_patch_tokens)
    else:
        use_patch_tokens = resolve_use_patch_tokens(
            agg_cfg.get('use_patch_tokens', 'auto'), agg_method
        )

    print(
        f"[CONFIG] method={agg_method}  use_patch_tokens={use_patch_tokens}  "
        f"intra_norm={not agg_kwargs.get('disable_intra_norm', False)}  "
        f"power_norm={agg_kwargs.get('vlad_power')}  "
        f"pca_dim={args.pca_dim}  pca_fit_dir={args.pca_fit_dir or '(eval set: transductive)'}  "
        f"eval_overlap={overlap}  "
        f"image_size={image_size}  "
        f"esvm={args.esvm}  esvm_neg_dir={args.esvm_neg_dir or '(eval set)'}  "
        f"binarisation={args.binarisation}  ink_threshold={ink_threshold}  "
        f"qe_k={args.qe_k} (alpha={args.qe_alpha}, reciprocal={args.qe_reciprocal})"
    )

    _max_image_size = int(getattr(cfg.dataset, 'max_image_size', 0) or 0) or None
    if args.max_image_size is not None:
        _max_image_size = int(args.max_image_size) or None

    _cache_dir = Path(args.descriptor_cache) if args.descriptor_cache else None
    if agg_method == 'salad':
        _agg_k = int(agg_kwargs.get('salad_support_points', 0) or 0)
    elif agg_method == 'vlad':
        _agg_k = int(vlad_clusters or 0)
    else:
        _agg_k = 0
    _bin_cfg = getattr(getattr(cfg, 'dataset', None), 'binarisation', None)
    import hashlib as _hl
    import json as _json
    _t = getattr(cfg, 'tiling', None)
    _n = getattr(getattr(cfg, 'model', None), 'input_norm', None)
    _fp_src = {
        'patch_size': getattr(_t, 'patch_size', None) if _t is not None else None,
        'min_stroke_pct': getattr(_t, 'min_stroke_pct', None) if _t is not None else None,
        'max_padding_pct': getattr(_t, 'max_padding_pct', None) if _t is not None else None,
        'bg_percentile': getattr(_t, 'bg_percentile', None) if _t is not None else None,
        'bg_margin': getattr(_t, 'bg_margin', None) if _t is not None else None,
        'canny': [getattr(_t, 'canny_low', None), getattr(_t, 'canny_high', None)] if _t is not None else None,
        'bin_method': getattr(_bin_cfg, 'method', None) if _bin_cfg is not None else None,
        'bin_morph': getattr(_bin_cfg, 'morph_open_kernel', None) if _bin_cfg is not None else None,
        'bin_after_resize': getattr(_bin_cfg, 'after_resize', None) if _bin_cfg is not None else None,
        'bin_invert': (False if _bin_cfg is not None and not _bin_injected
                       else None),
        'input_norm': ([list(getattr(_n, 'mean', []) or []),
                        list(getattr(_n, 'std', []) or [])] if _n is not None else None),
    }
    _cfg_fp = _hl.sha1(_json.dumps(_fp_src, sort_keys=True, default=str)
                       .encode()).hexdigest()[:12]
    desc_meta = {
        'cfg_fingerprint': _cfg_fp,
        'model_path': str(args.model_path),
        'model_mtime': (round(float(os.path.getmtime(args.model_path)), 3)
                        if os.path.exists(args.model_path) else 0.0),
        'image_size': int(image_size or 0),
        'max_image_size': int(_max_image_size or 0),
        'tile_size': int(tile_size),
        'overlap': float(overlap),
        'scales': '1',
        'use_patch_tokens': bool(use_patch_tokens),
        'agg': str(agg_method),
        'agg_k': int(_agg_k),
        'intra_norm': not bool(agg_kwargs.get('disable_intra_norm', False)),
        'power': (True if agg_kwargs.get('vlad_power') is None
                  else bool(agg_kwargs.get('vlad_power'))),
        'power_gamma': float(agg_kwargs.get('vlad_power_gamma') or 0.5),
        'binarisation': str(args.binarisation),
        'bin_enable': bool(getattr(_bin_cfg, 'enable', False)) if _bin_cfg is not None else False,
        'bin_mode': (str(getattr(_bin_cfg, 'mode', 'filter') or 'filter')
                     if _bin_cfg is not None else '-'),
        'bin_window': int(getattr(_bin_cfg, 'window', 51) or 51) if _bin_cfg is not None else 0,
        'bin_k': float(getattr(_bin_cfg, 'k', 0.2) or 0.2) if _bin_cfg is not None else 0.0,
        'ink_threshold': float(ink_threshold),
        'ink_max': None,
        'ink_method': 'sauvola',
        'ink_mask_dir': (str(ink_mask_dir) if ink_mask_dir else None),
        'ink_k': float(ink_k_eff),
        'ink_window': int(ink_window_eff),
        'simple_window': bool(args.simple_window),
        'distill_proj': bool(token_proj is not None),
        'grayscale_p': 0.0,
        'codebook': str(args.codebook_dir or args.input_dir),
        'codebook_samples': int(args.codebook_sample_images or 0),
        'codebook_max_tokens': int(args.codebook_max_tokens or 0),
        'codebook_max_per_image': int(args.codebook_max_per_image or 0),
        'codebook_algorithm': str(args.codebook_algorithm),
        'codebook_kmeans_n_init': int(args.codebook_kmeans_n_init),
        'seed': int(getattr(cfg.training, 'seed', 0) or 0),
        'max_tokens_per_page': int(args.max_tokens_per_page or 0),
        'foreground_weighting': False,
        'token_l2': bool(args.token_l2),
        'vlad_pre_l2': bool(args.vlad_pre_l2),
        'vlad_residual_norm': bool(args.vlad_residual_norm),
        'codebook_overlap': (float(args.codebook_overlap)
                             if args.codebook_overlap is not None else float(overlap)),
    }

    def _cache_file(tag: str) -> Path:
        token = 'patch' if use_patch_tokens else 'cls'
        return _cache_dir / f"{tag}_{agg_method}{_agg_k or ''}_{token}.pt"

    def _dir_tag(directory: str) -> str:
        import hashlib
        return 'dir_' + hashlib.sha1(
            str(Path(directory).resolve()).encode()
        ).hexdigest()[:10]

    _cached_eval = None
    _aux_cache: dict = {}
    if _cache_dir is not None:
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _cached_eval = _load_descriptor_cache(_cache_file('eval'), imgs, desc_meta)
        if _cached_eval is not None:
            print(f"[CACHE] Eval descriptors loaded ({_cached_eval[0].shape[0]} pages, "
                  f"{_cache_file('eval').name}) - embedding skipped; PCA/ESVM/QE "
                  "knobs remain free.")
        for _d in dict.fromkeys(d for d in (args.pca_fit_dir, args.esvm_neg_dir) if d):
            _hit = _load_descriptor_cache(
                _cache_file(_dir_tag(_d)), _collect_images(_d), desc_meta
            )
            if _hit is not None:
                _aux_cache[str(Path(_d).resolve())] = _hit[0]
                print(f"[CACHE] Aux descriptors for {_d} loaded from cache.")
    _needed_aux = [
        d for d, need in ((args.pca_fit_dir, int(args.pca_dim) > 0),
                          (args.esvm_neg_dir, bool(args.esvm)))
        if d and need
    ]
    _all_cached = (
        _cached_eval is not None
        and all(str(Path(d).resolve()) in _aux_cache for d in _needed_aux)
    )

    if agg_method in ('vlad', 'salad') and _all_cached:
        print("[CACHE] All descriptor sets cached - skipping shared codebook fitting.")
    if agg_method in ('vlad', 'salad') and not _all_cached:
        from vlad_codebook_utils import fit_codebook_from_image_paths
        k_centres = (
            int(vlad_clusters) if agg_method == 'vlad'
            else int(agg_kwargs['salad_support_points'])
        )
        seed = int(getattr(cfg.training, 'seed', 0) or 0)
        if args.codebook_dir:
            cb_imgs = _collect_images(args.codebook_dir)
            cb_src = args.codebook_dir
        else:
            cb_imgs = list(imgs)
            cb_src = args.input_dir
        if args.codebook_sample_images and len(cb_imgs) > int(args.codebook_sample_images):
            import random as _random
            _random.Random(seed).shuffle(cb_imgs)
            cb_imgs = cb_imgs[:int(args.codebook_sample_images)]
        auto_cfg = {
            'algorithm': str(args.codebook_algorithm),
            'kmeans_n_init': int(args.codebook_kmeans_n_init),
            'max_tokens': int(args.codebook_max_tokens) or None,
            'max_per_image': int(args.codebook_max_per_image) or None,
            'pre_l2_normalize': bool(args.vlad_pre_l2),
            'residual_normalization': bool(args.vlad_residual_norm),
            'seed': seed,
        }
        print(
            f"[INFO] Auto-fitting shared {agg_method.upper()} codebook (k={k_centres}) on "
            f"{len(cb_imgs)} page(s) from {cb_src}; use_patch_tokens={use_patch_tokens}, "
            f"labels=NONE …"
        )
        try:
            vlad_codebook = fit_codebook_from_image_paths(
                cb_imgs,
                backbone=backbone, device=device,
                batch_size=batch_size, config_path=args.config,
                vlad_clusters=k_centres, auto_cfg=auto_cfg, seed=seed,
                tile_image_kwargs={
                    'image_size': image_size,
                    'tile_size': tile_size,
                    'overlap': (float(args.codebook_overlap)
                                if args.codebook_overlap is not None else overlap),
                    'max_image_size': int(_max_image_size or 0),
                },
                progress_desc='codebook',
                use_patch_tokens=use_patch_tokens,
                ink_threshold=ink_threshold,
                ink_mask_dir=ink_mask_dir,
                patch_size=int(getattr(cfg.tiling, 'patch_size', 14) or 14),
                token_l2=bool(args.token_l2),
                token_proj=token_proj,
            )
        except Exception as exc:
            print(f"[WARN] Shared codebook fitting raised {exc!r}; falling back to the "
                  "degenerate per-page path.")
            vlad_codebook = None
        if vlad_codebook is not None:
            agg_kwargs['vlad_codebook'] = vlad_codebook
            if agg_method == 'vlad':
                agg_kwargs['vlad_clusters'] = max(int(vlad_clusters or 0), k_centres)
        else:
            print("[WARN] No shared codebook fitted; VLAD/SALAD will run the degenerate "
                  "per-page path and results WILL be poor.")

    _embed_workers = args.embed_workers
    if _embed_workers is None:
        _embed_workers = int(getattr(cfg.training, 'num_workers', 0) or 0)
    _embed_workers = max(0, int(_embed_workers))
    if _embed_workers > 0:
        print(f"[INFO] Parallel tiling with {_embed_workers} worker(s).")

    _embed_kwargs = dict(
        image_size=image_size, tile_size=tile_size,
        overlap=overlap, max_image_size=_max_image_size,
        mean=mean, std=std, config_path=args.config,
        embed_workers=_embed_workers,
        backbone=backbone, device=device, batch_size=batch_size,
        use_patch_tokens=use_patch_tokens,
        patch_size=int(getattr(cfg.tiling, 'patch_size', 14) or 14),
        ink_threshold=ink_threshold,
        ink_mask_dir=ink_mask_dir,
        agg_kwargs=agg_kwargs, salad_cache=salad_cache,
        max_tokens=int(args.max_tokens_per_page or 0),
        token_l2=bool(args.token_l2),
        token_proj=token_proj,
    )

    if _cached_eval is not None:
        E, valid_imgs, valid_labels = _cached_eval
    else:
        E, valid_imgs, valid_labels = _embed_collection(
            imgs, labels, desc="Embedding images", **_embed_kwargs,
        )
        if E is not None and _cache_dir is not None:
            _store_descriptor_cache(
                _cache_file('eval'), imgs, valid_imgs, valid_labels, E, desc_meta,
            )
    if E is None:
        print("No embeddings built; exiting.")
        return
    N = E.size(0)

    def _descriptors_for_dir(directory: str, what: str):
        key = str(Path(directory).resolve())
        if key in _aux_cache:
            return _aux_cache[key]
        aux_imgs = _collect_images(directory)
        if not aux_imgs:
            print(f"[WARN] {what}: no images found under {directory}; skipping.")
            _aux_cache[key] = None
            return None
        aux_E, aux_valid_imgs, aux_valid_labels = _embed_collection(
            aux_imgs, [0] * len(aux_imgs), desc=f"Embedding {what}", **_embed_kwargs,
        )
        if aux_E is None:
            print(f"[WARN] {what}: no descriptors could be built from {directory}.")
        elif _cache_dir is not None:
            _store_descriptor_cache(
                _cache_file(_dir_tag(directory)), aux_imgs,
                aux_valid_imgs, aux_valid_labels, aux_E, desc_meta,
            )
        _aux_cache[key] = aux_E
        return aux_E

    pca_stats = None
    pca_fit_label = '-'
    esvm_neg_label = '-'

    post_meta = {
        **desc_meta, 'stage': 'post',
        'pca_dim': int(args.pca_dim), 'pca_fit_dir': str(args.pca_fit_dir or ''),
        'pca_exact': bool(args.pca_exact),
        'esvm': bool(args.esvm), 'esvm_C': float(args.esvm_C),
        'esvm_num_neg': int(args.esvm_num_neg), 'esvm_neg_dir': str(args.esvm_neg_dir or ''),
    }
    post_file = (
        _cache_dir / f"post_{agg_method}{_agg_k or ''}_{'patch' if use_patch_tokens else 'cls'}.pt"
        if _cache_dir is not None else None
    )
    post_hit = False
    if post_file is not None and (args.pca_dim > 0 or args.esvm):
        hit = _load_descriptor_cache(post_file, valid_imgs, post_meta)
        if hit is not None and hit[0].shape[0] == len(valid_imgs):
            E = hit[0]
            post_hit = True
            if args.pca_dim > 0:
                pca_fit_label = str(args.pca_fit_dir) if args.pca_fit_dir else 'transductive'
            if args.esvm:
                esvm_neg_label = str(args.esvm_neg_dir) if args.esvm_neg_dir else 'eval-set'
            print(f"[CACHE] Post-PCA/ESVM descriptors loaded ({E.shape[0]} pages, "
                  f"{post_file.name}) - PCA+ESVM skipped; QE knobs remain free.")

    if not post_hit and args.pca_dim > 0:
        if args.pca_dim >= E.shape[1]:
            print(
                "[WARN] PCA target ≥ descriptor dim: this is FULL-RANK whitening "
                "(no compression). It re-weights the lowest-variance (noisiest) "
                "directions and frequently HURTS - most of all for low-dimensional "
                "CLS/GeM descriptors (e.g. 768-d). It also overlaps with ESVM (a "
                "per-exemplar whitening). Consider a smaller --pca_dim (256–512), "
                "or --pca_dim 0 to disable, when the descriptor is already compact."
            )
        E_fit = _descriptors_for_dir(args.pca_fit_dir, 'PCA fit set') if args.pca_fit_dir else None
        if E_fit is not None:
            pca_stats = _pca_whiten_fit(E_fit, args.pca_dim, device=device,
                                        exact=bool(args.pca_exact))
            k_eff = int(pca_stats['components'].shape[1])
            print(f"[INFO] PCA-whitening fitted on {pca_stats['n_fit']} held-out pages "
                  f"({args.pca_fit_dir}): {E.shape[1]} → {k_eff} dims (non-transductive"
                  f"{', exact SVD' if args.pca_exact else ''}).")
            E = _pca_whiten_apply(E, pca_stats, device=device)
            pca_fit_label = str(args.pca_fit_dir)
        else:
            if args.pca_fit_dir:
                print("[WARN] PCA fit set unusable; falling back to fitting on the "
                      "evaluation collection itself.")
            n_components = min(args.pca_dim, E.shape[0], E.shape[1])
            print(f"[INFO] PCA-whitening fitted on the EVALUATION collection "
                  f"(label-free but transductive): {E.shape[1]} → {n_components} dims. "
                  "Pass --pca_fit_dir (e.g. the validation set) for a protocol "
                  "directly comparable to published train-fit numbers.")
            E = _pca_whiten(E, n_components, device=device)
            pca_fit_label = 'transductive'

    if not post_hit and args.esvm:
        negatives = None
        if args.esvm_neg_dir:
            E_neg = _descriptors_for_dir(args.esvm_neg_dir, 'ESVM negative pool')
            if E_neg is not None:
                if pca_stats is not None:
                    E_neg = _pca_whiten_apply(E_neg, pca_stats, device=device)
                elif args.pca_dim > 0:
                    print("[WARN] PCA was fitted transductively on the evaluation set; "
                          "ESVM negatives cannot be projected into that space. "
                          "Ignoring --esvm_neg_dir for this run.")
                    E_neg = None
            negatives = E_neg
        if negatives is not None:
            esvm_neg_label = str(args.esvm_neg_dir)
            print(f"[INFO] ESVM normalisation against {negatives.shape[0]} writer-disjoint "
                  f"negatives from {args.esvm_neg_dir} "
                  f"(num_neg={args.esvm_num_neg}, C={args.esvm_C}, n_jobs={args.esvm_n_jobs})…")
        else:
            esvm_neg_label = 'eval-set'
            print(f"[INFO] ESVM normalisation with negatives sampled from the evaluation "
                  f"collection itself (num_neg={args.esvm_num_neg}, C={args.esvm_C}, "
                  f"n_jobs={args.esvm_n_jobs}) - note: a query's own same-writer pages "
                  "can land in its negative pool; prefer --esvm_neg_dir.")
        E = apply_esvm_normalization(
            E, num_negatives=args.esvm_num_neg, C=args.esvm_C, n_jobs=args.esvm_n_jobs,
            negatives=negatives,
        )

    if (not post_hit) and post_file is not None and (args.pca_dim > 0 or args.esvm):
        _store_descriptor_cache(post_file, valid_imgs, valid_imgs, valid_labels, E, post_meta)

    qe_label = '-'
    if args.qe_k and int(args.qe_k) > 0:
        print(f"[INFO] Query expansion (kRNN): k={args.qe_k}, alpha={args.qe_alpha}, "
              f"reciprocal={args.qe_reciprocal}, iters={args.qe_iters} - averaging each "
              "descriptor with its (reciprocal) nearest neighbours before ranking.")
        E = apply_query_expansion(
            E, int(args.qe_k), float(args.qe_alpha), device,
            reciprocal=bool(args.qe_reciprocal), iters=int(args.qe_iters),
        )
        qe_label = f"k{args.qe_k}a{args.qe_alpha:g}{'r' if args.qe_reciprocal else 'n'}"

    if metric == 'cosine':
        sims = _cosine_sim_matrix(E, device)
        sims.fill_diagonal_(-1e9)
        order = torch.argsort(sims, dim=1, descending=True)
    else:
        order_chunks = []
        bs = 512
        for i in range(0, N, bs):
            Qi = E[i:i + bs]
            a2 = (Qi * Qi).sum(1, keepdim=True)
            b2 = (E * E).sum(1).unsqueeze(0)
            d2 = a2 + b2 - 2.0 * (Qi @ E.t())
            for r in range(d2.size(0)):
                d2[r, i + r] = float('inf')
            order_chunks.append(torch.argsort(d2, dim=1, descending=False))
        order = torch.cat(order_chunks, 0)

    ks = (1, 3, 5)
    per_query_records: list = []
    topk_hits, ap_list, skipped, y_true, y_pred = _score_queries(
        order, valid_labels, ks, progress=True,
        per_query=per_query_records,
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    prefix = args.output_csv[:-4] if args.output_csv.lower().endswith('.csv') else args.output_csv
    _save_confusion(y_true, y_pred, prefix)

    per_query_path = f"{prefix}_per_query.csv"
    with open(per_query_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'query', 'writer', 'num_pos', 'ap', 'first_hit_rank',
            'top1', 'top1_writer', 'top1_hit',
            'top2', 'top2_writer', 'top2_hit',
            'top3', 'top3_writer', 'top3_hit',
            'top4', 'top4_writer', 'top4_hit',
            'top5', 'top5_writer', 'top5_hit',
        ])
        def _ap_key(rec):
            return 2.0 if rec['ap'] is None else rec['ap']
        for rec in sorted(per_query_records, key=_ap_key):
            row = [
                Path(valid_imgs[rec['idx']]).name, rec['label'], rec['num_pos'],
                ('' if rec['ap'] is None else f"{rec['ap']:.6f}"),
                ('' if rec['first_hit_rank'] is None else rec['first_hit_rank']),
            ]
            for j, hit in rec['top']:
                row.extend([Path(valid_imgs[j]).name, valid_labels[j], int(hit)])
            row.extend([''] * (20 - len(row)))
            w.writerow(row)
    print(f"Wrote per-query audit (worst-first) → {per_query_path}")

    outdir = Path(args.output_csv).parent
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        queries_total = len(valid_imgs)
        queries_used = queries_total - skipped
        map_value = (sum(ap_list) / queries_used) if queries_used > 0 else 0.0
        if agg_method == 'salad':
            agg_support_label = 'salad_support'
            agg_support_value = int(agg_kwargs.get('salad_support_points', 0))
        elif agg_method == 'vlad':
            agg_support_label = 'vlad_k'
            agg_support_value = int(vlad_clusters or 0)
        else:
            agg_support_label = 'agg_support'
            agg_support_value = '-'

        w.writerow([
            'model',
            'num_images', 'num_classes', 'queries_used', 'skipped_singleton_writers',
            'metric', 'agg', agg_support_label,
            'use_patch_tokens', 'intra_norm', 'power_norm', 'power_gamma',
            'pca_dim', 'pca_fit', 'scales', 'image_size', 'overlap',
            'max_image_size',
            'qe',
            'binarisation', 'fg_gate', 'simple_window', 'input_domain',
            'esvm', 'esvm_negatives',
            'token_l2', 'vlad_pre_l2', 'vlad_residual_norm',
            'codebook_dir', 'codebook_overlap', 'codebook_fit',
            'top1', 'top3', 'top5', 'mAP',
        ])
        topk_values = [
            (topk_hits[k] / queries_used) if queries_used > 0 else 0.0
            for k in ks
        ]
        binarisation_cfg = getattr(getattr(cfg, 'dataset', None), 'binarisation', None)
        binarisation_label = '-'
        if binarisation_cfg is not None and bool(getattr(binarisation_cfg, 'enable', False)):
            _bin_mode = str(getattr(binarisation_cfg, 'mode', 'filter') or 'filter')
            binarisation_label = f"{getattr(binarisation_cfg, 'method', 'sauvola')}/{_bin_mode}"

        if not use_patch_tokens:
            fg_gate_label = 'tile'
        elif bool(getattr(backbone, 'ink_foreground', False)):
            fg_gate_label = (f'ink-count@{ink_threshold:g}'
                             + ('/extmask' if ink_mask_dir else ''))
        else:
            fg_gate_label = 'edge-density'

        effective_intra_norm = (
            bool(agg_kwargs.get('vlad_intra_norm', True))
            and not bool(agg_kwargs.get('disable_intra_norm', False))
        )
        _cb_src_label = str(args.codebook_dir or args.input_dir or '-')
        w.writerow([
            str(args.model_path),
            queries_total, len(set(valid_labels)), queries_used, skipped,
            metric, agg_method, agg_support_value,
            use_patch_tokens,
            effective_intra_norm,
            agg_kwargs.get('vlad_power'),
            agg_kwargs.get('vlad_power_gamma'),
            args.pca_dim, pca_fit_label,
            '1', image_size, overlap,
            _max_image_size if _max_image_size is not None else 0,
            qe_label,
            binarisation_label, fg_gate_label, bool(args.simple_window),
            'color',
            args.esvm, esvm_neg_label,
            bool(args.token_l2), bool(args.vlad_pre_l2),
            bool(args.vlad_residual_norm),
            _cb_src_label,
            (args.codebook_overlap if args.codebook_overlap is not None else '-'),
            f"{args.codebook_algorithm}@{int(args.codebook_max_per_image or 0)}"
            f"/{int(args.codebook_max_tokens or 0)}",
            *topk_values,
            map_value,
        ])
    print(f"Wrote metrics → {args.output_csv}")
    print(
        f"[SUMMARY] queries_scored={queries_used}  skipped_singleton_writers={skipped}  "
        f"top1={topk_values[0]:.4f}  top3={topk_values[1]:.4f}  "
        f"top5={topk_values[2]:.4f}  mAP={map_value:.4f}"
    )


if __name__ == '__main__':
    main()
