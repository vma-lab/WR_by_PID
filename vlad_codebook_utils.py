"""Helper utilities for fitting VLAD codebooks from tile embeddings."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from inference_utils import (build_ink_mask_index, embed_tiles,
                             external_ink_scores, tile_image)


class ReservoirSampler:
    """Reservoir sampler for fixed-size random subsets of embeddings."""

    def __init__(self, capacity: Optional[int], rng: random.Random):
        self.capacity = capacity if capacity and capacity > 0 else None
        self.rng = rng
        self._data: List[np.ndarray] = []
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self._data)

    def is_full(self) -> bool:
        return self.capacity is not None and len(self._data) >= self.capacity

    def add_batch(self, rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.float32)
        for row in rows:
            self.add_row(row)

    def add_row(self, row: np.ndarray):
        row = np.asarray(row, dtype=np.float32)
        self.total_seen += 1
        if self.capacity is None:
            self._data.append(row)
            return

        if len(self._data) < self.capacity:
            self._data.append(row)
            return

        j = self.rng.randint(0, self.total_seen - 1)
        if j < self.capacity:
            self._data[j] = row

    def as_array(self, shuffle: bool = True) -> np.ndarray:
        if shuffle and self._data:
            self.rng.shuffle(self._data)
        if not self._data:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self._data, axis=0).astype(np.float32)


def collect_tile_embeddings(
    image_paths: Iterable[Path],
    *,
    sampler: ReservoirSampler,
    rng: random.Random,
    backbone,
    device,
    batch_size: int,
    tile_image_kwargs: Optional[dict] = None,
    max_per_image: Optional[int] = None,
    stop_when_full: bool = False,
    use_patch_tokens: bool = False,
    ink_threshold: Optional[float] = None,
    ink_mask_dir: Optional[str] = None,
    patch_size: int = 16,
    token_l2: bool = True,
    token_proj=None,
) -> int:
    """Embed tiles for ``image_paths`` and push them through ``sampler``.

    Parameters
    ----------
    use_patch_tokens : bool
        Forwarded to :func:`embed_tiles`.  When True the codebook is fitted
        on dense ViT patch tokens (~256 per tile) rather than CLS tokens -
        match this to whatever the inference path will use, otherwise the
        codebook centres land in a different descriptor space than the
        query/gallery vectors and VLAD assignments become noise.
    ink_threshold : float, optional
        Per-patch ink-fraction threshold for the ink-count gate, matching the
        evaluation-time gate.  Only effective in patch-token mode when the
        backbone emits ink scores (binarisation filter mode).

    Returns
    -------
    int
        Number of images that contributed at least one embedding.
    """

    kept_images = 0
    tile_kwargs = dict(tile_image_kwargs or {})
    mask_index = build_ink_mask_index(ink_mask_dir) if ink_mask_dir else None

    for img_path in image_paths:
        tiles, masks = tile_image(str(img_path), **tile_kwargs)
        if tiles is None or tiles.numel() == 0:
            continue
        ext_ink = None
        if mask_index is not None:
            mp = mask_index.get(Path(str(img_path)).stem.lower())
            if mp is None:
                raise RuntimeError(
                    f"[ink-mask] no mask for {Path(str(img_path)).name} in "
                    f"{ink_mask_dir} (codebook pass)")
            ext_ink = external_ink_scores(
                mp, patch_size=patch_size, tile_kwargs=tile_kwargs)
            if ext_ink is None or ext_ink.shape[0] != tiles.shape[0]:
                raise RuntimeError(
                    f"[ink-mask] tile-grid mismatch for "
                    f"{Path(str(img_path)).name} in the codebook pass: page "
                    f"{tiles.shape[0]} tiles, mask "
                    f"{0 if ext_ink is None else ext_ink.shape[0]}")

        embeddings, _, ink_scores = embed_tiles(
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

        if embeddings.numel() == 0:
            continue

        if use_patch_tokens:
            keep = None
            if (ink_threshold is not None
                    and not (ink_scores is not None
                             and ink_scores.numel() == embeddings.shape[0])):
                raise RuntimeError(
                    f"[codebook] ink gate requested (ink_threshold="
                    f"{ink_threshold}) but per-patch ink scores are "
                    f"{'missing' if ink_scores is None else 'length-mismatched'}"
                    f" - refusing the silent edge-mask fallback")
            if (
                ink_threshold is not None
                and ink_scores is not None
                and ink_scores.numel() == embeddings.shape[0]
            ):
                keep = ink_scores.reshape(-1) >= float(ink_threshold)
            elif masks is not None and masks.numel() == embeddings.shape[0]:
                keep = masks.reshape(-1).to(dtype=torch.bool)
            if keep is not None and bool(keep.any()):
                embeddings = embeddings[keep]

        arr = embeddings.cpu().numpy().astype(np.float32)

        if max_per_image is not None and arr.shape[0] > max_per_image:
            indices = list(range(arr.shape[0]))
            rng.shuffle(indices)
            arr = arr[indices[: max_per_image]]

        sampler.add_batch(arr)
        kept_images += 1

        if stop_when_full and sampler.is_full():
            break

    return kept_images


def _normalise_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return arr / norms


def prepare_embeddings_matrix(
    sampler: ReservoirSampler,
    *,
    pre_l2: bool,
) -> np.ndarray:
    """Prepare the sampled embedding matrix for the k-means fit."""

    matrix = sampler.as_array(shuffle=True)
    if matrix.size == 0:
        raise RuntimeError("No tile embeddings were collected; check the dataset and tiling settings")

    if pre_l2:
        matrix = _normalise_rows(matrix)

    return matrix.astype(np.float32)


def fit_kmeans(
    data: np.ndarray,
    *,
    k: int,
    algorithm: str,
    rng: random.Random,
    kmeans_max_iter: int,
    kmeans_n_init: int,
    minibatch_batch_size: int,
    minibatch_reassignment_ratio: float,
):
    """Fit a clustering model on ``data`` for VLAD centroids."""

    if data.shape[0] < k:
        raise RuntimeError(
            f"Collected {data.shape[0]} embeddings but k={k}. Increase the dataset or lower the cluster count."
        )

    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=k,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=rng.randint(0, 2**31 - 1),
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=k,
            max_iter=kmeans_max_iter,
            batch_size=minibatch_batch_size,
            reassignment_ratio=minibatch_reassignment_ratio,
            random_state=rng.randint(0, 2**31 - 1),
        )

    model.fit(data)
    return model


def build_vlad_payload(
    centroids: np.ndarray,
    *,
    pre_l2_normalize: bool,
    residual_normalization: bool,
    intra_normalization: bool,
    power_normalization: bool,
    power_gamma: float,
    power_eps: float,
    final_normalize: bool,
) -> dict:
    """Assemble a VLAD codebook payload compatible with ``aggregate_embeddings``."""

    payload = {
        "centroids": torch.from_numpy(np.asarray(centroids, dtype=np.float32)),
        "pre_l2_normalize": bool(pre_l2_normalize),
        "residual_normalization": bool(residual_normalization),
        "intra_normalization": bool(intra_normalization),
        "power_normalization": bool(power_normalization),
        "power_normalization_gamma": float(power_gamma),
        "power_normalization_eps": float(power_eps),
        "final_normalize": bool(final_normalize),
    }

    return payload


def fit_codebook_from_image_paths(
    image_paths: Iterable[Path],
    *,
    backbone,
    device,
    batch_size: int,
    config_path: str,
    vlad_clusters: int,
    auto_cfg: Optional[dict] = None,
    seed: Optional[int] = None,
    tile_image_kwargs: Optional[dict] = None,
    progress_desc: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = print,
    use_patch_tokens: bool = False,
    ink_threshold: Optional[float] = None,
    ink_mask_dir: Optional[str] = None,
    patch_size: int = 16,
    token_l2: bool = True,
    token_proj=None,
):
    if vlad_clusters is None or vlad_clusters < 1:
        return None

    auto = dict(auto_cfg or {})

    algorithm = str(auto.get('algorithm', 'minibatch')).lower()
    if algorithm not in {'kmeans', 'minibatch'}:
        algorithm = 'minibatch'

    seed_val = auto.get('seed', seed)
    if seed_val is None:
        seed_val = 0
    rng = random.Random(seed_val)

    sampler = ReservoirSampler(auto.get('max_tokens'), rng)

    paths = list(image_paths)
    if not paths:
        return None

    iterator: Iterable[Path] = paths
    if progress_desc:
        from tqdm import tqdm

        iterator = tqdm(paths, desc=progress_desc, leave=False)

    tile_kwargs = dict(tile_image_kwargs or {})
    tile_kwargs.setdefault('config_path', config_path)

    kept_images = collect_tile_embeddings(
        iterator,
        sampler=sampler,
        rng=rng,
        backbone=backbone,
        device=device,
        batch_size=batch_size,
        tile_image_kwargs=tile_kwargs,
        max_per_image=auto.get('max_per_image'),
        stop_when_full=bool(auto.get('stop_when_full', False)),
        use_patch_tokens=use_patch_tokens,
        ink_threshold=ink_threshold,
        ink_mask_dir=ink_mask_dir,
        patch_size=patch_size,
        token_l2=token_l2,
        token_proj=token_proj,
    )

    log = log_fn if log_fn is not None else (lambda *_: None)

    if len(sampler) < vlad_clusters:
        log(
            f"[auto-codebook] Only collected {len(sampler)} embeddings < k={vlad_clusters}; "
            "falling back to per-page aggregation."
        )
        return None

    try:
        matrix = prepare_embeddings_matrix(
            sampler,
            pre_l2=bool(auto.get('pre_l2_normalize', True)),
        )
    except RuntimeError as exc:
        log(f"[auto-codebook] Failed to prepare embeddings for codebook: {exc}")
        return None

    try:
        model = fit_kmeans(
            matrix,
            k=vlad_clusters,
            algorithm=algorithm,
            rng=rng,
            kmeans_max_iter=int(auto.get('kmeans_max_iter', 300)),
            kmeans_n_init=int(auto.get('kmeans_n_init', 10)),
            minibatch_batch_size=int(auto.get('minibatch_batch_size', 8192)),
            minibatch_reassignment_ratio=float(auto.get('minibatch_reassignment_ratio', 0.01)),
        )
    except RuntimeError as exc:
        log(f"[auto-codebook] KMeans failed: {exc}")
        return None

    payload = build_vlad_payload(
        model.cluster_centers_.astype(np.float32),
        pre_l2_normalize=bool(auto.get('pre_l2_normalize', True)),
        residual_normalization=bool(auto.get('residual_normalization', True)),
        intra_normalization=bool(auto.get('intra_normalization', True)),
        power_normalization=bool(auto.get('power_normalization', True)),
        power_gamma=float(auto.get('power_gamma', 0.5)),
        power_eps=float(auto.get('power_eps', 1e-12)),
        final_normalize=bool(auto.get('final_normalize', True)),
    )

    log(
        f"[auto-codebook] Fitted shared k-means codebook with k={vlad_clusters} using "
        f"{len(sampler)} embeddings from {kept_images} images."
    )
    return payload

