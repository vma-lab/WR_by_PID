"""SALAD pooling using entropic Sinkhorn optimal transport."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    from geomloss import utils as geomloss_utils

    HAVE_GEOMLOSS = True
    GEOMLOSS_IMPORT_ERROR = None
except Exception as _geomloss_exc:
    geomloss_utils = None
    HAVE_GEOMLOSS = False
    GEOMLOSS_IMPORT_ERROR = repr(_geomloss_exc)

_LOGGER = logging.getLogger(__name__)
_WARNED_MISSING_GEOMLOSS = False


@dataclass
class SaladResult:
    """Container holding the outcome of SALAD pooling."""

    descriptor: torch.Tensor
    prototypes: torch.Tensor
    weights: torch.Tensor
    transport: Optional[torch.Tensor]
    sinkhorn_iterations: int
    converged: bool
    used_geomloss: bool


def _maybe_warn_no_geomloss() -> None:
    global _WARNED_MISSING_GEOMLOSS
    if HAVE_GEOMLOSS or _WARNED_MISSING_GEOMLOSS:
        return
    _LOGGER.info(
        "geomloss unavailable (%s); using the built-in matmul pairwise "
        "distances - equivalent speed, geomloss is not required.",
        GEOMLOSS_IMPORT_ERROR,
    )
    _WARNED_MISSING_GEOMLOSS = True


def _pairwise_squared(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if geomloss_utils is not None:
        return geomloss_utils.squared_distances(x.unsqueeze(0), y.unsqueeze(0))[0]
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1).unsqueeze(0)
    d = x2 + y2 - 2.0 * (x @ y.t())
    return d.clamp_min_(0.0)


def _initialise_prototypes(
    features: torch.Tensor,
    k: int,
    mode: str,
) -> torch.Tensor:
    if features.numel() == 0:
        raise ValueError("Cannot initialise prototypes from an empty feature set")
    if k < 1:
        raise ValueError("Number of prototypes must be at least one")

    N = features.shape[0]
    if N == 1:
        mode = "identity"
    mode = (mode or "kmeans++").lower()

    if mode == "identity":
        if N >= k:
            return features[:k].clone()
        pad = torch.zeros(k, features.shape[1], dtype=features.dtype, device=features.device)
        pad[:N] = features
        if N < k:
            pad[N:].copy_(features[-1:].expand(k - N, -1))
        return pad

    if mode == "random":
        if N >= k:
            idx = torch.randperm(N, device=features.device)[:k]
            return features[idx].clone()
        pad = torch.zeros(k, features.shape[1], dtype=features.dtype, device=features.device)
        idx = torch.randperm(N, device=features.device)
        pad[:N] = features[idx]
        pad[N:].copy_(features[idx[:1]].expand(k - N, -1))
        return pad

    if N >= k:
        chosen = [0]
        dists = _pairwise_squared(features, features[0:1]).squeeze(1)
        for _ in range(1, k):
            idx = torch.argmax(dists).item()
            chosen.append(idx)
            new_dists = _pairwise_squared(features, features[idx : idx + 1]).squeeze(1)
            dists = torch.minimum(dists, new_dists)
        return features[chosen].clone()

    pad = torch.zeros(k, features.shape[1], dtype=features.dtype, device=features.device)
    pad[:N] = features
    if N > 1:
        pad[N:] = features[1:2].expand(k - N, -1)
    else:
        pad[N:] = features[:1].expand(k - N, -1)
    return pad


def _sinkhorn(
    cost: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, bool, int]:
    if epsilon <= 0:
        raise ValueError("Sinkhorn regularisation epsilon must be positive")
    if max_iter <= 0:
        raise ValueError("Sinkhorn iterations must be a positive integer")

    a = torch.clamp(source_weights, min=1e-12)
    b = torch.clamp(target_weights, min=1e-12)
    log_a = torch.log(a)
    log_b = torch.log(b)

    K = -cost / epsilon
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    converged = False
    for it in range(max_iter):
        u_prev = u
        u = log_a - torch.logsumexp(K + v.view(1, -1), dim=1)
        v = log_b - torch.logsumexp(K + u.view(-1, 1), dim=0)
        if torch.max(torch.abs(u - u_prev)) < tol:
            converged = True
            break
    transport = torch.exp(u.view(-1, 1) + v.view(1, -1) + K)
    return transport, converged, it + 1


def salad_pooling(
    embeddings: torch.Tensor,
    *,
    num_support: int,
    sinkhorn_reg: float,
    sinkhorn_iters: int,
    sinkhorn_tol: float,
    update_iters: int,
    init_mode: str = "kmeans++",
    init_prototypes: Optional[torch.Tensor] = None,
    sort_by_mass: bool = True,
    dustbin: bool = False,
) -> SaladResult:
    """Compute SALAD pooling for a set of embeddings.

    When ``dustbin=True``, an extra "dustbin" prototype is added that
    absorbs non-informative features.  The dustbin prototype is excluded
    from the final descriptor, effectively filtering out background or
    uninformative patches at the aggregation level.  This follows the
    original SALAD paper (Izquierdo & Civera, CVPR 2024).
    """

    if embeddings.ndim != 2:
        raise ValueError("SALAD pooling expects embeddings with shape [N, D]")
    if num_support < 1:
        raise ValueError("num_support must be at least one")

    device = embeddings.device
    dtype = embeddings.dtype
    N, D = embeddings.shape

    if N == 0:
        prototypes = torch.zeros(num_support, D, device=device, dtype=dtype)
        weights = torch.zeros(num_support, device=device, dtype=dtype)
        descriptor = prototypes.reshape(-1)
        return SaladResult(
            descriptor=descriptor,
            prototypes=prototypes,
            weights=weights,
            transport=None,
            sinkhorn_iterations=0,
            converged=True,
            used_geomloss=HAVE_GEOMLOSS,
        )

    if not HAVE_GEOMLOSS:
        _maybe_warn_no_geomloss()

    solve_support = num_support + 1 if dustbin else num_support
    active_support = min(max(1, solve_support), N)

    if init_prototypes is not None:
        init = init_prototypes.to(device=device, dtype=dtype)
        if init.ndim != 2 or init.shape[1] != D:
            raise ValueError("init_prototypes must have shape [K, D]")
        if init.shape[0] < active_support:
            base = _initialise_prototypes(embeddings, active_support, init_mode)
            base[: init.shape[0]] = init
            prototypes = base
        else:
            prototypes = init[:active_support].clone()
    else:
        prototypes = _initialise_prototypes(embeddings, active_support, init_mode)

    source_weights = torch.full((N,), 1.0 / N, dtype=dtype, device=device)
    support_weights = torch.full((active_support,), 1.0 / active_support, dtype=dtype, device=device)

    transport = None
    converged = False
    total_sinkhorn_iters = 0

    prev_prototypes = prototypes.clone()
    for _ in range(max(1, update_iters)):
        cost = _pairwise_squared(prototypes, embeddings)
        transport, conv, n_iter = _sinkhorn(
            cost,
            support_weights,
            source_weights,
            epsilon=float(sinkhorn_reg),
            max_iter=int(sinkhorn_iters),
            tol=float(sinkhorn_tol),
        )
        converged = conv
        total_sinkhorn_iters += n_iter

        mass = transport.sum(dim=1)
        mass = torch.clamp(mass, min=1e-12)
        updated = transport @ embeddings
        updated = updated / mass.view(-1, 1)

        if sort_by_mass:
            order = torch.argsort(mass, descending=True)
            mass = mass[order]
            updated = updated[order]

        prototypes = updated
        support_weights = mass / mass.sum()

        if torch.allclose(prototypes, prev_prototypes, atol=1e-4, rtol=1e-4):
            break
        prev_prototypes = prototypes.clone()

    if dustbin and active_support > 1:
        prototypes = prototypes[:num_support]
        mass = mass[:num_support]

    if prototypes.shape[0] < num_support:
        pad_count = num_support - prototypes.shape[0]
        pad = torch.zeros(pad_count, D, dtype=dtype, device=device)
        mass_pad = torch.zeros(pad_count, dtype=dtype, device=device)
        prototypes = torch.cat([prototypes, pad], dim=0)
        mass = torch.cat([mass, mass_pad], dim=0)

    descriptor = prototypes.reshape(-1)
    return SaladResult(
        descriptor=descriptor,
        prototypes=prototypes,
        weights=mass,
        transport=transport,
        sinkhorn_iterations=total_sinkhorn_iters,
        converged=converged,
        used_geomloss=HAVE_GEOMLOSS,
    )
