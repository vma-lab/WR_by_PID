import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from data_loader import CV2Loader, canonical_image_size
from omegaconf import DictConfig, OmegaConf
from unfold_transform import UnfoldTransform, compute_patch_mask

logger = logging.getLogger(__name__)

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

def resolve_aggregation_config(cfg: Any) -> Dict[str, Any]:
    """Return aggregation defaults combined with configuration overrides."""

    defaults = {
        'method': 'salad',
        'vlad_clusters': 64,
        'use_patch_tokens': 'auto',
        'intra_norm': True,
        'power_norm': True,
        'power_gamma': 0.5,
        'salad': {
            'support_points': 16,
            'sinkhorn_reg': 0.05,
            'sinkhorn_iters': 100,
            'sinkhorn_tolerance': 1e-6,
            'update_iters': 5,
            'init': 'kmeans++',
            'cache': True,
            'sort_by_mass': True,
        },
    }

    if cfg is None:
        return {k: (v.copy() if isinstance(v, dict) else v) for k, v in defaults.items()}

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(cfg, dict):
        return {k: (v.copy() if isinstance(v, dict) else v) for k, v in defaults.items()}

    resolved: Dict[str, Any] = {
        'method': cfg.get('method', defaults['method']) or defaults['method'],
        'vlad_clusters': int(cfg.get('vlad_clusters', defaults['vlad_clusters']) or defaults['vlad_clusters']),
        'use_patch_tokens': cfg.get('use_patch_tokens', defaults['use_patch_tokens']),
        'intra_norm': bool(cfg.get('intra_norm', defaults['intra_norm'])),
        'power_norm': bool(cfg.get('power_norm', defaults['power_norm'])),
        'power_gamma': float(cfg.get('power_gamma', defaults['power_gamma'])),
        'salad': defaults['salad'].copy(),
    }

    salad_cfg = cfg.get('salad')
    if isinstance(salad_cfg, dict):
        for key, default_value in defaults['salad'].items():
            value = salad_cfg.get(key, default_value)
            if value is not None:
                resolved['salad'][key] = value

    return resolved


def resolve_use_patch_tokens(setting: Any, method: str) -> bool:
    """Resolve the ``use_patch_tokens`` setting from the aggregation config.

    'auto' (the default) maps to True for the dense aggregators (vlad/salad)
    and False for the per-tile aggregators (mean/max/gem) where one vector
    per tile is the correct input.
    """

    if isinstance(setting, bool):
        return setting
    if isinstance(setting, str):
        text = setting.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        if text == "auto":
            return method in {"vlad", "salad"}
    if setting is None:
        return method in {"vlad", "salad"}
    raise ValueError(
        f"Invalid use_patch_tokens setting: {setting!r}. "
        "Expected True, False, or 'auto'."
    )


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SaladAvailability:
    """Describe whether SALAD pooling can run and if acceleration is enabled."""

    method: str
    downgraded: bool
    salad_enabled: bool
    accelerated: bool
    message: str


def resolve_salad_pooling_status(
    method: str,
    *,
    user_override: bool,
    fallback: str = 'vlad',
) -> SaladAvailability:
    """Return the runtime SALAD status along with a user-facing summary."""

    if method != 'salad':
        message = (
            f"[SALAD] Using '{method}' pooling; SALAD descriptors are disabled."
        )
        return SaladAvailability(
            method=method,
            downgraded=False,
            salad_enabled=False,
            accelerated=False,
            message=message,
        )

    try:
        from pooling import salad as salad_mod
    except Exception:
        message = (
            f"[SALAD] Optional dependencies failed to import; falling back to "
            f"'{fallback}' pooling. Install the 'salad' extra to re-enable SALAD."
        )
        if user_override:
            message = (
                "[SALAD] Optional dependencies failed to import; continuing with "
                "SALAD using the PyTorch Sinkhorn fallback (expect slower OT solves)."
            )
            _LOGGER.warning(message)
            return SaladAvailability(
                method='salad',
                downgraded=False,
                salad_enabled=True,
                accelerated=False,
                message=message,
            )

        _LOGGER.warning(message)
        return SaladAvailability(
            method=fallback,
            downgraded=True,
            salad_enabled=False,
            accelerated=False,
            message=message,
        )

    have_geomloss = bool(getattr(salad_mod, 'HAVE_GEOMLOSS', False))
    if have_geomloss:
        message = "[SALAD] SALAD pooling enabled with geomloss/pykeops acceleration."
        return SaladAvailability(
            method='salad',
            downgraded=False,
            salad_enabled=True,
            accelerated=True,
            message=message,
        )

    _gl_err = getattr(salad_mod, 'GEOMLOSS_IMPORT_ERROR', None)
    if user_override:
        message = (
            "[SALAD] geomloss unavailable"
            + (f" (import error: {_gl_err})" if _gl_err else "")
            + "; SALAD runs on the built-in matmul/Sinkhorn kernels "
            "(equivalent speed - geomloss is not required)."
        )
        _LOGGER.warning(message)
        return SaladAvailability(
            method='salad',
            downgraded=False,
            salad_enabled=True,
            accelerated=False,
            message=message,
        )

    message = (
        "[SALAD] geomloss unavailable"
        + (f" (import error: {_gl_err})" if _gl_err else "")
        + f"; falling back to '{fallback}' "
        "pooling for the config-default path (pass --method salad / "
        "AGGREGATOR=salad to keep SALAD - the built-in kernels are "
        "equivalent in speed)."
    )
    _LOGGER.warning(message)
    return SaladAvailability(
        method=fallback,
        downgraded=True,
        salad_enabled=False,
        accelerated=False,
        message=message,
    )


def ensure_salad_pooling_available(
    method: str,
    *,
    user_override: bool,
    fallback: str = 'vlad',
) -> Tuple[str, bool]:
    """Downgrade SALAD pooling to a safer default when dependencies are missing."""

    status = resolve_salad_pooling_status(
        method,
        user_override=user_override,
        fallback=fallback,
    )
    return status.method, status.downgraded


def resolve_input_norm(cfg: Any) -> Tuple[Sequence[float], Sequence[float]]:
    """Return ``(mean, std)`` honouring an optional binarisation-aware swap.

    Default path: ``model.input_norm.{mean,std}`` (ImageNet stats).  When ``dataset.binarisation.enable`` is true we
    swap to grayscale-appropriate stats so the binarised ``{0, 1}`` input
    centres at zero with unit range - applying ImageNet's per-channel
    offsets to a high-contrast binary image leaves the post-norm
    distribution badly skewed and asymmetric.

    The binarisation block may override the swap explicitly via
    ``dataset.binarisation.input_norm.{mean,std}``; ``"auto"`` (or
    missing) falls back to ``[0.5,0.5,0.5] / [0.5,0.5,0.5]``.
    """

    default_mean = list(cfg.model.input_norm.mean)
    default_std = list(cfg.model.input_norm.std)

    bin_cfg = getattr(getattr(cfg, "dataset", None), "binarisation", None)
    if bin_cfg is None or not bool(getattr(bin_cfg, "enable", False)):
        return default_mean, default_std

    mode = str(getattr(bin_cfg, "mode", "filter") or "filter").lower()
    if mode != "input":
        return default_mean, default_std

    override = getattr(bin_cfg, "input_norm", None)
    if override is not None:
        override_mean = getattr(override, "mean", None)
        override_std = getattr(override, "std", None)
        if (override_mean not in (None, "auto")
                and override_std not in (None, "auto")):
            return list(override_mean), list(override_std)

    return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def build_ink_mask_index(mask_dir):
    """{lowercase stem -> path} for an external foreground-mask directory.

    Raises on a duplicate stem: an ambiguous mask must never be resolved by
    chance, because the mask selects WHICH patches aggregate and the wrong one
    would silently gate the wrong tokens.
    """
    from collections import defaultdict
    from pathlib import Path as _P
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    idx = defaultdict(list)
    for p in sorted(_P(mask_dir).rglob('*')):
        if p.is_file() and p.suffix.lower() in exts:
            idx[p.stem.lower()].append(p)
    dupes = {k: v for k, v in idx.items() if len(v) > 1}
    if dupes:
        first = list(dupes.items())[:5]
        raise RuntimeError(
            f"[ink-mask] ambiguous stems in {mask_dir}: "
            + "; ".join(f"{k} -> {[str(x) for x in v]}" for k, v in first))
    return {k: v[0] for k, v in idx.items()}


def external_ink_scores(mask_path, *, patch_size, tile_kwargs, chunk=256):
    """Per-patch foreground fractions for ONE page, read from an EXTERNAL mask
    image and tiled on the SAME grid as the page.

    The mask is put through the SAME ``tile_image`` call as the photograph -
    identical loader, geometry, scale and stride - so the tile grids cannot
    drift.  ``mean``/``std`` are forced to
    identity so the tiles come back as raw [0,1] values and no de-normalisation
    guesswork is involved.  The mask is never binarised, so the selector is
    the same mask for every model, whatever the row's input binarisation.

    Foreground is taken to be the MINORITY class of the page, so either mask
    polarity works.  Returns ``[n_tiles, (tile/patch)**2]`` or ``None``.
    """
    tk = dict(tile_kwargs)
    tk['mean'] = [0.0, 0.0, 0.0]
    tk['std'] = [1.0, 1.0, 1.0]
    tk['binarise'] = False
    mt, _ = tile_image(str(mask_path), **tk)
    if mt is None or mt.numel() == 0:
        return None
    ph = max(1, int(patch_size))
    ink_is_bright = float(mt.mean()) <= 0.5
    outs = []
    for i in range(0, mt.shape[0], int(chunk)):
        blk = mt[i:i + int(chunk)]
        g = blk.to(torch.float32).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        fg = (g >= 0.5) if ink_is_bright else (g < 0.5)
        outs.append(F.avg_pool2d(fg.to(torch.float32), (ph, ph),
                                 stride=(ph, ph)).reshape(blk.shape[0], -1))
        del blk, g, fg
    del mt
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def configure_ink_foreground(backbone, cfg) -> bool:
    """Turn on the backbone's on-device per-patch ink scoring whenever
    binarisation is enabled - both ``mode: filter`` (the loaded colour image
    reaches the backbone; a Sauvola ink
    mask, with an optional opening per config, drives the ink-count gate) and
    ``mode: input`` (the backbone sees the binary image; the same per-patch
    ink-count gate then selects foreground tokens, matching the reference
    system's t_fg on binary input).  No-op on backbones without the hook (non-ViTWrapper) or when
    binarisation is off.  Returns whether ink scoring is now on.
    """
    if not hasattr(backbone, "ink_foreground"):
        return False
    bin_cfg = getattr(getattr(cfg, "dataset", None), "binarisation", None)
    enable = bool(getattr(bin_cfg, "enable", False)) if bin_cfg is not None else False
    mode = (str(getattr(bin_cfg, "mode", "filter") or "filter").lower()
            if bin_cfg is not None else "filter")
    on = bool(enable and mode in ("filter", "input"))
    backbone.ink_foreground = on
    if on:
        mean, std = resolve_input_norm(cfg)
        backbone._input_mean = list(mean)
        backbone._input_std = list(std)
        backbone._ink_window = int(getattr(bin_cfg, "window", 51) or 51)
        backbone._ink_k = float(getattr(bin_cfg, "k", 0.2))
        backbone._ink_morph_open = int(getattr(bin_cfg, "morph_open_kernel", 3) or 0)
        _sees = "binary" if mode == "input" else "colour"
        print(f"[INK] {mode} mode ON: per-patch ink gate from "
              f"sauvola "
              f"(window={backbone._ink_window}, k={backbone._ink_k}, "
              f"morph_open={backbone._ink_morph_open}); backbone sees {_sees}.",
              flush=True)
    return on


def build_agg_kwargs(
    args: Any,
    agg_cfg: Mapping[str, Any],
    method: str,
    vlad_clusters: Optional[int],
    vlad_codebook: Any = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Build the ``agg_kwargs`` dict consumed by ``aggregate_embeddings``.

    Resolves the VLAD/SALAD parameters for the evaluation.  Argparse values take
    precedence; falls back to ``agg_cfg['salad']`` defaults when the CLI
    flag is ``None``.  Returns ``(agg_kwargs, salad_cache)`` where
    ``salad_cache`` is a fresh dict when caching is enabled for SALAD,
    otherwise ``None``.
    """

    salad_defaults = agg_cfg['salad']
    salad_support = int(args.salad_support_points or salad_defaults['support_points'])
    salad_reg = float(args.salad_reg if args.salad_reg is not None else salad_defaults['sinkhorn_reg'])
    salad_iters = int(args.salad_sinkhorn_iters if args.salad_sinkhorn_iters is not None else salad_defaults['sinkhorn_iters'])
    salad_tol = float(args.salad_sinkhorn_tol if args.salad_sinkhorn_tol is not None else salad_defaults['sinkhorn_tolerance'])
    salad_updates = int(args.salad_update_iters if args.salad_update_iters is not None else salad_defaults['update_iters'])
    salad_init = args.salad_init or str(salad_defaults['init'])
    salad_cache_enabled = (
        bool(salad_defaults.get('cache', True)) if args.salad_cache is None else bool(args.salad_cache)
    )
    salad_sort_by_mass = (
        bool(salad_defaults.get('sort_by_mass', True))
        if args.salad_sort_by_mass is None
        else bool(args.salad_sort_by_mass)
    )
    salad_dustbin = (
        bool(salad_defaults.get('dustbin', False)) if args.salad_dustbin is None else bool(args.salad_dustbin)
    )

    if method == 'vlad' and (vlad_clusters is None or vlad_clusters < 1):
        raise ValueError('--vlad_clusters must be a positive integer when using VLAD pooling')
    if method == 'salad' and salad_support < 1:
        raise ValueError('--salad_support_points must be a positive integer when using SALAD pooling')

    salad_cache = {} if (salad_cache_enabled and method == 'salad') else None

    cfg_intra_norm = bool(agg_cfg.get('intra_norm', True))
    cfg_power_norm = bool(agg_cfg.get('power_norm', True))
    cfg_power_gamma = float(agg_cfg.get('power_gamma', 0.5))

    effective_power = args.vlad_power if args.vlad_power is not None else cfg_power_norm
    effective_power_gamma = (
        args.vlad_power_gamma if args.vlad_power_gamma is not None else cfg_power_gamma
    )

    agg_kwargs: Dict[str, Any] = {
        'method': method,
        'vlad_clusters': vlad_clusters,
        'vlad_codebook': vlad_codebook,
        'vlad_power': effective_power,
        'vlad_power_gamma': effective_power_gamma,
        'vlad_power_eps': args.vlad_power_eps,
        'vlad_intra_norm': cfg_intra_norm,
        'disable_intra_norm': args.disable_intra_norm,
        'disable_final_norm': args.disable_final_norm,
        'salad_support_points': salad_support,
        'salad_sinkhorn_reg': salad_reg,
        'salad_sinkhorn_iters': salad_iters,
        'salad_sinkhorn_tolerance': salad_tol,
        'salad_update_iters': salad_updates,
        'salad_init': salad_init,
        'salad_sort_by_mass': salad_sort_by_mass,
        'salad_dustbin': salad_dustbin,
        'salad_intra_norm': cfg_intra_norm,
        'salad_power_norm': cfg_power_norm,
        'salad_power_gamma': cfg_power_gamma,
    }
    if salad_cache is not None:
        agg_kwargs['salad_cache'] = salad_cache
    return agg_kwargs, salad_cache


def resolve_image_size(
    image_path: str,
    image_size: int,
    tile_size: int,
    overlap: float,
    max_image_size: Optional[int] = None,
) -> int:
    """Resolve the ``image_size`` to pass to ``tile_image`` for one page.

    ``-1`` (native-raw) and ``> 0`` are returned as they are.  With
    ``image_size == 0`` (native), peeks at the page's native dimensions and
    returns the canonical native canvas (re-canonicalised so the tile grid
    lands exactly), or 0 when the page cannot be read or is smaller than a
    tile.
    """

    if image_size is not None and int(image_size) == -1:
        return -1

    if image_size and int(image_size) > 0:
        return int(image_size)

    import cv2 as _cv2_local

    img = _cv2_local.imread(str(image_path), _cv2_local.IMREAD_COLOR)
    if img is None or img.shape[0] < int(tile_size) or img.shape[1] < int(tile_size):
        return 0
    long_side = int(max(img.shape[:2]))
    max_cap = int(max_image_size or 0) or None
    native_canonical = canonical_image_size(
        long_side, tile_size=int(tile_size), overlap=float(overlap), max_size=max_cap,
    )
    return canonical_image_size(
        native_canonical, tile_size=int(tile_size), overlap=float(overlap), max_size=max_cap,
    )


def tile_image(
        image_path: str,
        image_size: int = None,
        tile_size: int = None,
        mean=None,
        std=None,
        overlap: float = 0.0,
        config_path: str = None,
        min_stroke_pct: float = None,
        max_image_size: int = None,
        binarise: bool = True,
):
    """
    Load an image → TILE FIRST (so padding-based pruning uses correct coordinates) → normalize per tile.

    ``config_path`` (the evaluation YAML) is required.  If any of image_size,
    tile_size, mean, or std is None, read them from the config.
    Uses model.input_norm.{mean,std} and tiling.{tile_size, overlap, max_padding_pct, min_stroke_pct}.

    ``binarise=False`` loads the image without the config's input
    binarisation (used for external token-selector masks); geometry and the
    tile grid are unchanged.
    """
    import torch
    from torchvision import transforms
    from omegaconf import OmegaConf

    if config_path is None:
        raise ValueError("tile_image: config_path (the evaluation YAML) is required")
    cfg = OmegaConf.load(config_path)
    if image_size is None:
        image_size = int(cfg.dataset.image_size)
    if tile_size is None:
        tile_size = int(cfg.tiling.tile_size)
    if max_image_size is None:
        max_image_size = int(getattr(cfg.dataset, "max_image_size", 0) or 0)
    if mean is None or std is None:
        mean, std = resolve_input_norm(cfg)
    max_pad = getattr(cfg.tiling, "max_padding_pct", None)
    if min_stroke_pct is None:
        min_stroke_pct = getattr(cfg.tiling, "min_stroke_pct", None)

    bg_percentile = float(getattr(cfg.tiling, "bg_percentile", 95))
    bg_margin = float(getattr(cfg.tiling, "bg_margin", 10))
    canny_low = int(getattr(cfg.tiling, "canny_low", 100))
    canny_high = int(getattr(cfg.tiling, "canny_high", 200))

    patch_size = getattr(cfg.tiling, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(cfg.model, "patch_size", None)
    if patch_size is None:
        patch_size = tile_size
    patch_size = max(1, int(patch_size))

    binarisation_cfg = getattr(getattr(cfg, 'dataset', None), 'binarisation', None)
    if binarisation_cfg is not None:
        binarisation_dict = OmegaConf.to_container(binarisation_cfg, resolve=True)
    else:
        binarisation_dict = None
    loader = CV2Loader(
        image_size=image_size,
        min_size=tile_size,
        tile_overlap=overlap,
        binarisation=(binarisation_dict if binarise else None),
        max_image_size=max_image_size,
    )
    pil_img = loader(image_path)

    binary_mode = bool(binarisation_dict and binarisation_dict.get('enable', False))
    binary_ink_threshold = float(getattr(cfg.tiling, 'binary_ink_threshold', 0.05))

    tensor = transforms.ToTensor()(pil_img)

    unfold = UnfoldTransform(
        tile_size=tile_size,
        overlap=overlap,
        max_padding_pct=max_pad,
        min_stroke_pct=min_stroke_pct,
        patch_size=patch_size,
        bg_percentile=bg_percentile,
        bg_margin=bg_margin,
        canny_low=canny_low,
        canny_high=canny_high,
        binary_mode=binary_mode,
        binary_ink_threshold=binary_ink_threshold,
    )
    tiles = unfold(tensor)
    mask_max_pad = max_pad
    mask_min_stroke = min_stroke_pct

    if tiles.size(0) == 0:
        relaxed = 0.75 if (max_pad is None) else max(max_pad, 0.75)
        tiles = UnfoldTransform(
            tile_size=tile_size,
            overlap=overlap,
            max_padding_pct=relaxed,
            min_stroke_pct=min_stroke_pct,
            patch_size=patch_size,
            bg_percentile=bg_percentile,
            bg_margin=bg_margin,
            canny_low=canny_low,
            canny_high=canny_high,
            binary_mode=binary_mode,
            binary_ink_threshold=binary_ink_threshold,
        )(tensor)
        mask_max_pad = relaxed
    if tiles.size(0) == 0:
        tiles = UnfoldTransform(
            tile_size=tile_size,
            overlap=overlap,
            max_padding_pct=None,
            min_stroke_pct=None,
            patch_size=patch_size,
        )(tensor)
        mask_max_pad = None
        mask_min_stroke = None
    if tiles.size(0) == 0:
        C, H, W = tensor.shape
        ci = max((H - tile_size) // 2, 0)
        cj = max((W - tile_size) // 2, 0)
        tiles = tensor[:, ci:ci + tile_size, cj:cj + tile_size].unsqueeze(0)
        mask_max_pad = None
        mask_min_stroke = None


    def _compute_masks(tile_batch: torch.Tensor) -> torch.Tensor:
        if tile_batch.numel() == 0:
            return torch.empty((0, 1, 1), dtype=torch.bool, device=tile_batch.device)

        masks = []
        for tile in tile_batch:
            eff_patch = max(1, min(patch_size, tile.shape[-2], tile.shape[-1]))
            try:
                mask = compute_patch_mask(
                    tile,
                    eff_patch,
                    mask_max_pad,
                    mask_min_stroke,
                    bg_percentile=bg_percentile,
                    bg_margin=bg_margin,
                    canny_low=canny_low,
                    canny_high=canny_high,
                    binary_mode=binary_mode,
                    binary_ink_threshold=binary_ink_threshold,
                )
            except ValueError:
                mask = torch.ones((1, 1), dtype=torch.bool, device=tile.device)
            masks.append(mask)
        return torch.stack(masks, dim=0)

    masks = _compute_masks(tiles)

    if tiles.numel() > 0:
        mean_t = torch.tensor(mean, dtype=tiles.dtype, device=tiles.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=tiles.dtype, device=tiles.device).view(1, 3, 1, 1)
        tiles = (tiles - mean_t) / std_t

    return tiles, masks


def _split_backbone_outputs(
    outputs: Any,
    *,
    prefer_patch_tokens: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Separate tile features and optional global descriptors from backbone output.

    When ``prefer_patch_tokens`` is True and the backbone returns a
    ``(cls_out, patches)`` tuple (the ``return_patch_tokens=True`` branch of
    the ViT wrappers), the ``[B, N_patches, D]`` patch tensor is treated as the
    feature output and the CLS as the global descriptor.  Otherwise the
    behaviour matches the historical CLS-only path.
    """

    global_descriptor = None
    features = outputs

    if isinstance(outputs, dict):
        feature_candidates = (
            'tile_embeddings',
            'embeddings',
            'patch_embeddings',
            'patch_tokens',
            'tokens',
            'last_hidden_state',
            'features',
        )
        for key in feature_candidates:
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                features = value
                break
        else:
            for value in outputs.values():
                if isinstance(value, torch.Tensor):
                    features = value
                    break

        descriptor_candidates = (
            'global_descriptor',
            'cls_token',
            'cls',
            'pooled',
            'pooled_output',
        )
        for key in descriptor_candidates:
            value = outputs.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                global_descriptor = value
                break
    elif isinstance(outputs, (tuple, list)) and outputs:
        if (
            prefer_patch_tokens
            and len(outputs) >= 2
            and isinstance(outputs[0], torch.Tensor)
            and isinstance(outputs[1], torch.Tensor)
            and outputs[1].ndim == 3
        ):
            global_descriptor = outputs[0]
            features = outputs[1]
        else:
            features = outputs[0]
            for candidate in outputs[1:]:
                if isinstance(candidate, torch.Tensor) and candidate.ndim >= 1:
                    global_descriptor = candidate
                    break

    return features, global_descriptor


def embed_tiles(
        tiles: torch.Tensor,
        backbone: torch.nn.Module,
        device: torch.device,
        batch_size: int = 32,
        amp: Optional[bool] = None,
        *,
        use_patch_tokens: bool = False,
        token_l2: bool = True,
        token_proj: torch.nn.Module = None,
):
    """
    Embed a batch of tiles through the backbone and L2-normalise the features.

    Parameters
    ----------
    amp : Optional[bool]
        Enable ``torch.amp.autocast`` around the backbone forward
        passes (float16 on CUDA).  Defaults to ``True`` on CUDA and
        ``False`` on CPU/MPS where autocast adds no speedup.
    use_patch_tokens : bool, default False
        When True the backbone is asked for dense patch tokens
        (``return_patch_tokens=True``) and the returned embeddings are the
        L2-normalised patch tokens flattened to ``[N_tiles * N_patches, D]``.
        This is the configuration that VLAD / SALAD aggregation is
        actually designed for; with the CLS path instead, aggregators see
        only ~1 vector per tile and degrade to noisy clustering of a tiny
        descriptor set.

    Returns
    -------
    embeddings : torch.Tensor
        Per-element embeddings.  Shape ``[N_tiles, D]`` for the legacy
        CLS path; ``[N_tiles * N_patches, D]`` when ``use_patch_tokens``
        is True.
    global_descriptors : Optional[torch.Tensor]
        Optional global descriptors (e.g., CLS tokens) aligned with the
        original tile batch (always ``[N_tiles, D]``).  When unavailable
        ``None`` is returned.
    ink_scores : Optional[torch.Tensor]
        Per-patch ink fractions aligned with ``embeddings`` in patch-token
        mode (the ink-count gate), ``None`` when the gate is off.
    """
    backbone.eval()

    if amp is None:
        amp = device.type == 'cuda'
    autocast_ctx = (
        torch.amp.autocast(device.type) if amp else contextlib.nullcontext()
    )

    embeddings = []
    global_batches = []
    ink_score_batches = []
    has_global_descriptor = True

    with torch.no_grad():
        for i in range(0, tiles.size(0), batch_size):
            batch = tiles[i:i + batch_size].to(device)

            with autocast_ctx:
                fwd_kwargs = {}
                if use_patch_tokens:
                    fwd_kwargs["return_patch_tokens"] = True
                try:
                    outputs = backbone(batch, **fwd_kwargs)
                except TypeError:
                    fwd_kwargs.pop("return_patch_tokens", None)
                    outputs = backbone(batch, **fwd_kwargs)

                if use_patch_tokens:
                    ink_pp = getattr(backbone, "_ink_fraction_per_patch", None)
                    if ink_pp is not None:
                        ink_score_batches.append(ink_pp.float().cpu())

                feats, global_desc = _split_backbone_outputs(
                    outputs, prefer_patch_tokens=use_patch_tokens
                )
                if not isinstance(feats, torch.Tensor):
                    raise TypeError("Backbone must return tensor features")

                if use_patch_tokens and feats.ndim == 3:
                    if token_proj is not None:
                        feats = token_proj(feats.float())
                    embs = F.normalize(feats, dim=-1) if token_l2 else feats
                    embs = embs.reshape(-1, embs.shape[-1])
                    if isinstance(global_desc, torch.Tensor):
                        if global_desc.ndim == 1:
                            global_desc = global_desc.view(1, -1)
                        if global_desc.ndim == 2:
                            if token_proj is not None:
                                global_desc = token_proj(global_desc.float())
                            global_desc = F.normalize(global_desc, dim=1)
                        else:
                            global_desc = None
                else:
                    embs = F.normalize(feats, dim=-1)
                    if isinstance(global_desc, torch.Tensor):
                        if global_desc.ndim == 1:
                            global_desc = global_desc.view(1, -1)
                        if global_desc.ndim == 2:
                            global_desc = F.normalize(global_desc, dim=1)
                        else:
                            global_desc = None

            embeddings.append(embs.float().cpu())

            if not isinstance(global_desc, torch.Tensor):
                has_global_descriptor = False
                global_batches.append(None)
            else:
                global_batches.append(global_desc.float().cpu())

    embed_tensor = torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0))

    global_descriptor = None
    if has_global_descriptor and global_batches and all(batch is not None for batch in global_batches):
        global_descriptor = torch.cat(global_batches, dim=0)

    ink_scores = None
    if ink_score_batches:
        ink_scores = torch.cat(ink_score_batches, dim=0)
        if ink_scores.ndim == 2:
            ink_scores = ink_scores.reshape(-1)

    return embed_tensor, global_descriptor, ink_scores


def gem_pooling(embeddings: torch.Tensor, p: float = 3.0, eps: float = 1e-6):
    """Generalized-Mean (GeM) pooling over tile embeddings.

    Uses the signed power formulation so that negative embedding
    dimensions (common with L2-normalised features) are preserved
    instead of being clamped to zero.
    """
    abs_emb = embeddings.abs().clamp(min=eps)
    powered = embeddings.sign() * abs_emb.pow(p)
    mean_pow = powered.mean(dim=0)
    return mean_pow.sign() * mean_pow.abs().clamp(min=eps).pow(1.0 / p)


def _coerce_array(value: Any, *, dtype=None) -> torch.Tensor:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def aggregate_embeddings(
        embs: torch.Tensor,
        method: str = 'vlad',
        vlad_clusters: int = 64,
        mask: torch.Tensor = None,
        vlad_codebook: Any = None,
        *,
        vlad_power: Optional[bool] = None,
        vlad_power_gamma: Optional[float] = None,
        vlad_power_eps: Optional[float] = None,
        vlad_intra_norm: Optional[bool] = None,
        disable_intra_norm: bool = False,
        disable_final_norm: bool = False,
        fallback: Optional[torch.Tensor] = None,
        fallback_mode: str = 'auto',
        fallback_pool: str = 'gem',
        salad_support_points: int = 16,
        salad_sinkhorn_reg: float = 0.05,
        salad_sinkhorn_iters: int = 100,
        salad_sinkhorn_tolerance: float = 1e-6,
        salad_update_iters: int = 5,
        salad_init: str = 'kmeans++',
        salad_cache: Optional[Dict[Any, torch.Tensor]] = None,
        salad_cache_key: Optional[Any] = None,
        salad_sort_by_mass: bool = True,
        salad_dustbin: bool = False,
        salad_intra_norm: bool = True,
        salad_power_norm: bool = True,
        salad_power_gamma: float = 0.5,
        salad_power_eps: float = 1e-12,
):
    """Aggregate tile embeddings into a single vector and L2-normalize the result.

    Parameters
    ----------
    embs:
        Tensor of shape ``[N, D]`` containing the per-tile (or per-patch) embeddings.
    method:
        Pooling strategy, one of ``{"mean", "max", "gem", "vlad", "salad"}``.
    vlad_clusters:
        Maximum number of clusters for VLAD pooling.
    mask:
        Optional tensor indicating which embeddings should contribute to the
        aggregation.  Any additional spatial dimensions are reduced via
        ``any()`` so both per-token ``[N, H, W]`` and simple ``[N]`` masks are
        supported.  Entries with ``mask == 0`` are zeroed before pooling and are
        excluded from the aggregation statistics.
    vlad_codebook:
        Optional tensor or mapping describing the fitted VLAD codebook.
        When a mapping is supplied it must contain a ``centroids`` entry with
        shape ``[K, D]`` and may include normalisation flags.  Assignments are computed via a
        nearest-centre lookup instead of fitting KMeans online.
    vlad_power:
        Optional boolean override controlling signed power normalisation on
        VLAD residual buckets.  When ``None`` the value stored in the codebook
        (if any) is honoured.
    vlad_power_gamma:
        Optional exponent used for signed power normalisation.  Falls back to
        the codebook value or ``0.5`` when unspecified.
    vlad_power_eps:
        Optional epsilon guarding the signed power transform.  Defaults to the
        codebook value or ``1e-12`` when unspecified.
    disable_intra_norm:
        Disable per-cluster L2 (intra) normalisation regardless of the
        codebook settings.
    disable_final_norm:
        Skip the final global L2 normalisation step.
    fallback:
        Optional descriptor or batch of descriptors to use when all tiles are
        filtered out.
    fallback_mode:
        Strategy controlling how the fallback descriptor is used.  ``"auto"``
        prefers the provided descriptor when available and otherwise pools the
        pre-filter embeddings.  ``"provided"`` requires a descriptor, while
        ``"prefilter"`` forces pooling over the pre-filter embeddings and
        ``"disable"`` retains the legacy zero-vector behaviour.
    fallback_pool:
        Pooling operation applied to 2D fallback descriptors.  One of
        ``{"gem", "mean", "max"}``.
    salad_support_points:
        Number of support prototypes used in SALAD pooling.
    salad_sinkhorn_reg:
        Entropic regularisation strength (epsilon) for the Sinkhorn solver.
    salad_sinkhorn_iters:
        Maximum number of Sinkhorn iterations per refinement step.
    salad_sinkhorn_tolerance:
        Convergence tolerance for the Sinkhorn updates.
    salad_update_iters:
        Number of prototype refinement steps (outer loop iterations).
    salad_init:
        Prototype initialisation strategy.  One of ``{"kmeans++", "random", "identity"}``.
    salad_cache:
        Optional mutable mapping used to cache OT prototypes between calls.
    salad_cache_key:
        Identifier for the cache entry corresponding to the current embedding
        batch.  When provided alongside ``salad_cache`` the cached prototypes
        are used as warm-starts and refreshed after pooling.
    salad_sort_by_mass:
        Sort prototypes in descending order of transported mass before
        flattening.  Helps keep cache entries stable across iterations.
    """

    if embs.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, D], got {embs.shape}")

    method = (method or 'vlad').lower()

    device = embs.device
    logger = logging.getLogger(__name__)
    N, D = embs.shape

    def _prepare_mask(raw_mask: torch.Tensor, target: int) -> torch.BoolTensor:
        if raw_mask is None:
            return torch.ones(target, dtype=torch.bool, device=device)

        if isinstance(raw_mask, torch.Tensor):
            mask_t = raw_mask.to(device=device)
        else:
            mask_t = torch.as_tensor(raw_mask, device=device)

        if mask_t.numel() == 0:
            return torch.zeros(target, dtype=torch.bool, device=device)

        if mask_t.ndim == 0:
            mask_t = mask_t.view(1)

        if mask_t.shape[0] != target:
            if mask_t.shape[0] == 1:
                expand_shape = (target,) + tuple(mask_t.shape[1:])
                mask_t = mask_t.expand(expand_shape)
            elif mask_t.numel() == target:
                mask_t = mask_t.reshape(target)
            else:
                raise ValueError(
                    f"Mask first dimension {mask_t.shape[0]} does not match embeddings {target}"
                )

        if mask_t.ndim > 1:
            mask_t = mask_t.reshape(mask_t.shape[0], -1).any(dim=1)

        if mask_t.dtype != torch.bool:
            mask_t = mask_t != 0

        return mask_t.to(device=device)

    keep_mask = _prepare_mask(mask, N)

    prefilter_embs = embs.clone()

    mask_float = keep_mask.to(dtype=embs.dtype)
    embs = embs * mask_float.unsqueeze(1)

    valid_embs = embs[keep_mask]

    codebook_clusters = 0
    codebook_meta: Dict[str, Any] = {}
    centers = None
    if method in ('vlad', 'salad') and vlad_codebook is not None:
        if isinstance(vlad_codebook, dict):
            if 'centroids' not in vlad_codebook:
                raise ValueError("VLAD codebook dictionary must include 'centroids'")
            centers = torch.as_tensor(
                vlad_codebook['centroids'], device=device, dtype=embs.dtype
            )
            codebook_meta = {
                key: value for key, value in vlad_codebook.items() if key != 'centroids'
            }
        elif isinstance(vlad_codebook, torch.Tensor):
            centers = vlad_codebook.to(device=device, dtype=embs.dtype)
        else:
            centers = torch.as_tensor(vlad_codebook, device=device, dtype=embs.dtype)
        if centers.ndim != 2:
            raise ValueError(
                f"VLAD codebook must have shape [K, D], got {tuple(centers.shape)}"
            )
        codebook_clusters = centers.shape[0]

    def _apply_preprocess(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if codebook_meta.get('pre_l2_normalize', False):
            x = F.normalize(x, dim=1)
        return x

    processed_embs = valid_embs
    residual_dim = D
    if method == 'vlad' or (method == 'salad' and centers is not None):
        processed_embs = _apply_preprocess(valid_embs)
        if centers is not None:
            residual_dim = centers.shape[1]
            if processed_embs.numel() > 0 and processed_embs.shape[1] != centers.shape[1]:
                raise ValueError(
                    "Processed embeddings do not match VLAD centroid dimensionality"
                )
        elif processed_embs.numel() > 0:
            residual_dim = processed_embs.shape[1]

    fallback_tensor = None
    if fallback is not None:
        fallback_tensor = _coerce_array(fallback, dtype=embs.dtype)
        if fallback_tensor is not None:
            fallback_tensor = fallback_tensor.to(device=device)

    def _pool_fallback(batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return batch
        if batch.ndim != 2:
            raise ValueError("Fallback pooling expects a 2D tensor")
        mode = fallback_pool.lower()
        if mode == 'gem':
            return gem_pooling(batch)
        if mode == 'mean':
            return batch.mean(dim=0)
        if mode == 'max':
            return batch.max(dim=0).values
        raise ValueError(f"Unknown fallback_pool '{fallback_pool}'")

    fallback_mode = (fallback_mode or 'auto').lower()
    if fallback_mode not in {'auto', 'prefilter', 'provided', 'disable'}:
        raise ValueError("fallback_mode must be one of {'auto', 'prefilter', 'provided', 'disable'}")

    def _prepare_prefilter() -> torch.Tensor:
        source = prefilter_embs.to(device=device)
        if method == 'vlad':
            return _apply_preprocess(source)
        return source

    salad_points: Optional[int] = None

    def _format_salad_vector(vec: torch.Tensor) -> torch.Tensor:
        if salad_points is None:
            raise RuntimeError('SALAD fallback formatting requested for non-SALAD method')
        if vec.ndim != 1:
            raise ValueError("SALAD fallback descriptor must be a 1D tensor")
        if vec.numel() == salad_points * D:
            return vec.clone()
        if vec.numel() == D:
            buf = torch.zeros(salad_points, D, dtype=embs.dtype, device=device)
            buf[0] = vec
            return buf.reshape(-1)
        raise ValueError(
            "Fallback descriptor dimensionality does not match SALAD support × embedding size"
        )

    if method == 'salad':
        salad_points = int(salad_support_points)
        if centers is not None:
            salad_points = int(centers.shape[0])
        if salad_points < 1:
            raise ValueError('salad_support_points must be a positive integer when using SALAD')

    if valid_embs.numel() == 0:
        fallback_vec = None
        fallback_source = None

        if fallback_mode in {'auto', 'provided'} and fallback_tensor is not None:
            if fallback_tensor.ndim == 1:
                fallback_vec = fallback_tensor.view(-1)
            elif fallback_tensor.ndim == 2:
                fallback_vec = _pool_fallback(fallback_tensor)
            else:
                raise ValueError("Fallback descriptor must be 1D or 2D tensor")
            if fallback_vec is not None and fallback_vec.numel() > 0:
                fallback_source = 'provided'
            else:
                fallback_vec = None

        if fallback_vec is None and fallback_mode in {'auto', 'prefilter'}:
            prefilter = _prepare_prefilter()
            if prefilter.numel() > 0:
                fallback_vec = _pool_fallback(prefilter)
                fallback_source = 'prefilter'

        if fallback_vec is None or fallback_vec.numel() == 0:
            if method == 'vlad':
                clusters = vlad_clusters
                if centers is not None:
                    clusters = max(clusters or 0, codebook_clusters)
                if clusters is None or clusters < 1:
                    clusters = codebook_clusters or 64
                if clusters < 1:
                    raise ValueError("vlad_clusters must be positive when using VLAD pooling")
                clusters = int(clusters)
                vec = torch.zeros(clusters * residual_dim, dtype=embs.dtype, device=device)
            elif method == 'salad':
                vec = torch.zeros(salad_points * D, dtype=embs.dtype, device=device)
            else:
                vec = torch.zeros(D, dtype=embs.dtype, device=device)
        else:
            if method == 'vlad':
                clusters = vlad_clusters
                if centers is not None:
                    clusters = max(clusters or 0, codebook_clusters)
                if clusters is None or clusters < 1:
                    clusters = codebook_clusters or 64
                if clusters < 1:
                    raise ValueError("vlad_clusters must be positive when using VLAD pooling")
                clusters = int(clusters)

                final_dim = clusters * residual_dim
                if fallback_vec.numel() == final_dim:
                    buf = fallback_vec.view(clusters, residual_dim).clone()
                else:
                    if fallback_vec.numel() != residual_dim:
                        raise ValueError(
                            "Fallback descriptor dimensionality does not match residual dimension"
                        )
                    buf = torch.zeros(clusters, residual_dim, dtype=embs.dtype, device=device)
                    buf[0] = fallback_vec

                do_power = vlad_power if vlad_power is not None else codebook_meta.get('power_normalization', False)
                if do_power and buf.numel() > 0:
                    gamma = float(vlad_power_gamma if vlad_power_gamma is not None else codebook_meta.get('power_normalization_gamma', 0.5))
                    eps = float(vlad_power_eps if vlad_power_eps is not None else codebook_meta.get('power_normalization_eps', 1e-12))
                    buf = torch.sign(buf) * torch.pow(buf.abs() + eps, gamma)

                if vlad_intra_norm is not None:
                    intra_default = bool(vlad_intra_norm)
                else:
                    intra_default = bool(codebook_meta.get('intra_normalization', False))
                do_intra = intra_default and not disable_intra_norm
                if do_intra and buf.numel() > 0:
                    norms = buf.norm(dim=1, keepdim=True)
                    eps = 1e-12
                    buf = torch.where(norms > eps, buf / torch.clamp(norms, min=eps), buf)

                vec = buf.reshape(-1)
            elif method == 'salad':
                vec = _format_salad_vector(fallback_vec)
            else:
                if fallback_vec.numel() != D:
                    raise ValueError("Fallback descriptor dimensionality does not match embedding dimension")
                vec = fallback_vec

            if fallback_source is not None:
                logger.info("No valid embeddings; using %s fallback via %s pooling", fallback_source, fallback_pool)
    elif method == 'mean':
        vec = valid_embs.mean(dim=0)
    elif method == 'max':
        vec = valid_embs.max(dim=0).values
    elif method == 'gem':
        vec = gem_pooling(valid_embs)
    elif method == 'salad' and centers is not None and processed_embs.numel() > 0:
        from pooling.salad import _sinkhorn

        ot_device = device
        if device.type != 'cuda' and torch.cuda.is_available():
            ot_device = torch.device('cuda')
        xg = processed_embs.to(device=ot_device, dtype=torch.float32)
        cg = centers.to(device=ot_device, dtype=torch.float32)
        n_support, n_pts = cg.shape[0], xg.shape[0]
        c_sq = (cg * cg).sum(dim=1, keepdim=True)
        x_sq = (xg * xg).sum(dim=1)
        cost = (c_sq + x_sq.unsqueeze(0) - 2.0 * (cg @ xg.t())).clamp_min_(0.0)
        src_w = torch.full((n_support,), 1.0 / n_support, dtype=cost.dtype, device=ot_device)
        tgt_w = torch.full((n_pts,), 1.0 / max(n_pts, 1), dtype=cost.dtype, device=ot_device)
        transport, _, _ = _sinkhorn(
            cost, src_w, tgt_w,
            epsilon=float(salad_sinkhorn_reg),
            max_iter=int(salad_sinkhorn_iters),
            tol=float(salad_sinkhorn_tolerance),
        )
        mass = transport.sum(dim=1, keepdim=True)
        buf = (transport @ xg - mass * cg).to(device=device, dtype=embs.dtype)
        if salad_power_norm and buf.numel() > 0:
            gamma = float(salad_power_gamma)
            eps_p = float(salad_power_eps)
            buf = torch.sign(buf) * torch.pow(buf.abs() + eps_p, gamma)
        if salad_intra_norm and buf.numel() > 0 and not disable_intra_norm:
            buf = F.normalize(buf, dim=1)
        vec = buf.reshape(-1)
    elif method == 'salad':
        from pooling.salad import salad_pooling

        if salad_points is None:
            salad_points = int(salad_support_points)
            if salad_points < 1:
                raise ValueError('salad_support_points must be a positive integer when using SALAD')

        init_proto: Optional[torch.Tensor] = None
        if salad_cache is not None and salad_cache_key is not None:
            cached = salad_cache.get(salad_cache_key)
            if cached is not None:
                tensor = torch.as_tensor(cached, device=device, dtype=processed_embs.dtype)
                if tensor.ndim == 2 and tensor.shape[1] == processed_embs.shape[1]:
                    init_proto = tensor

        result = salad_pooling(
            processed_embs,
            num_support=salad_points,
            sinkhorn_reg=float(salad_sinkhorn_reg),
            sinkhorn_iters=int(salad_sinkhorn_iters),
            sinkhorn_tol=float(salad_sinkhorn_tolerance),
            update_iters=int(salad_update_iters),
            init_mode=(salad_init or 'kmeans++'),
            init_prototypes=init_proto,
            sort_by_mass=bool(salad_sort_by_mass),
            dustbin=bool(salad_dustbin),
        )
        proto_matrix = result.prototypes
        if salad_intra_norm and proto_matrix.numel() > 0 and not disable_intra_norm:
            proto_matrix = F.normalize(proto_matrix, dim=1)
        if salad_power_norm and proto_matrix.numel() > 0:
            gamma = float(salad_power_gamma)
            eps_p = float(salad_power_eps)
            proto_matrix = torch.sign(proto_matrix) * torch.pow(
                proto_matrix.abs() + eps_p, gamma
            )
        vec = proto_matrix.reshape(-1)

        if salad_cache is not None and salad_cache_key is not None:
            salad_cache[salad_cache_key] = result.prototypes.detach().cpu()

    elif method == 'vlad':
        clusters = vlad_clusters
        if centers is not None:
            clusters = max(clusters or 0, codebook_clusters)
        if clusters is None or clusters < 1:
            clusters = codebook_clusters or 64
        if clusters < 1:
            raise ValueError("vlad_clusters must be positive when using VLAD pooling")
        if centers is not None and clusters < codebook_clusters:
            raise ValueError(
                "vlad_clusters must be >= number of centres in vlad_codebook"
            )

        vlad_clusters = int(clusters)

        buf = torch.zeros(vlad_clusters, residual_dim, dtype=embs.dtype, device=device)

        n_samples = processed_embs.size(0)

        if n_samples == 0:
            pass
        elif centers is not None:
            if codebook_clusters == 0:
                pass
            else:
                assignments = torch.cdist(processed_embs, centers).argmin(dim=1)
                for c in range(codebook_clusters):
                    assign_mask = assignments == c
                    if assign_mask.any():
                        residuals = processed_embs[assign_mask] - centers[c]
                        if codebook_meta.get('residual_normalization', False):
                            residuals = F.normalize(residuals, dim=1)
                        buf[c] = residuals.sum(dim=0)
                    else:
                        buf[c].zero_()
        elif n_samples < vlad_clusters:
            buf[:n_samples] = processed_embs
        elif vlad_clusters < 2:
            buf[0] = processed_embs.mean(dim=0)
        else:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=vlad_clusters)
            arr = processed_embs.detach().cpu().numpy()
            assignments_np = km.fit_predict(arr)
            centers = torch.from_numpy(km.cluster_centers_).to(
                device=device,
                dtype=embs.dtype,
            )
            assignments = torch.from_numpy(assignments_np).to(device)
            for c in range(vlad_clusters):
                assign_mask = assignments == c
                if assign_mask.any():
                    residuals = processed_embs[assign_mask] - centers[c]
                    buf[c] = residuals.sum(dim=0)

        do_power = vlad_power if vlad_power is not None else codebook_meta.get('power_normalization', False)
        if do_power and buf.numel() > 0:
            gamma = float(vlad_power_gamma if vlad_power_gamma is not None else codebook_meta.get('power_normalization_gamma', 0.5))
            eps = float(vlad_power_eps if vlad_power_eps is not None else codebook_meta.get('power_normalization_eps', 1e-12))
            buf = torch.sign(buf) * torch.pow(buf.abs() + eps, gamma)

        if vlad_intra_norm is not None:
            intra_default = bool(vlad_intra_norm)
        else:
            intra_default = bool(codebook_meta.get('intra_normalization', False))
        do_intra = intra_default and not disable_intra_norm
        if do_intra and buf.numel() > 0:
            norms = buf.norm(dim=1, keepdim=True)
            eps = 1e-12
            buf = torch.where(norms > eps, buf / torch.clamp(norms, min=eps), buf)

        vec = buf.reshape(-1)
    else:
        raise ValueError(f"Unknown aggregation method '{method}'")

    if method == 'vlad':
        final_normalize = codebook_meta.get('final_normalize', True)
    else:
        final_normalize = True

    if disable_final_norm:
        final_normalize = False

    if final_normalize and vec.numel() > 0:
        vec = F.normalize(vec, p=2, dim=0)
    return vec


def apply_esvm_normalization(
    embeddings: torch.Tensor,
    *,
    num_negatives: int = 1000,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 42,
    n_jobs: int = 1,
    progress: bool = True,
    negatives: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exemplar-SVM (ESVM) re-normalisation of retrieval descriptors.

    For every descriptor, train a linear SVM with that descriptor as the *sole
    positive* against a fixed pool of ``num_negatives`` generic negatives,
    then replace the descriptor with the L2-normalised SVM weight vector.
    Comparing the resulting hyperplane normals by cosine is the standard,
    codebook-free ESVM post-processing that has long topped ICDAR
    writer-retrieval benchmarks (Christlein et al.); it adapts each query
    against a common background instead of relying on a global transform (so,
    unlike PCA-whitening, it does not collapse the shared descriptor
    directions).

    Parameters
    ----------
    embeddings : ``[N, D]`` page descriptors.
    num_negatives : negatives per exemplar (``<=0`` → use the whole pool).  A
        shared pool sampled once keeps the background consistent and bounds cost.
    C : SVM regularisation strength.
    n_jobs : parallel workers (joblib threads); 1 = serial.
    negatives : optional ``[M, D]`` WRITER-DISJOINT negative pool (e.g. the
        validation-set descriptors, Christlein's protocol).  When omitted the
        pool is sampled from ``embeddings`` itself - beware: on a leave-one-out
        corpus a query's own same-writer pages can then land in its negative
        pool and the SVM is trained to push its true positives away.

    Returns ``[N, D']`` float tensor of L2-normalised ESVM descriptors.  Falls
    back to the (L2-normalised) input when scikit-learn is unavailable or N is
    too small.
    """
    import numpy as np

    X = embeddings.detach().cpu().float().numpy()
    N = X.shape[0]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if N < 3:
        return torch.from_numpy(X).to(dtype=torch.float32)

    try:
        from sklearn.svm import LinearSVC
    except Exception:
        logging.getLogger(__name__).warning(
            "ESVM requested but scikit-learn is unavailable; skipping."
        )
        return torch.from_numpy(X).to(dtype=torch.float32)

    rng = np.random.default_rng(seed)
    pool = int(num_negatives or 0)

    X_neg = None
    if negatives is not None and negatives.numel() > 0:
        X_neg = negatives.detach().cpu().float().numpy()
        X_neg = X_neg / (np.linalg.norm(X_neg, axis=1, keepdims=True) + 1e-12)
        if X_neg.shape[1] != X.shape[1]:
            raise ValueError(
                f"ESVM negatives dim {X_neg.shape[1]} != descriptor dim {X.shape[1]} "
                "(apply the same PCA/whitening to both)."
            )
        if 0 < pool < X_neg.shape[0]:
            keep = rng.choice(X_neg.shape[0], size=pool, replace=False)
            X_neg = X_neg[keep]

    if X_neg is None:
        if pool <= 0 or pool >= N:
            pool = N - 1
        neg_pool_idx = rng.choice(N, size=min(pool + 1, N), replace=False)

    def _fit(i: int) -> np.ndarray:
        if X_neg is not None:
            Xtr = np.vstack([X[i][None, :], X_neg])
        else:
            negs = neg_pool_idx[neg_pool_idx != i][:pool]
            Xtr = np.vstack([X[i][None, :], X[negs]])
        ytr = np.zeros(Xtr.shape[0], dtype=np.int8)
        ytr[0] = 1
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=max_iter)
        clf.fit(Xtr, ytr)
        w = clf.coef_.ravel().astype(np.float32)
        nrm = float(np.linalg.norm(w))
        return (w / nrm) if nrm > 0 else X[i]

    idxs = range(N)
    if progress:
        try:
            from tqdm import tqdm
            idxs = tqdm(idxs, total=N, desc="ESVM")
        except Exception:
            pass

    rows = None
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            rows = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
                delayed(_fit)(i) for i in range(N)
            )
        except Exception:
            rows = None
    if rows is None:
        rows = [_fit(i) for i in idxs]

    return torch.from_numpy(np.stack(rows, axis=0)).to(dtype=torch.float32)
