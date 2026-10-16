from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2


__all__ = [
    "is_foreground",
    "compute_patch_mask",
    "UnfoldTransform",
    "build_image_edge_map",
    "tile_passes_filter",
]


DEFAULT_CANNY_LOW = 100
DEFAULT_CANNY_HIGH = 200

DEFAULT_BG_PERCENTILE = 95
DEFAULT_BG_MARGIN = 10


def is_foreground(
    tile_np,
    max_padding_pct,
    min_stroke_pct,
    *,
    bg_percentile: float = DEFAULT_BG_PERCENTILE,
    bg_margin: float = DEFAULT_BG_MARGIN,
    canny_low: int = DEFAULT_CANNY_LOW,
    canny_high: int = DEFAULT_CANNY_HIGH,
    binary_mode: bool = False,
    binary_ink_threshold: float = 0.05,
) -> bool:
    """Decide whether a tile contains meaningful foreground (ink/strokes).

    Stage 1: **padding gate**: An adaptive percentile-based threshold is
    used to estimate the background intensity of the tile.  Pixels darker
    than ``percentile(gray, bg_percentile) - bg_margin`` are treated as
    potential content.  A tight bounding box is computed around these
    pixels and the tile is rejected when the fraction of the tile that
    lies *outside* the bounding box is at least ``max_padding_pct``.

    Stage 2: **ink-density gate**: Within the content bounding box,
    measure the fraction of ink pixels.  When ``binary_mode=False`` (the
    default, grayscale input) this is done via CLAHE → median blur →
    Canny edge density, compared against ``min_stroke_pct``.  When
    ``binary_mode=True`` we instead count the dark pixels in the bbox
    directly (the input is already binary, so CLAHE+Canny is a wasted
    op and over-counts edges), compared against
    ``binary_ink_threshold``.

    Parameters
    ----------
    tile_np : np.ndarray
        Tile image in RGB uint8 ([H, W, 3]) or grayscale ([H, W]).
    max_padding_pct : float or None
        Maximum allowed fraction of the tile that is background padding.
    min_stroke_pct : float or None
        Minimum required edge-pixel fraction within the content bbox
        (grayscale path only; ignored when ``binary_mode=True``).
    bg_percentile : float
        Percentile (0–100) used to estimate the background intensity.
    bg_margin : float
        Intensity margin below the background reference for the
        foreground mask.
    canny_low, canny_high : int
        Lower and upper Canny thresholds for edge detection (grayscale
        path only).
    binary_mode : bool
        When True, replace the Canny edge-density Stage 2 with a simple
        ink-pixel count.  Auto-enabled by callers when
        ``dataset.binarisation.enable`` is true; otherwise leave False.
    binary_ink_threshold : float
        Minimum ink-pixel fraction within the content bbox in
        binary_mode.  Default 0.05 = 5%.
    """
    gray = (
        cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        if tile_np.ndim == 3 and tile_np.shape[2] == 3
        else tile_np.squeeze()
    )

    bg_ref = float(np.percentile(gray, bg_percentile))
    fg_threshold = max(bg_ref - bg_margin, 0)
    mask = gray < fg_threshold

    if not mask.any():
        return False

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    bbox_area = (y1 - y0) * (x1 - x0)

    tile_area = tile_np.shape[0] * tile_np.shape[1]
    pad_frac = 1.0 - bbox_area / tile_area
    if max_padding_pct is not None and pad_frac >= max_padding_pct:
        return False

    if binary_mode:
        bbox_mask = mask[y0:y1, x0:x1]
        ink_frac = float(np.count_nonzero(bbox_mask)) / float(bbox_area)
        if ink_frac < float(binary_ink_threshold):
            return False
        return True

    if min_stroke_pct is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        blur = cv2.medianBlur(eq, 3)
        edges = cv2.Canny(blur, canny_low, canny_high)
        edges[:y0, :] = 0
        edges[y1:, :] = 0
        edges[:, :x0] = 0
        edges[:, x1:] = 0
        edge_frac = float(np.count_nonzero(edges)) / bbox_area
        if edge_frac < min_stroke_pct:
            return False

    return True


def build_image_edge_map(
    gray_full: np.ndarray,
    *,
    canny_low: int = DEFAULT_CANNY_LOW,
    canny_high: int = DEFAULT_CANNY_HIGH,
) -> np.ndarray:
    """Precompute the CLAHE + median blur + Canny edge map for a full image.

    The CLAHE/median/Canny pipeline matches ``is_foreground``'s stroke-density
    gate, but is applied to the whole image exactly once.  Callers can then
    slice the returned edge map per-tile when deciding which tiles to keep,
    avoiding O(n_tiles) redundant CLAHE/Canny calls.

    Parameters
    ----------
    gray_full : np.ndarray
        Grayscale uint8 image of shape ``[H, W]``.
    canny_low, canny_high : int
        Lower / upper Canny thresholds; defaults match :data:`DEFAULT_CANNY_LOW`
        and :data:`DEFAULT_CANNY_HIGH`.

    Returns
    -------
    np.ndarray
        Edge map of shape ``[H, W]`` with values in ``{0, 255}`` (the raw
        output of :func:`cv2.Canny`).
    """
    if gray_full.ndim != 2:
        raise ValueError("build_image_edge_map expects a 2-D grayscale array")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray_full)
    blur = cv2.medianBlur(eq, 3)
    edges = cv2.Canny(blur, int(canny_low), int(canny_high))
    return edges


def tile_filter_verdict(
    gray_full: np.ndarray,
    edge_map_full: Optional[np.ndarray],
    top: int,
    left: int,
    tile_size: int,
    *,
    max_padding_pct,
    min_stroke_pct,
    bg_percentile: float = DEFAULT_BG_PERCENTILE,
    bg_margin: float = DEFAULT_BG_MARGIN,
    binary_mode: bool = False,
    binary_ink_threshold: float = 0.05,
) -> str:
    """Classify a tile at ``(top, left)``: ``"fg"`` | ``"bg"`` | ``"skip"``.

    Same computation as the original boolean filter (see
    :func:`tile_passes_filter`, which wraps this), but the reject path is
    split into two classes:

    ``"bg"``
        The tile is real page material - it PASSES the padding gate (has
        non-uniform content covering enough of the tile) - but FAILS the
        stroke/ink gate: paper texture, stains, degradation, no handwriting.
    ``"skip"``
        Uniform or padding-dominated tiles (scanner bed, borders, blank
        margins with no measurable content).

    When no stroke/ink gate is active (``min_stroke_pct is None`` and not
    ``binary_mode``) every content tile is ``"fg"`` and no ``"bg"`` class
    exists.
    """
    if gray_full.ndim != 2:
        raise ValueError("tile_filter_verdict expects a 2-D grayscale array")

    ts = int(tile_size)
    gray = gray_full[top:top + ts, left:left + ts]
    if gray.shape[0] != ts or gray.shape[1] != ts:
        return "skip"

    bg_ref = float(np.percentile(gray, bg_percentile))
    fg_threshold = max(bg_ref - bg_margin, 0)
    mask = gray < fg_threshold
    if not mask.any():
        return "skip"

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_area = (y1 - y0) * (x1 - x0)
    if bbox_area <= 0:
        return "skip"

    tile_area = ts * ts
    pad_frac = 1.0 - bbox_area / tile_area
    if max_padding_pct is not None and pad_frac >= max_padding_pct:
        return "skip"

    if binary_mode:
        bbox_mask = mask[y0:y1, x0:x1]
        ink_frac = int(np.count_nonzero(bbox_mask)) / float(bbox_area)
        if ink_frac < float(binary_ink_threshold):
            return "bg"
        return "fg"

    if min_stroke_pct is not None:
        if edge_map_full is None:
            raise ValueError(
                "edge_map_full is required when min_stroke_pct is not None"
            )
        edges_tile = edge_map_full[top:top + ts, left:left + ts]
        edges_bbox = edges_tile[y0:y1, x0:x1]
        edge_count = int(np.count_nonzero(edges_bbox))
        edge_frac = edge_count / float(bbox_area)
        if edge_frac < min_stroke_pct:
            return "bg"

    return "fg"


def tile_passes_filter(
    gray_full: np.ndarray,
    edge_map_full: Optional[np.ndarray],
    top: int,
    left: int,
    tile_size: int,
    *,
    max_padding_pct,
    min_stroke_pct,
    bg_percentile: float = DEFAULT_BG_PERCENTILE,
    bg_margin: float = DEFAULT_BG_MARGIN,
    binary_mode: bool = False,
    binary_ink_threshold: float = 0.05,
) -> bool:
    """Decide whether a tile at ``(top, left)`` passes the foreground filter.

    This is semantically equivalent to :func:`is_foreground` but reuses an
    image-level edge map produced by :func:`build_image_edge_map` so that the
    expensive CLAHE/median/Canny pipeline runs once per full image instead of
    once per candidate tile.

    The adaptive background percentile is still evaluated per-tile so the
    filter decisions match :func:`is_foreground` pixel-for-pixel on the
    padding gate.

    Thin wrapper over :func:`tile_filter_verdict` (``verdict == "fg"``) so
    existing callers keep their boolean contract.

    Parameters
    ----------
    gray_full : np.ndarray
        Grayscale uint8 image containing the tile (same coordinates as the
        image used to build ``edge_map_full``).
    edge_map_full : np.ndarray or None
        Precomputed edge map from :func:`build_image_edge_map`.  May be
        ``None`` when ``min_stroke_pct`` is ``None``.
    top, left, tile_size : int
        Tile location and side length within ``gray_full``.
    max_padding_pct, min_stroke_pct : float or None
        Matching :func:`is_foreground` semantics.
    bg_percentile, bg_margin : float
        Matching :func:`is_foreground` semantics.
    """
    return tile_filter_verdict(
        gray_full,
        edge_map_full,
        top,
        left,
        tile_size,
        max_padding_pct=max_padding_pct,
        min_stroke_pct=min_stroke_pct,
        bg_percentile=bg_percentile,
        bg_margin=bg_margin,
        binary_mode=binary_mode,
        binary_ink_threshold=binary_ink_threshold,
    ) == "fg"


def compute_patch_mask(
    tile: torch.Tensor,
    patch_size: int,
    max_padding_pct,
    min_stroke_pct,
    *,
    bg_percentile: float = DEFAULT_BG_PERCENTILE,
    bg_margin: float = DEFAULT_BG_MARGIN,
    canny_low: int = DEFAULT_CANNY_LOW,
    canny_high: int = DEFAULT_CANNY_HIGH,
    binary_mode: bool = False,
    binary_ink_threshold: float = 0.05,
) -> torch.BoolTensor:
    """Compute a per-patch foreground mask for a tile.

    Instead of running ``is_foreground`` on each tiny patch independently
    (which is noisy at 14×14 resolution), this function computes a
    single edge-density map for the full tile and then downsamples to the
    patch grid.  This is both faster and more stable.

    When ``binary_mode=True``, the per-patch test uses ink-pixel density
    (``content_pooled >= binary_ink_threshold``) instead of Canny edge
    density.  CLAHE/Canny are skipped - they're irrelevant on binary
    input.
    """
    if tile.dim() != 3:
        raise ValueError("tile must have shape [C, H, W]")

    _, H, W = tile.shape
    if H < patch_size or W < patch_size:
        raise ValueError("patch_size must be <= tile spatial dimensions")
    if H % patch_size != 0 or W % patch_size != 0:
        raise AssertionError(
            f"Tile spatial dims ({H}, {W}) must be divisible by patch_size {patch_size}"
        )

    expected_rows = H // patch_size
    expected_cols = W // patch_size

    tile_np = (
        tile.detach()
        .mul(255)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    tile_np = np.ascontiguousarray(tile_np)

    gray = (
        cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        if tile_np.ndim == 3 and tile_np.shape[2] == 3
        else tile_np.squeeze()
    )

    bg_ref = float(np.percentile(gray, bg_percentile))
    fg_threshold = max(bg_ref - bg_margin, 0)
    content_mask = (gray < fg_threshold).astype(np.float32)

    if binary_mode:
        edge_pooled = None
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        blur = cv2.medianBlur(eq, 3)
        edges = cv2.Canny(blur, canny_low, canny_high)
        edge_map = (edges > 0).astype(np.float32)
        edge_pooled = cv2.resize(
            edge_map, (expected_cols, expected_rows),
            interpolation=cv2.INTER_AREA,
        )

    content_pooled = cv2.resize(
        content_mask, (expected_cols, expected_rows),
        interpolation=cv2.INTER_AREA,
    )

    content_floor = float(min_stroke_pct) if min_stroke_pct is not None else 0.0
    padding_ok = content_pooled > content_floor
    if binary_mode:
        ink_ok = content_pooled >= float(binary_ink_threshold)
        mask_np = padding_ok & ink_ok
    elif min_stroke_pct is not None:
        stroke_ok = edge_pooled >= min_stroke_pct
        mask_np = padding_ok & stroke_ok
    else:
        mask_np = padding_ok

    mask_tensor = torch.tensor(mask_np, dtype=torch.bool, device=tile.device)
    if mask_tensor.shape != (expected_rows, expected_cols):
        raise AssertionError(
            f"Patch mask has shape {tuple(mask_tensor.shape)} but expected {(expected_rows, expected_cols)}"
        )
    return mask_tensor


class UnfoldTransform:
    """Sliding-window tiling on a Tensor [C,H,W] → [n_tiles, C, tile, tile].

    Tiles are filtered by padding fraction (``max_padding_pct``) and by a simple
    stroke-density heuristic (``min_stroke_pct``).  Non-white pixels are used to
    compute a tight bounding box of content within each tile.  The padding
    fraction is the complement of the bbox area relative to the full tile.  The
    stroke heuristic applies CLAHE, a small median blur, and Canny edge
    detection inside the bbox; the edge fraction is treated as "ink" density.
    Tiles with too much padding or too little edge content are discarded.

    The ``patch_size`` argument does not change the tiling behaviour, but is
    exposed as an attribute so downstream datasets can align per-tile patch
    masks with the vision backbone's native patch resolution.
    """

    def __init__(
        self,
        tile_size: int,
        overlap: float = 0.0,
        max_padding_pct: float = None,
        min_stroke_pct: float = None,
        *,
        patch_size: int = None,
        bg_percentile: float = DEFAULT_BG_PERCENTILE,
        bg_margin: float = DEFAULT_BG_MARGIN,
        canny_low: int = DEFAULT_CANNY_LOW,
        canny_high: int = DEFAULT_CANNY_HIGH,
        binary_mode: bool = False,
        binary_ink_threshold: float = 0.05,
    ):
        self.tile_size = tile_size
        self.step = max(int(tile_size * (1 - overlap)), 1)
        self.max_padding_pct = max_padding_pct
        self.min_stroke_pct = min_stroke_pct
        self.patch_size = (
            max(1, int(patch_size)) if patch_size is not None else tile_size
        )
        self.bg_percentile = float(bg_percentile)
        self.bg_margin = float(bg_margin)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.binary_mode = bool(binary_mode)
        self.binary_ink_threshold = float(binary_ink_threshold)

    def _is_fg(self, tile_np) -> bool:
        """Per-tile foreground check (CLAHE/Canny on one tile).

        The image-level ``__call__`` path uses :func:`build_image_edge_map` +
        :func:`tile_passes_filter` instead.
        """
        return is_foreground(
            tile_np, self.max_padding_pct, self.min_stroke_pct,
            bg_percentile=self.bg_percentile,
            bg_margin=self.bg_margin,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            binary_mode=self.binary_mode,
            binary_ink_threshold=self.binary_ink_threshold,
        )

    def _prepare_filter_maps(self, x: torch.Tensor):
        """Return ``(gray_full, edge_map_full)`` for ``x`` or ``(None, None)``.

        Produces numpy uint8 grayscale and edge maps for the full padded
        image so that tile-level filtering can be performed by slicing
        instead of re-running CLAHE/Canny per tile.
        """
        if self.max_padding_pct is None and self.min_stroke_pct is None:
            return None, None

        image_np = (
            x.detach().mul(255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
        )
        image_np = np.ascontiguousarray(image_np)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            gray_full = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray_full = image_np.squeeze()

        edge_map_full = None
        if self.min_stroke_pct is not None and not self.binary_mode:
            edge_map_full = build_image_edge_map(
                gray_full,
                canny_low=self.canny_low,
                canny_high=self.canny_high,
            )
        return gray_full, edge_map_full

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, H0, W0 = x.shape

        pad_h = max(0, self.tile_size - H0)
        pad_w = max(0, self.tile_size - W0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h or pad_w:
            x = F.pad(
                x,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant', value=1.0
            )

        C, H, W = x.shape

        gray_full, edge_map_full = self._prepare_filter_maps(x)
        do_filter = gray_full is not None

        tiles = []
        for i in range(0, max(H - self.tile_size, 0) + 1, self.step):
            for j in range(0, max(W - self.tile_size, 0) + 1, self.step):
                tile = x[:, i: i + self.tile_size, j: j + self.tile_size]
                if tile.shape[1] != self.tile_size or tile.shape[2] != self.tile_size:
                    continue

                if do_filter and not tile_passes_filter(
                    gray_full,
                    edge_map_full,
                    i,
                    j,
                    self.tile_size,
                    max_padding_pct=self.max_padding_pct,
                    min_stroke_pct=self.min_stroke_pct,
                    bg_percentile=self.bg_percentile,
                    bg_margin=self.bg_margin,
                    binary_mode=self.binary_mode,
                    binary_ink_threshold=self.binary_ink_threshold,
                ):
                    continue
                tiles.append(tile)

        if not tiles:
            ci = max((H - self.tile_size) // 2, 0)
            cj = max((W - self.tile_size) // 2, 0)
            tile = x[:, ci: ci + self.tile_size, cj: cj + self.tile_size]
            if tile.shape[1] != self.tile_size or tile.shape[2] != self.tile_size:
                return torch.empty(0, C, self.tile_size, self.tile_size)

            if do_filter and not tile_passes_filter(
                gray_full,
                edge_map_full,
                ci,
                cj,
                self.tile_size,
                max_padding_pct=self.max_padding_pct,
                min_stroke_pct=self.min_stroke_pct,
                bg_percentile=self.bg_percentile,
                bg_margin=self.bg_margin,
                binary_mode=self.binary_mode,
                binary_ink_threshold=self.binary_ink_threshold,
            ):
                return torch.empty(0, C, self.tile_size, self.tile_size)
            tiles = [tile]

        return torch.stack(tiles, dim=0)
