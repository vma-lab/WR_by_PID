import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm


_cv2 = None
_unfold = None


def _ensure_cv2():
    """Lazy-load cv2 on first use and pin its thread pool."""
    global _cv2
    if _cv2 is None:
        import cv2
        cv2.setNumThreads(0)
        _cv2 = cv2
    return _cv2


def _ensure_unfold():
    """Lazy-load unfold_transform on first use."""
    global _unfold
    if _unfold is None:
        import unfold_transform as _uf
        _unfold = _uf
    return _unfold


def canonical_image_size(
    target: int,
    *,
    tile_size: int,
    overlap: float,
    max_size: int | None = None,
) -> int:
    """Round ``target`` down to the nearest tile-grid-aligned canvas side.

    A page of side ``S = tile_size + step * n`` (``step = tile_size * (1 -
    overlap)``) tiles perfectly with no edge truncation and no right/bottom
    strip lost to the foreground filter.  Non-canonical sizes silently drop
    a strip of up to ``step`` pixels per edge.  When ``max_size`` is set the
    result is capped; the floor is always ``tile_size``.
    """

    if target <= tile_size:
        return tile_size
    step = max(int(round(tile_size * (1.0 - float(overlap)))), 1)
    n = (target - tile_size) // step
    candidate = tile_size + step * int(n)
    if max_size and max_size > 0:
        if candidate > max_size:
            n_cap = (max_size - tile_size) // step
            candidate = tile_size + step * int(max(0, n_cap))
    return max(int(candidate), tile_size)


def _apply_binarisation(
    img_rgb: np.ndarray,
    *,
    method: str,
    window: int,
    k: float,
    morph_open_kernel: int = 0,
) -> np.ndarray:
    """Return a Sauvola-binarised RGB image (black ink on white paper).

    Output is always 3-channel RGB so downstream tensor conversion and
    normalisation continue to work unchanged.  ``morph_open_kernel`` (>0)
    applies a morphological opening with a ``(k×k)`` elliptical kernel to
    the *ink* mask (0 in the shipped configs).
    """
    cv2 = _ensure_cv2()
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    method_norm = (method or "sauvola").lower()
    if method_norm == "sauvola":
        win = int(window) if window else 25
        if win % 2 == 0:
            win += 1
        gray_f = gray.astype(np.float32)
        mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
        sq = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
        var = np.maximum(sq - mean * mean, 0.0)
        std = np.sqrt(var)
        threshold = mean * (1.0 + float(k) * ((std / 128.0) - 1.0))
        binary = (gray_f >= threshold).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unsupported binarisation method '{method}'. Expected 'sauvola'.")

    open_k = int(morph_open_kernel or 0)
    if open_k > 0:
        ink_mask = (binary == 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
        binary = (ink_mask == 0).astype(np.uint8) * 255

    return np.stack([binary] * 3, axis=-1)


class CV2Loader:
    def __init__(
        self,
        image_size: int,
        min_size: int,
        tile_overlap: float = 0.1,
        *,
        binarisation: dict | None = None,
        max_image_size: int | None = None,
    ):
        """
        Args:
            image_size: max square side (e.g. 827).  ``0`` means "keep
                native size", clamped down to the nearest canonical
                tile-aligned value.
            min_size: tile_size (e.g. 224)
            tile_overlap: fraction overlap between tiles (0 ≤ overlap < 1)
            binarisation: optional dict ``{enable, mode, after_resize, method,
                window, k, morph_open_kernel}``.  The loaded RGB image is
                replaced by its Sauvola binarisation only when ``enable=True``
                and ``mode == "input"`` (``mode="filter"`` keeps the image and
                drives the ink-count gate instead).
            max_image_size: safety cap (px) applied to the long side when
                ``image_size=0`` (native); ``0``/``None`` disables it.  Callers
                normally pass ``cfg.dataset.max_image_size`` here.  For backward
                compatibility a ``max_image_size`` key inside ``binarisation`` is
                used as a fallback when this argument is not given.
        """
        self.image_size = image_size
        self.tile_size = min_size
        self.overlap = tile_overlap
        self.binarisation = binarisation or {}
        if max_image_size is None:
            max_image_size = (binarisation or {}).get("max_image_size", 0)
        self.max_image_size = int(max_image_size or 0)

    def __call__(self, path: str) -> Image.Image:
        cv2 = _ensure_cv2()
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bin_mode = str(self.binarisation.get("mode", "filter") or "filter").lower()
        bin_enabled = bool(self.binarisation.get("enable", False)) and bin_mode == "input"
        after_resize = bool(self.binarisation.get("after_resize", True))

        if self.image_size is not None and int(self.image_size) == -1:
            if bin_enabled:
                img = self._binarise(img)
            h, w = img.shape[:2]
            t = int(self.tile_size)
            if h < t or w < t:
                img = cv2.copyMakeBorder(
                    img, 0, max(0, t - h), 0, max(0, t - w),
                    borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255),
                )
            return Image.fromarray(img)

        if bin_enabled and not after_resize:
            img = self._binarise(img)

        h, w = img.shape[:2]
        geom = self.geometry(h, w)

        padded_rgb = self._resize_and_pad(
            img, geom["out_h"], geom["out_w"],
            binarise=(bin_enabled and after_resize),
        )

        return Image.fromarray(padded_rgb)

    def geometry(self, h: int, w: int) -> dict:
        """The frame mapping ``__call__`` applies to an (h, w) page - no
        pixels touched.  THE single source of truth for anything that must
        index the loaded page in a different source image (e.g. the
        distillation ``teacher.view_dir`` path).

        Returns ``{mode, out_h, out_w, scale, new_h, new_w, pad_top,
        pad_left}``: native pixel (y, x) lands at
        ``(round(y*scale)+pad_top, round(x*scale)+pad_left)`` on the
        ``out_h × out_w`` output; everything outside the ``new_h × new_w``
        window is padding.  ``image_size=-1`` (native-raw) is identity with
        bottom/right pad only for pages smaller than one tile.
        """
        t = int(self.tile_size)
        if self.image_size is not None and int(self.image_size) == -1:
            return dict(mode="native-raw", out_h=max(h, t), out_w=max(w, t),
                        scale=1.0, new_h=h, new_w=w, pad_top=0, pad_left=0)

        short_side, long_side = (min(h, w), max(h, w))
        ratio = long_side / max(short_side, 1)
        stride = max(int(round(t * (1 - self.overlap))), 1)

        if self.image_size and self.image_size > 0:
            effective_image_size = int(self.image_size)
        else:
            effective_image_size = canonical_image_size(
                long_side,
                tile_size=t,
                overlap=self.overlap,
                max_size=self.max_image_size or None,
            )

        grid = []
        k = 1
        while True:
            size_k = int(round(t + (k - 1) * stride))
            if size_k > effective_image_size:
                break
            grid.append(size_k)
            k += 1
        if not grid:
            grid = [min(effective_image_size, t)]

        candidates = [g for g in grid if int(round(g * ratio)) <= effective_image_size]
        if candidates:
            target_short = min(candidates, key=lambda x: abs(x - short_side))
            target_side = int(round(target_short * ratio))
        else:
            target_side = max(grid)

        scale = min(target_side / h, target_side / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        pad_top = (target_side - new_h) // 2
        pad_left = (target_side - new_w) // 2
        return dict(mode="canvas", out_h=target_side, out_w=target_side,
                    scale=scale, new_h=new_h, new_w=new_w,
                    pad_top=pad_top, pad_left=pad_left)

    def _binarise(self, img: np.ndarray) -> np.ndarray:
        return _apply_binarisation(
            img,
            method=self.binarisation.get("method", "sauvola"),
            window=self.binarisation.get("window", 25),
            k=self.binarisation.get("k", 0.2),
            morph_open_kernel=self.binarisation.get("morph_open_kernel", 3),
        )

    def _resize_and_pad(self,
                        img: np.ndarray,
                        target_h: int,
                        target_w: int,
                        binarise: bool = False) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        cv2 = _ensure_cv2()
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if binarise:
            resized = self._binarise(resized)

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom,
            pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        return padded


class FlattenedTileDataset(Dataset):
    """Flatten every sliding-window tile into a separate sample.

    Parameters
    ----------
    folder_dataset:
        The base :class:`ImageFolder` style dataset containing the full images.
    tile_fn:
        Callable that produces tiles from an image.  This is typically an
        :class:`~torchvision.transforms.Compose` containing
        :class:`~unfold_transform.UnfoldTransform`.  The tile geometry
        (``tile_size``, ``step``) and all foreground-filter thresholds
        (``max_padding_pct``, ``min_stroke_pct``, ``bg_percentile``,
        ``bg_margin``, ``canny_low``, ``canny_high``) are read off this
        object so training stays aligned with inference.
    tile_transform:
        Additional transform applied to each extracted tile (e.g. normalisation).
    min_stroke_pct:
        Optional stroke-density heuristic. After cropping away white padding via
        a content bounding box, CLAHE, a small median blur, and Canny edge
        detection are applied inside the box. Tiles whose edge-pixel fraction
        falls below this threshold are discarded.
    cache_size:
        Size of the per-worker LRU cache used by :meth:`_load_image` to avoid
        re-decoding the same raw image repeatedly when several tiles from it
        appear close together in a batch.  Each cached entry holds one decoded
        PIL image.
    scan_workers:
        Optional override for the number of threads used by the initial tile
        scan.  ``None`` picks an auto value based on ``os.cpu_count()`` and the
        dataset size.  Because the scan is dominated by OpenCV calls that
        release the GIL, threads yield real parallelism and are Windows-safe
        (no extra child processes).
    """

    DEFAULT_CACHE_SIZE = 16

    def __init__(
        self,
        folder_dataset,
        tile_fn,
        tile_transform,
        *,
        min_stroke_pct: float = None,
        cache_size: int = None,
        scan_workers: int = None,
    ):
        self.dataset = folder_dataset
        self.tile_fn = tile_fn
        self.tile_transform = tile_transform
        self.min_stroke_pct = min_stroke_pct
        self.tile_records = []
        self.labels = []

        self._cache_size = int(
            cache_size if cache_size is not None else self.DEFAULT_CACHE_SIZE
        )

        self.tile_size = None
        self.step = None
        self.max_padding_pct = None
        uf = _ensure_unfold()
        self.bg_percentile = uf.DEFAULT_BG_PERCENTILE
        self.bg_margin = uf.DEFAULT_BG_MARGIN
        self.canny_low = uf.DEFAULT_CANNY_LOW
        self.canny_high = uf.DEFAULT_CANNY_HIGH

        self._extract_geometry_from_tile_fn()

        scan_args = [
            (img_idx, path, lbl)
            for img_idx, (path, lbl) in enumerate(self.dataset.samples)
        ]

        if not scan_args:
            return

        workers = self._resolve_scan_workers(scan_workers, len(scan_args))

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results_iter = executor.map(self._scan_one_image, scan_args)
                for result in tqdm(
                    results_iter,
                    total=len(scan_args),
                    desc="Creating tiles",
                    unit="img",
                ):
                    self._extend_from_scan_result(result)
        else:
            for arg in tqdm(
                scan_args, desc="Creating tiles", unit="img"
            ):
                self._extend_from_scan_result(self._scan_one_image(arg))

    def _extract_geometry_from_tile_fn(self):
        """Populate geometry/filter thresholds from ``self.tile_fn``."""
        candidates = []
        if self.tile_fn is not None:
            candidates.append(self.tile_fn)
            inner = getattr(self.tile_fn, "transforms", None)
            if inner:
                candidates.extend(inner)

        for t in candidates:
            if hasattr(t, "tile_size") and hasattr(t, "step"):
                self.tile_size = int(t.tile_size)
                self.step = int(t.step)
                self.max_padding_pct = getattr(t, "max_padding_pct", None)
                if self.min_stroke_pct is None:
                    self.min_stroke_pct = getattr(t, "min_stroke_pct", None)
                uf = _ensure_unfold()
                self.bg_percentile = float(
                    getattr(t, "bg_percentile", uf.DEFAULT_BG_PERCENTILE)
                )
                self.bg_margin = float(
                    getattr(t, "bg_margin", uf.DEFAULT_BG_MARGIN)
                )
                self.canny_low = int(
                    getattr(t, "canny_low", uf.DEFAULT_CANNY_LOW)
                )
                self.canny_high = int(
                    getattr(t, "canny_high", uf.DEFAULT_CANNY_HIGH)
                )
                break

    @staticmethod
    def _resolve_scan_workers(override, num_images: int) -> int:
        """Decide the thread count used by the initial tile scan."""
        if override is not None:
            try:
                return max(1, int(override))
            except (TypeError, ValueError):
                pass

        env_value = os.environ.get("METRIC_TILE_SCAN_WORKERS", "").strip()
        if env_value:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass

        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, num_images, 16))

    def _extend_from_scan_result(self, result):
        _, lbl, records = result
        for rec in records:
            self.tile_records.append(rec)
            self.labels.append(lbl)

    def _scan_one_image(self, arg):
        """Return ``(img_idx, label, [records])`` for a single image.

        Runs in a worker thread.  OpenCV calls release the GIL so multiple
        threads make real progress on large datasets.
        """
        img_idx, path, lbl = arg
        raw_img = self.dataset.loader(path)
        raw_np = np.asarray(raw_img)
        records = []

        if self.tile_size is None or self.step is None:
            records.append((img_idx, 0, 0))
            return img_idx, lbl, records

        h, w = raw_np.shape[:2]
        do_filter = (
            self.max_padding_pct is not None
            or self.min_stroke_pct is not None
        )

        gray_full = None
        edge_map_full = None
        if do_filter:
            cv2 = _ensure_cv2()
            uf = _ensure_unfold()
            if raw_np.ndim == 3 and raw_np.shape[2] == 3:
                gray_full = cv2.cvtColor(
                    np.ascontiguousarray(raw_np), cv2.COLOR_RGB2GRAY
                )
            else:
                gray_full = raw_np.squeeze()
            if self.min_stroke_pct is not None:
                edge_map_full = uf.build_image_edge_map(
                    gray_full,
                    canny_low=self.canny_low,
                    canny_high=self.canny_high,
                )

        def _verdict(top: int, left: int) -> str:
            if not do_filter:
                return "fg"
            return _ensure_unfold().tile_filter_verdict(
                gray_full,
                edge_map_full,
                top,
                left,
                self.tile_size,
                max_padding_pct=self.max_padding_pct,
                min_stroke_pct=self.min_stroke_pct,
                bg_percentile=self.bg_percentile,
                bg_margin=self.bg_margin,
            )

        for top in range(0, max(h - self.tile_size, 0) + 1, self.step):
            for left in range(0, max(w - self.tile_size, 0) + 1, self.step):
                if top + self.tile_size > h or left + self.tile_size > w:
                    continue
                if _verdict(top, left) == "fg":
                    records.append((img_idx, top, left))

        if not records:
            ci = max((h - self.tile_size) // 2, 0)
            cj = max((w - self.tile_size) // 2, 0)
            if ci + self.tile_size <= h and cj + self.tile_size <= w and _verdict(ci, cj) == "fg":
                records.append((img_idx, ci, cj))

        return img_idx, lbl, records

    def __len__(self):
        return len(self.tile_records)

    def _load_image(self, img_idx):
        """Load a full image with a per-worker LRU cache.

        The cache is populated lazily on first access inside each DataLoader
        worker, so there is no interaction with the parent-process tile scan.
        Using :class:`collections.OrderedDict` gives true LRU semantics where
        recently-touched entries are kept and the oldest are evicted.
        """
        cache = getattr(self, "_img_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._img_cache = cache

        if img_idx in cache:
            cache.move_to_end(img_idx)
            return cache[img_idx]

        path, _ = self.dataset.samples[img_idx]
        img = self.dataset.loader(path)
        cache[img_idx] = img
        cache.move_to_end(img_idx)
        while len(cache) > self._cache_size:
            cache.popitem(last=False)
        return img

    def __getitem__(self, tile_index):
        img_idx, top, left = self.tile_records[tile_index]
        _, lbl = self.dataset.samples[img_idx]
        full_img = self._load_image(img_idx)
        tile = full_img if self.tile_size is None else full_img.crop(
            (left, top, left + self.tile_size, top + self.tile_size)
        )
        tile = transforms.ToTensor()(tile)
        if self.tile_transform:
            tile = self.tile_transform(tile)
        return tile, lbl
