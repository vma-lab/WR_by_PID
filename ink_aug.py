"""Sauvola ink mask: the teacher view of Stage 2 (distill_train.py)."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception as _exc:
    cv2 = None
    _CV2_ERR = _exc


_SAUVOLA_R = 128.0


def sauvola_ink_mask(
    gray_u8: np.ndarray,
    *,
    window: int = 25,
    k: float = 0.2,
    morph_open_kernel: int = 3,
) -> np.ndarray:
    """Boolean ink mask (True = ink) via box-filter Sauvola + despeckle.

    Mirrors the on-device gate in ``model_factory.patch_ink_fraction`` so the
    training-time ink masks and the inference-time ink-count gate agree.
    """
    if cv2 is None:
        raise RuntimeError(f"OpenCV unavailable: {_CV2_ERR!r}")
    g = gray_u8.astype(np.float32)
    w = max(3, int(window) | 1)
    mean = cv2.boxFilter(g, -1, (w, w), borderType=cv2.BORDER_REPLICATE)
    sq = cv2.boxFilter(g * g, -1, (w, w), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.clip(sq - mean * mean, 0.0, None))
    thr = mean * (1.0 + float(k) * (std / _SAUVOLA_R - 1.0))
    ink = g < thr
    mok = int(morph_open_kernel or 0)
    if mok >= 2:
        kern = np.ones((mok, mok), np.uint8)
        ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, kern).astype(bool)
    return ink
