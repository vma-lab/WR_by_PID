"""Pooling layers and aggregation utilities."""

from .salad import (
    HAVE_GEOMLOSS as SALAD_HAS_GEOMLOSS,
    SaladResult,
    salad_pooling,
)

__all__ = [
    "SALAD_HAS_GEOMLOSS",
    "SaladResult",
    "salad_pooling",
]
