"""Task-conditioned line pruning for Headroom, powered by the Squeez span model.

Install the package and opt in by name; installing alone changes nothing::

    ContentRouterConfig(active_external_compressors=["winnow"])

See :class:`~headroom_winnow.compressor.WinnowCompressor` for the contract.
"""

from __future__ import annotations

from .backends import (
    BackendUnavailableError,
    HighlighterBackend,
    PooledBackend,
    SpanBackend,
    make_backend,
)
from .compressor import WinnowCompressor, WinnowSettings

__all__ = [
    "BackendUnavailableError",
    "HighlighterBackend",
    "PooledBackend",
    "SpanBackend",
    "WinnowCompressor",
    "WinnowSettings",
    "make_backend",
]

__version__ = "0.2.0"
