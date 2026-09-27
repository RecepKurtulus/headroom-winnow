"""Task-conditioned line pruning for Headroom, powered by the Squeez span model.

Install the package and opt in by name; installing alone changes nothing::

    ContentRouterConfig(active_external_compressors=["squeez"])

See :class:`~headroom_squeez.compressor.SqueezCompressor` for the contract.
"""

from __future__ import annotations

from .backends import BackendUnavailableError, HighlighterBackend, SpanBackend
from .compressor import SqueezCompressor, SqueezSettings

__all__ = [
    "BackendUnavailableError",
    "HighlighterBackend",
    "SpanBackend",
    "SqueezCompressor",
    "SqueezSettings",
]

__version__ = "0.1.0"
