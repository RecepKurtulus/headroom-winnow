"""CCR retrieval markers and content hashes for dropped line runs.

Every run of lines the pruner drops is replaced by exactly one marker line and
its full text goes into ``CompressOutput.recoverable``. The router persists
that map into the CCR store with ``explicit_hash`` and the agent's retrieval
tool resolves the marker back to the original bytes, so a wrong drop costs one
retrieval round trip instead of lost information.

Two constraints come from Headroom, not from us:

  * The store only accepts lowercase hex hashes, and the agent-side marker
    regex (``headroom/ccr/marker_resolution.py``) is
    ``<<ccr:([a-f0-9]{12,24})[^>]*>>``. A hash outside that shape is silently
    unretrievable.
  * SmartCrusher's row-offload marker is ``<<ccr:HASH N_rows_offloaded>>``; we
    mirror it with ``N_lines_offloaded`` so agents see one familiar format.

The hash is ``sha256(text)[:24]`` — the same rule Headroom uses — so identical
input always yields byte-identical output, which keeps the provider's prompt
cache warm across turns.
"""

from __future__ import annotations

import hashlib
import re

__all__ = ["HASH_LENGTH", "MARKER_RE", "content_hash", "format_marker"]

#: Hex characters kept from the sha256 digest (the store accepts 12-24).
HASH_LENGTH = 24

#: Same pattern Headroom uses to find markers on the agent side.
MARKER_RE = re.compile(r"<<ccr:([a-f0-9]{12,24})[^>]*>>")


def content_hash(text: str) -> str:
    """Return the CCR hash for ``text``.

    Args:
        text: The exact dropped text (line endings included).

    Returns:
        The first :data:`HASH_LENGTH` lowercase hex characters of its sha256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def format_marker(ccr_hash: str, n_lines: int) -> str:
    """Return the single-line marker that stands in for a dropped run.

    Args:
        ccr_hash: Hash from :func:`content_hash` for the dropped text.
        n_lines: Number of lines the marker replaces.

    Returns:
        ``<<ccr:HASH N_lines_offloaded>>`` without a trailing newline.
    """
    return f"<<ccr:{ccr_hash} {n_lines}_lines_offloaded>>"
