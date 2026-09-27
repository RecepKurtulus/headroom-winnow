"""Marker format and hash rules must match what Headroom can store and resolve."""

from __future__ import annotations

import hashlib

from headroom_squeez.markers import HASH_LENGTH, MARKER_RE, content_hash, format_marker


def test_hash_is_sha256_prefix_lowercase_hex() -> None:
    text = "some dropped\nlines\n"
    h = content_hash(text)
    assert h == hashlib.sha256(text.encode()).hexdigest()[:24]
    assert len(h) == HASH_LENGTH
    assert h == h.lower()
    int(h, 16)


def test_marker_matches_headroom_resolution_regex() -> None:
    h = content_hash("x")
    marker = format_marker(h, 132)
    assert marker == f"<<ccr:{h} 132_lines_offloaded>>"
    match = MARKER_RE.search(f"before {marker} after")
    assert match is not None
    assert match.group(1) == h


def test_marker_regex_is_headrooms() -> None:
    from headroom.ccr.marker_resolution import _MARKER_RE

    assert MARKER_RE.pattern == _MARKER_RE.pattern
