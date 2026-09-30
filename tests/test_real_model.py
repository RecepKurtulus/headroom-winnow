"""Real-model tests: download and run the pinned highlighter. ``pytest -m slow``."""

from __future__ import annotations

import pytest
from conftest import make_pytest_log
from headroom.transforms.compressor_registry import CompressInput

from headroom_winnow.backends import HighlighterBackend
from headroom_winnow.compressor import WinnowCompressor
from headroom_winnow.markers import MARKER_RE

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def backend() -> HighlighterBackend:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return HighlighterBackend()


def test_spans_point_at_the_failure(backend: HighlighterBackend) -> None:
    log = make_pytest_log()
    spans = backend.find_spans("why does test_login_redirect fail", log)
    assert spans
    assert all(0 <= s < e <= len(log) for s, e in spans)
    touched = " ".join(log[s:e] for s, e in spans)
    assert "login" in touched or "302" in touched or "500" in touched


def test_compressor_end_to_end_with_real_model(backend: HighlighterBackend) -> None:
    log = make_pytest_log(n_passed=200)
    out = WinnowCompressor(backend).compress(
        CompressInput(
            content=log, content_type="text/x-log", query="why does test_login_redirect fail"
        )
    )
    assert out.compressed is True
    assert "AssertionError: expected 302, got 500" in out.content
    assert MARKER_RE.search(out.content)
    assert out.tokens_after < out.tokens_before
