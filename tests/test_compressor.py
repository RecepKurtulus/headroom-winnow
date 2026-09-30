"""``WinnowCompressor`` against the contract, with a fake backend."""

from __future__ import annotations

import logging

import pytest
from conftest import FakeBackend, make_pytest_log
from headroom.transforms.compressor_registry import (
    CompressInput,
    Compressor,
    CompressOutput,
)

from headroom_winnow.compressor import CONTENT_TYPES, WinnowCompressor, WinnowSettings
from headroom_winnow.markers import MARKER_RE

QUERY = "why does test_login_redirect fail"


def _inp(content: str, query: str = QUERY) -> CompressInput:
    return CompressInput(content=content, content_type="text/x-log", query=query)


def _restore(out: CompressOutput) -> str:
    return "".join(
        out.recoverable[m.group(1)] if (m := MARKER_RE.match(line)) else line
        for line in out.content.splitlines(keepends=True)
    )


def test_satisfies_protocol_and_descriptor() -> None:
    comp = WinnowCompressor(FakeBackend())
    assert isinstance(comp, Compressor)
    d = comp.descriptor
    assert d.name == "winnow"
    assert d.content_types == list(CONTENT_TYPES)
    assert (d.lossless, d.cost_tier, d.recoverable) == (False, "ml", True)


def test_default_construction_loads_nothing() -> None:
    comp = WinnowCompressor()
    assert comp._backend._model is None  # type: ignore[attr-defined]


def test_prunes_and_keeps_what_matters(sample_log: str) -> None:
    backend = FakeBackend(("test_login_redirect",))
    out = WinnowCompressor(backend).compress(_inp(sample_log))

    assert out.compressed is True
    assert backend.calls == [(QUERY, sample_log)]
    assert out.tokens_after < out.tokens_before * 0.8
    for needed in (
        "test_login_redirect FAILED",
        "Traceback (most recent call last):",
        'File "app/views.py", line 88',
        "AssertionError: expected 302, got 500",
        "test session starts",
        "1 failed, 80 passed",
    ):
        assert needed in out.content
    assert MARKER_RE.search(out.content)
    assert _restore(out) == sample_log


def test_output_is_deterministic(sample_log: str) -> None:
    comp = WinnowCompressor(FakeBackend(("test_login_redirect",)))
    assert comp.compress(_inp(sample_log)) == comp.compress(_inp(sample_log))


@pytest.mark.parametrize(
    ("content", "query", "keywords"),
    [
        (make_pytest_log(), "", ("test_login_redirect",)),  # no query
        (make_pytest_log(), "   ", ("test_login_redirect",)),  # blank query
        (make_pytest_log(n_passed=20), QUERY, ("test_login_redirect",)),  # too short
        (make_pytest_log(), QUERY, ()),  # model found nothing
        (make_pytest_log(), QUERY, ("PASSED",)),  # keeps nearly everything
    ],
    ids=["no-query", "blank-query", "too-short", "no-spans", "low-savings"],
)
def test_passthrough_cases(content: str, query: str, keywords: tuple[str, ...]) -> None:
    out = WinnowCompressor(FakeBackend(keywords)).compress(_inp(content, query))
    assert out.compressed is False
    assert out.content == content
    assert out.recoverable == {}


def test_backend_unavailable_warns_once_then_skips(
    sample_log: str, caplog: pytest.LogCaptureFixture
) -> None:
    backend = FakeBackend(unavailable=True)
    comp = WinnowCompressor(backend)
    with caplog.at_level(logging.WARNING, logger="headroom_winnow"):
        first = comp.compress(_inp(sample_log))
        second = comp.compress(_inp(sample_log))
    assert first.compressed is False and second.compressed is False
    assert second.content == sample_log
    assert len(backend.calls) == 1
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_inference_error_passes_through_without_disabling(sample_log: str) -> None:
    backend = FakeBackend(raises=True)
    comp = WinnowCompressor(backend)
    assert comp.compress(_inp(sample_log)).compressed is False
    assert comp.compress(_inp(sample_log)).compressed is False
    assert len(backend.calls) == 2


def test_never_raises_on_bad_backend_output(sample_log: str) -> None:
    class Broken:
        def find_spans(self, query: str, content: str) -> list[tuple[int, int]]:
            return [("a", "b")]  # type: ignore[list-item]

    out = WinnowCompressor(Broken()).compress(_inp(sample_log))
    assert out.compressed is False
    assert out.content == sample_log


def test_settings_are_respected(sample_log: str) -> None:
    comp = WinnowCompressor(FakeBackend(("test_login_redirect",)), WinnowSettings(min_lines=10_000))
    assert comp.compress(_inp(sample_log)).compressed is False


def test_over_budget_passes_through_without_calling_the_model(sample_log: str) -> None:
    backend = FakeBackend(("test_login_redirect",))
    comp = WinnowCompressor(backend, WinnowSettings(max_tokens=50))
    out = comp.compress(_inp(sample_log))
    assert out.compressed is False
    assert out.content == sample_log
    assert backend.calls == []


def test_budget_comes_from_backend_when_not_set(sample_log: str) -> None:
    backend = FakeBackend(("test_login_redirect",))
    backend.max_input_tokens = 50  # type: ignore[attr-defined]
    assert WinnowCompressor(backend).compress(_inp(sample_log)).compressed is False
    assert backend.calls == []

    backend.max_input_tokens = 100_000  # type: ignore[attr-defined]
    assert WinnowCompressor(backend).compress(_inp(sample_log)).compressed is True


def test_settings_budget_overrides_backend(sample_log: str) -> None:
    backend = FakeBackend(("test_login_redirect",))
    backend.max_input_tokens = 50  # type: ignore[attr-defined]
    comp = WinnowCompressor(backend, WinnowSettings(max_tokens=100_000))
    assert comp.compress(_inp(sample_log)).compressed is True
