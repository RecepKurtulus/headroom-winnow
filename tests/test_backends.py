"""Backend configuration that needs no model download."""

from __future__ import annotations

import pytest

from headroom_squeez.backends import (
    CPU_TOKEN_BUDGET,
    CUDA_TOKEN_BUDGET,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    HighlighterBackend,
    PooledBackend,
    lines_to_spans,
    make_backend,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MODEL", "REVISION", "DEVICE", "DTYPE", "MAX_TOKENS"):
        monkeypatch.delenv(f"HEADROOM_SQUEEZ_{name}", raising=False)


def test_defaults_are_pinned_fp32_auto() -> None:
    b = HighlighterBackend()
    assert (b.model_id, b.revision) == (DEFAULT_MODEL, DEFAULT_REVISION)
    assert (b.device, b.dtype) == ("auto", "float32")


def test_custom_model_without_revision_is_unpinned() -> None:
    assert HighlighterBackend("someone/other-model").revision is None


@pytest.mark.parametrize(
    ("device", "budget"), [("cpu", CPU_TOKEN_BUDGET), ("cuda", CUDA_TOKEN_BUDGET)]
)
def test_budget_follows_device(device: str, budget: int) -> None:
    assert HighlighterBackend(device=device).max_input_tokens == budget


def test_budget_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_SQUEEZ_MAX_TOKENS", "4096")
    assert HighlighterBackend(device="cpu").max_input_tokens == 4096
    assert HighlighterBackend(device="cpu", max_input_tokens=100).max_input_tokens == 100
    monkeypatch.setenv("HEADROOM_SQUEEZ_MAX_TOKENS", "lots")
    assert HighlighterBackend(device="cpu").max_input_tokens == CPU_TOKEN_BUDGET


def test_env_configures_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_SQUEEZ_DEVICE", "cpu")
    monkeypatch.setenv("HEADROOM_SQUEEZ_DTYPE", "float16")
    b = HighlighterBackend()
    assert (b.resolved_device(), b.dtype) == ("cpu", "float16")


def test_lines_to_spans_covers_kept_lines_only() -> None:
    content = "keep me\nskip\n\nkeep too\n"
    spans = lines_to_spans(content, [0.9, 0.1, 0.9, 0.5, 0.0], threshold=0.5)
    assert [content[s:e] for s, e in spans] == ["keep me", "keep too"]


def test_make_backend_selects_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(make_backend(), HighlighterBackend)
    monkeypatch.setenv("HEADROOM_SQUEEZ_BACKEND", "pooled")
    monkeypatch.setenv("HEADROOM_SQUEEZ_MODEL", "some/dir")
    backend = make_backend()
    assert isinstance(backend, PooledBackend)
    assert backend.model_id == "some/dir"
    assert backend._model is None
    monkeypatch.setenv("HEADROOM_SQUEEZ_BACKEND", "nonsense")
    assert isinstance(make_backend(), HighlighterBackend)


def test_pooled_needs_a_model() -> None:
    with pytest.raises(ValueError, match="HEADROOM_SQUEEZ_MODEL"):
        PooledBackend()
