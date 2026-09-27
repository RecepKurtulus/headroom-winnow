"""End to end through Headroom's ``ContentRouter`` with ``active_external_compressors=["squeez"]``.

Uses the same isolation as Headroom's own external-dispatch tests: an
in-memory CCR store, the pure-Python content detector, and Kompress off so
the built-in fallback never loads a model.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from conftest import FakeBackend
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.transforms.compressor_registry import CompressorRegistry
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)

from headroom_squeez.compressor import SqueezCompressor
from headroom_squeez.markers import MARKER_RE

QUERY = "which worker lost the lease on shard kappa"

# Prose-like plain text: no blank runs, no repeated lines and no shared path
# prefixes, so the STAGE-0 lossless folds find nothing and the block reaches
# the external dispatch branch.
_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
_PLAIN = "".join(
    f"{_WORDS[i % 8]} worker {i} renewed lease on shard {_WORDS[(i * 3) % 8]} after {i * 13 % 97} ms\n"
    for i in range(60)
).replace(
    "hotel worker 31 renewed lease on shard foxtrot",
    "hotel worker 31 lost the lease on shard kappa",
)


@pytest.fixture(autouse=True)
def _memory_ccr(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")
    reset_compression_store()
    yield
    reset_compression_store()


def _router(comp: SqueezCompressor, selection: list[str] | None = None) -> ContentRouter:
    router = ContentRouter(
        ContentRouterConfig(
            enable_kompress=False,
            active_external_compressors=["squeez"] if selection is None else selection,
        )
    )
    router.compressor_registry.register(comp, replace=True)
    router._active_external_compressors = router._resolve_active_external_compressors()
    return router


def test_entry_point_is_discovered() -> None:
    registry = CompressorRegistry()
    assert "squeez" in registry.discover()
    assert isinstance(registry.get("squeez"), SqueezCompressor)


def test_router_uses_squeez_and_markers_are_retrievable() -> None:
    backend = FakeBackend(("lost the lease",))
    router = _router(SqueezCompressor(backend))

    compressed, _tokens, chain = router._apply_strategy_to_content(
        _PLAIN, CompressionStrategy.TEXT, QUERY
    )

    assert chain == ["external:squeez"]
    assert backend.calls and backend.calls[0][0] == QUERY
    assert "lost the lease on shard kappa" in compressed
    assert MARKER_RE.search(compressed)

    # Every marker resolves through the CCR store, and resolving them all
    # restores the original block byte for byte.
    store = get_compression_store()
    restored: list[str] = []
    for line in compressed.splitlines(keepends=True):
        m = MARKER_RE.match(line)
        if m is None:
            restored.append(line)
            continue
        entry = store.retrieve(m.group(1))
        assert entry is not None, f"marker {m.group(1)} not retrievable"
        assert entry.compression_strategy == "external:squeez"
        restored.append(entry.original_content)
    assert "".join(restored) == _PLAIN


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Headroom 0.39.1 ignores CompressOutput.compressed=False and adopts the "
        "passthrough as the result, so the built-in path never runs. Fixed by "
        "upstream/router-respect-passthrough.patch; strict so this flips when it lands."
    ),
)
def test_router_falls_back_when_backend_unavailable() -> None:
    router = _router(SqueezCompressor(FakeBackend(unavailable=True)))
    compressed, _tokens, chain = router._apply_strategy_to_content(
        _PLAIN, CompressionStrategy.TEXT, QUERY
    )
    assert "external:squeez" not in chain
    assert not MARKER_RE.search(compressed)


def test_not_selected_means_never_called() -> None:
    backend = FakeBackend(("lost the lease",))
    router = _router(SqueezCompressor(backend), selection=[])
    router._apply_strategy_to_content(_PLAIN, CompressionStrategy.TEXT, QUERY)
    assert backend.calls == []


def test_known_gap_lossless_fold_preempts_external() -> None:
    """Documents the Step-0 finding: a foldable LOG block never reaches us.

    STAGE 0 (lossless fold) returns before the external dispatch branch, so a
    log with repeated lines is folded by Headroom and SqueezCompressor is not
    called. If this test starts failing, Headroom changed the ordering and the
    gap is closed.
    """
    log = "".join(["retrying connection to db...\n"] * 30) + "".join(
        f"request {i} served in {i} ms\n" for i in range(30)
    )
    backend = FakeBackend(("retrying",))
    router = _router(SqueezCompressor(backend))
    _compressed, _tokens, chain = router._apply_strategy_to_content(
        log, CompressionStrategy.LOG, QUERY
    )
    assert "external:squeez" not in chain
    assert backend.calls == []
