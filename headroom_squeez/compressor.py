"""``SqueezCompressor``: the ``headroom.compressor`` entry point.

Headroom's content router hands a selected external compressor one block of
tool output plus a query (the user's prompt, enriched with the triggering tool
call's args), and expects a pure-data :class:`CompressOutput` back. This class
implements that contract (``headroom/transforms/compressor_registry.py``) with
a task-conditioned line pruner:

  1. **Gate** — pass through (``compressed=False``) without touching the model
     when there is no query (no task, no basis for relevance), when the block is
     too short for markers to pay off, when it is too long for the model to
     score within the latency budget (Headroom's own compressors handle it
     instead), or when the backend is known to be unavailable.
  2. **Score** — ask the span backend which characters matter for the query.
  3. **Protect** — keep failure lines, tracebacks, neighbours and edges no
     matter what the model said (:mod:`headroom_squeez.selection`).
  4. **Render** — replace each dropped run with a ``<<ccr:HASH N_lines_offloaded>>``
     marker and return ``hash -> original`` in ``recoverable`` so the router can
     persist it for retrieval.

The router is already fail-open (it falls back to its built-in path when an
external raises, returns empty output, or grows the block), but we never rely
on that: ``compress`` does not raise, and every "not worth it" outcome is an
explicit passthrough so the router's own compressors still get their turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.transforms.compressor_registry import (
    CompressInput,
    CompressorDescriptor,
    CompressOutput,
)

from .backends import BackendUnavailableError, SpanBackend, make_backend
from .selection import expand_keep, lines_touched, mandatory_lines, render, split_lines

log = logging.getLogger(__name__)

__all__ = ["CONTENT_TYPES", "COMPRESSOR_NAME", "SqueezCompressor", "SqueezSettings"]

COMPRESSOR_NAME = "squeez"

#: MIME types we accept from the router. JSON, code, HTML, CSV and config are
#: left to Headroom's structure-aware compressors.
CONTENT_TYPES: tuple[str, ...] = (
    "text/x-log",
    "text/x-search-results",
    "text/x-diff",
    "text/plain",
)

_TOKENS = EstimatingTokenCounter()


@dataclass(frozen=True)
class SqueezSettings:
    """Pruning knobs. Defaults are recall-first.

    Attributes:
        min_lines: Blocks with fewer lines pass through untouched.
        context_lines: Neighbours kept on each side of every kept line.
        edge_lines: Lines always kept at the start and at the end.
        min_savings: Minimum fraction of tokens saved; below it we pass through.
        max_tokens: Blocks larger than this (query included, Headroom's token
            estimate) pass through without running the model. ``None`` uses the
            backend's ``max_input_tokens`` when it has one, else no limit.
    """

    min_lines: int = 40
    context_lines: int = 2
    edge_lines: int = 2
    min_savings: float = 0.2
    max_tokens: int | None = None


class SqueezCompressor:
    """Task-conditioned line pruner implementing Headroom's ``Compressor`` protocol.

    Args:
        backend: Span backend. Defaults to the one
            :func:`~headroom_squeez.backends.make_backend` picks
            (``$HEADROOM_SQUEEZ_BACKEND``); constructing it loads nothing, so
            discovery stays cheap.
        settings: Pruning knobs.
    """

    def __init__(
        self,
        backend: SpanBackend | None = None,
        settings: SqueezSettings | None = None,
    ) -> None:
        self._backend: SpanBackend = backend if backend is not None else make_backend()
        self._settings = settings or SqueezSettings()
        self._disabled_reason: str | None = None

    @property
    def descriptor(self) -> CompressorDescriptor:
        """Return this compressor's static capability metadata."""
        return CompressorDescriptor(
            name=COMPRESSOR_NAME,
            content_types=list(CONTENT_TYPES),
            lossless=False,
            cost_tier="ml",
            recoverable=True,
        )

    def compress(self, inp: CompressInput) -> CompressOutput:
        """Prune ``inp.content`` against ``inp.query``; never raises.

        Args:
            inp: The block, its MIME type and the task query.

        Returns:
            A pruned result with markers and a recovery map, or a passthrough
            (``compressed=False``, original content) when pruning is skipped.
        """
        try:
            return self._compress(inp)
        except Exception as exc:  # noqa: BLE001 - never break the request; router falls back
            log.warning("squeez: unexpected failure, passing through: %s", exc)
            return self._passthrough(inp.content, f"error: {exc}")

    def _compress(self, inp: CompressInput) -> CompressOutput:
        content = inp.content
        settings = self._settings
        if self._disabled_reason is not None:
            return self._passthrough(content, self._disabled_reason)
        query = inp.query.strip()
        if not query:
            return self._passthrough(content, "no query")
        lines = split_lines(content)
        if len(lines) < settings.min_lines:
            return self._passthrough(content, "too short")
        tokens_before = _TOKENS.count_text(content)
        budget = self._token_budget()
        if budget is not None and tokens_before + _TOKENS.count_text(query) > budget:
            # Scoring this would stall the agent; Headroom's own path is the
            # better trade for blocks this large.
            return self._passthrough(content, "over token budget", tokens_before)

        try:
            spans = self._backend.find_spans(query, content)
        except BackendUnavailableError as exc:
            # One warning per process; every later block skips straight to
            # passthrough instead of retrying a download or import.
            self._disabled_reason = f"backend unavailable: {exc}"
            log.warning("squeez: disabled for this process: %s", exc)
            return self._passthrough(content, self._disabled_reason)
        except Exception as exc:  # noqa: BLE001 - a single bad inference must not break the request
            log.warning("squeez: inference failed, passing through: %s", exc)
            return self._passthrough(content, f"inference failed: {exc}")

        if not spans:
            # The model saw nothing relevant. That is more often a weak query
            # than an irrelevant block, so we do not prune on it.
            return self._passthrough(content, "no spans")

        keep = lines_touched(spans, lines) | mandatory_lines(lines)
        keep = expand_keep(
            keep, len(lines), context=settings.context_lines, edge=settings.edge_lines
        )
        plan = render(lines, keep)

        tokens_after = _TOKENS.count_text(plan.content)
        if not plan.recoverable or tokens_after > tokens_before * (1 - settings.min_savings):
            return self._passthrough(content, "savings below threshold", tokens_before)

        return CompressOutput(
            content=plan.content,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            lossless=False,
            markers=[f"{COMPRESSOR_NAME}:{plan.dropped_lines}_lines_offloaded"],
            recoverable=plan.recoverable,
            warnings=[],
            compressed=True,
        )

    def _token_budget(self) -> int | None:
        if self._settings.max_tokens is not None:
            return self._settings.max_tokens
        budget = getattr(self._backend, "max_input_tokens", None)
        return budget if isinstance(budget, int) else None

    @staticmethod
    def _passthrough(content: str, reason: str, tokens: int | None = None) -> CompressOutput:
        log.debug("squeez: passthrough (%s)", reason)
        count = tokens if tokens is not None else _TOKENS.count_text(content)
        return CompressOutput(
            content=content,
            tokens_before=count,
            tokens_after=count,
            lossless=True,
            markers=[],
            recoverable={},
            warnings=[f"passthrough: {reason}"],
            compressed=False,
        )
