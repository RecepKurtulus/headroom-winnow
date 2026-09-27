"""Span-model backends: lazy, pinned, and isolated from the request path.

The compressor only needs one thing from a model — character spans of
``content`` that matter for ``query`` — so backends implement the tiny
:class:`SpanBackend` protocol. That keeps the pruning logic testable with a
fake backend and leaves room for a second backend (e.g. ``squeez-2b`` behind a
local vLLM server) without touching the compressor.

The default :class:`HighlighterBackend` wraps ``KRLabsOrg/verbatim-rag-modern-bert-v2``,
the ~150M ModernBERT span model Squeez itself uses for its extractive path.
Three rules shape it:

  * **Lazy.** Headroom instantiates entry-point classes during discovery, for
    every process that builds a router, selected or not. Loading a model there
    would tax every Headroom user who merely has this package installed, so
    nothing heavy happens until the first :meth:`HighlighterBackend.find_spans`.
  * **Pinned.** The model ships its own code (``trust_remote_code=True``), so
    both the weights and the tokenizer are loaded at a fixed commit. The model's
    ``process()`` would otherwise fetch the tokenizer from the moving ``main``
    branch, so we load it ourselves at the same revision and hand it over.
  * **Loud once, then quiet.** Any load failure (no torch, no network, bad
    revision) is raised as :class:`BackendUnavailableError`; the compressor
    turns that into a single warning and passes through for the rest of the
    process.

Long outputs need no chunking here: ``process()`` tokenizes with overflowing
windows (``max_length`` tokens, ``doc_stride`` overlap) and maps every window
back to character offsets in the original ``content``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "BackendUnavailableError",
    "HighlighterBackend",
    "SpanBackend",
]

DEFAULT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"
#: Commit of :data:`DEFAULT_MODEL` whose weights and remote code we reviewed.
DEFAULT_REVISION = "6a967332efedfe5aca9b85b8310cc68d9ac6f881"

_ENV_MODEL = "HEADROOM_SQUEEZ_MODEL"
_ENV_REVISION = "HEADROOM_SQUEEZ_REVISION"
_ENV_DEVICE = "HEADROOM_SQUEEZ_DEVICE"
_ENV_DTYPE = "HEADROOM_SQUEEZ_DTYPE"


class BackendUnavailableError(RuntimeError):
    """The backend cannot run in this process (missing deps, download failed, ...)."""


@runtime_checkable
class SpanBackend(Protocol):
    """Anything that can point at the parts of ``content`` relevant to ``query``."""

    def find_spans(self, query: str, content: str) -> list[tuple[int, int]]:
        """Return ``[start, end)`` character spans of ``content`` relevant to ``query``.

        Raises:
            BackendUnavailableError: If the backend can never run in this process.
        """
        ...


class HighlighterBackend:
    """Verbatim-RAG ModernBERT highlighter, loaded on first use.

    Args:
        model_id: Hugging Face repo id. Defaults to ``$HEADROOM_SQUEEZ_MODEL``
            or :data:`DEFAULT_MODEL`.
        revision: Commit to load. Defaults to ``$HEADROOM_SQUEEZ_REVISION``, or
            :data:`DEFAULT_REVISION` when the default model is used. A custom
            model without a revision loads ``main``.
        device: ``"cpu"``, ``"cuda"``, ... or ``"auto"`` (CUDA when available,
            else CPU). Defaults to ``$HEADROOM_SQUEEZ_DEVICE`` or ``"auto"``.
        dtype: ``"float32"`` or ``"float16"``. Defaults to ``$HEADROOM_SQUEEZ_DTYPE``
            or ``"float32"``: float16 is not a safe default, since GPUs without
            tensor cores (e.g. GTX 16xx) run it several times slower than float32.
        threshold: Per-token probability for a token to join a span. Squeez's
            recall-tuned value for technical content.
        min_span_chars: Spans shorter than this are discarded by the model.
        merge_gap_chars: Spans closer than this are merged by the model.
        max_length: Token window per forward pass.
        doc_stride: Token overlap between consecutive windows.
    """

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        *,
        device: str | None = None,
        dtype: str | None = None,
        threshold: float = 0.1,
        min_span_chars: int = 10,
        merge_gap_chars: int = 20,
        max_length: int = 8192,
        doc_stride: int = 256,
    ) -> None:
        self.model_id = model_id or os.environ.get(_ENV_MODEL) or DEFAULT_MODEL
        env_revision = os.environ.get(_ENV_REVISION)
        if revision is not None:
            self.revision: str | None = revision
        elif env_revision:
            self.revision = env_revision
        else:
            self.revision = DEFAULT_REVISION if self.model_id == DEFAULT_MODEL else None
        self.device = device or os.environ.get(_ENV_DEVICE) or "auto"
        self.dtype = dtype or os.environ.get(_ENV_DTYPE) or "float32"
        self.threshold = threshold
        self.min_span_chars = min_span_chars
        self.merge_gap_chars = merge_gap_chars
        self.max_length = max_length
        self.doc_stride = doc_stride
        self._model: Any = None
        self._lock = threading.Lock()

    def resolved_device(self) -> str:
        """Return the concrete device ``"auto"`` resolves to in this process."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - no torch means the load will fail anyway
            return "cpu"

    def _load(self) -> Any:
        """Load (once) and return the model; raise ``BackendUnavailableError`` on failure."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                device = self.resolved_device()
                dtype = getattr(torch, self.dtype)
                model = AutoModel.from_pretrained(
                    self.model_id, revision=self.revision, trust_remote_code=True, dtype=dtype
                )
                # process() lazily loads its tokenizer by name at ``main``;
                # pre-seed it so tokenizer and weights come from one commit.
                model._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, revision=self.revision, use_fast=True
                )
                model.to(device)
                model.eval()
            except Exception as exc:  # noqa: BLE001 - any load failure disables the backend
                raise BackendUnavailableError(
                    f"cannot load {self.model_id}@{self.revision or 'main'}: {exc}"
                ) from exc
            log.debug("loaded %s@%s on %s", self.model_id, self.revision, device)
            self._model = model
            return model

    def find_spans(self, query: str, content: str) -> list[tuple[int, int]]:
        """Run the highlighter and return its character spans over ``content``."""
        model = self._load()
        result = model.process(
            question=query,
            context=content,
            threshold=self.threshold,
            max_length=self.max_length,
            doc_stride=self.doc_stride,
            min_span_chars=self.min_span_chars,
            merge_gap_chars=self.merge_gap_chars,
        )
        spans = result.get("spans") or []
        return [(int(sp["start"]), int(sp["end"])) for sp in spans]
