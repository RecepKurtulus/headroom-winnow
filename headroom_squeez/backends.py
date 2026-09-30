"""Span-model backends: lazy, pinned, and isolated from the request path.

The compressor only needs one thing from a model — character spans of
``content`` that matter for ``query`` — so backends implement the tiny
:class:`SpanBackend` protocol. That keeps the pruning logic testable with a
fake backend and lets models be swapped without touching the compressor.

Two backends ship, selected by ``$HEADROOM_SQUEEZ_BACKEND`` (see
:func:`make_backend`):

  * :class:`HighlighterBackend` (default) wraps
    ``KRLabsOrg/verbatim-rag-modern-bert-v2``, the ~150M ModernBERT span model
    Squeez itself uses for its extractive path.
  * :class:`PooledBackend` loads a Squeez *pooled* line classifier, which can
    be trained on a much smaller encoder (``training/``).

Three rules shape both:

  * **Lazy.** Headroom instantiates entry-point classes during discovery, for
    every process that builds a router, selected or not. Loading a model there
    would tax every Headroom user who merely has this package installed, so
    nothing heavy happens until the first ``find_spans``.
  * **Pinned.** The models ship their own code (``trust_remote_code=True``), so
    weights and tokenizer load at a fixed commit. The highlighter's
    ``process()`` would otherwise fetch its tokenizer from the moving ``main``
    branch, so we load it ourselves at the same revision and hand it over.
  * **Loud once, then quiet.** Any load failure (no torch, no network, bad
    revision) is raised as :class:`BackendUnavailableError`; the compressor
    turns that into a single warning and passes through for the rest of the
    process.

Long outputs need no chunking here: both models window long inputs themselves
and map results back onto ``content``. Shrinking the highlighter's window to go
faster is not an option: on the Squeez test split, gold-line recall at 90%
compression falls from 0.78 (8192) to 0.47 (2048) and 0.31 (512), because the
model needs the whole output in view.

What bounds latency instead is ``max_input_tokens``: the largest input a
backend can score within the latency budget on its device. Measured highlighter
forward times (fp32, ModernBERT-base):

    tokens      GTX 1650 Ti     4-core laptop CPU
    512         70 ms           660 ms
    2048        490 ms          3.7 s
    8192        4.2 s           31 s

so the defaults are :data:`CUDA_TOKEN_BUDGET` and :data:`CPU_TOKEN_BUDGET`.
The compressor passes anything larger straight to Headroom's own path.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

__all__ = [
    "CPU_TOKEN_BUDGET",
    "CUDA_TOKEN_BUDGET",
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "BackendUnavailableError",
    "HighlighterBackend",
    "PooledBackend",
    "SpanBackend",
    "lines_to_spans",
    "make_backend",
]

DEFAULT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"
#: Commit of :data:`DEFAULT_MODEL` whose weights and remote code we reviewed.
DEFAULT_REVISION = "6a967332efedfe5aca9b85b8310cc68d9ac6f881"

_ENV_BACKEND = "HEADROOM_SQUEEZ_BACKEND"
_ENV_MODEL = "HEADROOM_SQUEEZ_MODEL"
_ENV_REVISION = "HEADROOM_SQUEEZ_REVISION"
_ENV_DEVICE = "HEADROOM_SQUEEZ_DEVICE"
_ENV_DTYPE = "HEADROOM_SQUEEZ_DTYPE"
_ENV_MAX_TOKENS = "HEADROOM_SQUEEZ_MAX_TOKENS"

#: Largest input scored on a GPU (~0.5 s on a GTX 1650 Ti).
CUDA_TOKEN_BUDGET = 2048
#: Largest input scored on CPU (~0.7 s on a 4-core laptop).
CPU_TOKEN_BUDGET = 512


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


class _TransformersBackend:
    """Shared plumbing: lazy load, device/dtype resolution, token budget.

    Subclasses say how the model is loaded (:meth:`_load_model`) and how its
    output becomes spans (``find_spans``).
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        *,
        device: str | None,
        dtype: str | None,
        max_input_tokens: int | None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device or os.environ.get(_ENV_DEVICE) or "auto"
        self.dtype = dtype or os.environ.get(_ENV_DTYPE) or "float32"
        self._max_input_tokens = max_input_tokens
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

    @property
    def max_input_tokens(self) -> int:
        """Largest input (in tokens) this backend scores within its latency budget."""
        if self._max_input_tokens is not None:
            return self._max_input_tokens
        env = os.environ.get(_ENV_MAX_TOKENS)
        if env:
            try:
                return int(env)
            except ValueError:
                log.warning("ignoring non-integer %s=%r", _ENV_MAX_TOKENS, env)
        return CUDA_TOKEN_BUDGET if self.resolved_device().startswith("cuda") else CPU_TOKEN_BUDGET

    def _load_model(self, dtype: Any) -> Any:
        """Load and return the model on CPU; the caller moves it to the device."""
        raise NotImplementedError

    def _load(self) -> Any:
        """Load (once) and return the model; raise ``BackendUnavailableError`` on failure."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            device = self.resolved_device()
            try:
                import torch

                model = self._load_model(getattr(torch, self.dtype))
                model.to(device)
                model.eval()
            except Exception as exc:  # noqa: BLE001 - any load failure disables the backend
                raise BackendUnavailableError(
                    f"cannot load {self.model_id}@{self.revision or 'main'}: {exc}"
                ) from exc
            log.debug("loaded %s@%s on %s", self.model_id, self.revision, device)
            self._model = model
            return model


class HighlighterBackend(_TransformersBackend):
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
        max_input_tokens: Largest input to score. Defaults to
            ``$HEADROOM_SQUEEZ_MAX_TOKENS``, else :data:`CUDA_TOKEN_BUDGET` or
            :data:`CPU_TOKEN_BUDGET` for the resolved device.
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
        max_input_tokens: int | None = None,
    ) -> None:
        model_id = model_id or os.environ.get(_ENV_MODEL) or DEFAULT_MODEL
        if revision is None:
            revision = os.environ.get(_ENV_REVISION) or (
                DEFAULT_REVISION if model_id == DEFAULT_MODEL else None
            )
        super().__init__(
            model_id, revision, device=device, dtype=dtype, max_input_tokens=max_input_tokens
        )
        self.threshold = threshold
        self.min_span_chars = min_span_chars
        self.merge_gap_chars = merge_gap_chars
        self.max_length = max_length
        self.doc_stride = doc_stride

    def _load_model(self, dtype: Any) -> Any:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True, dtype=dtype
        )
        # process() lazily loads its tokenizer by name at ``main``; pre-seed it
        # so tokenizer and weights come from one commit.
        model._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, use_fast=True
        )
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


class PooledBackend(_TransformersBackend):
    """Squeez pooled line classifier (``squeez.encoder.train --classifier-type pooled``).

    The model scores whole lines, so each kept line becomes one span covering
    it. It is trained on the same data as the highlighter but can sit on a much
    smaller encoder (e.g. ``jhu-clsp/ettin-encoder-32m``), which is what lets
    the token budget grow.

    Args:
        model_id: Local directory or Hugging Face repo with the trained model
            (it ships ``modeling_squeez_pooled.py`` for ``trust_remote_code``).
            Defaults to ``$HEADROOM_SQUEEZ_MODEL``.
        revision: Commit to load for a hub repo. Defaults to
            ``$HEADROOM_SQUEEZ_REVISION``.
        device: As for :class:`HighlighterBackend`.
        dtype: As for :class:`HighlighterBackend`.
        threshold: Line probability at or above which a line is kept.
        max_input_tokens: As for :class:`HighlighterBackend`.

    Raises:
        ValueError: If no model is given and ``$HEADROOM_SQUEEZ_MODEL`` is unset.
    """

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        *,
        device: str | None = None,
        dtype: str | None = None,
        threshold: float = 0.5,
        max_input_tokens: int | None = None,
    ) -> None:
        model_id = model_id or os.environ.get(_ENV_MODEL)
        if not model_id:
            raise ValueError(f"PooledBackend needs a model path or ${_ENV_MODEL}")
        super().__init__(
            model_id,
            revision or os.environ.get(_ENV_REVISION) or None,
            device=device,
            dtype=dtype,
            max_input_tokens=max_input_tokens,
        )
        self.threshold = threshold
        self._tokenizer: Any = None

    def _load_model(self, dtype: Any) -> Any:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True, dtype=dtype
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        return model

    def line_probabilities(self, query: str, content: str) -> list[float]:
        """Return per-line relevance over ``content.split("\\n")``."""
        model = self._load()
        result = model.process(
            task=query,
            tool_output=content,
            tokenizer=self._tokenizer,
            threshold=self.threshold,
            return_line_probabilities=True,
        )
        return [float(p) for p in result.get("line_probabilities") or []]

    def find_spans(self, query: str, content: str) -> list[tuple[int, int]]:
        """Return one span per line the classifier keeps."""
        return lines_to_spans(content, self.line_probabilities(query, content), self.threshold)


def lines_to_spans(content: str, probs: list[float], threshold: float) -> list[tuple[int, int]]:
    """Map per-line probabilities over ``content.split("\\n")`` to character spans.

    Args:
        content: The scored text.
        probs: One probability per ``"\\n"``-separated line.
        threshold: Lines at or above it (and not blank) become spans.

    Returns:
        ``[start, end)`` spans, one per kept line, newline excluded.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for line, p in zip(content.split("\n"), probs):
        if p >= threshold and line.strip():
            spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def make_backend() -> SpanBackend:
    """Build the backend named by ``$HEADROOM_SQUEEZ_BACKEND`` (default ``highlighter``).

    Constructing a backend loads nothing, so this is safe during discovery.
    """
    name = (os.environ.get(_ENV_BACKEND) or "highlighter").strip().lower()
    if name == "pooled":
        return PooledBackend()
    if name != "highlighter":
        log.warning("unknown %s=%r; using highlighter", _ENV_BACKEND, name)
    return HighlighterBackend()
