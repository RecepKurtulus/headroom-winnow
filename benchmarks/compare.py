"""Gate 3: does headroom-winnow beat Headroom's own ``relevance_split``?

Runs every method on the same split of ``KRLabsOrg/tool-output-extraction-swebench``
(query + raw tool output + gold relevant lines) and reports, per method:

  * span precision / recall / F1 over gold lines — Squeez's own metric
    (``squeez/training/evaluate.py:compute_span_metrics``: sets of stripped,
    non-empty lines), so numbers are comparable with the Squeez README table;
  * line compression — Squeez's ``1 - kept_lines / total_lines``;
  * token reduction — Headroom's ``EstimatingTokenCounter``, markers included;
  * mandatory-line losses — failure/traceback lines dropped (target: 0);
  * CPU latency p50 / p95, overall and by output size.

Two views, because they answer different questions:

  **Natural operating point.** Each method runs as it would in production:
  ``relevance_split`` with Headroom's defaults (adaptive Otsu cut floored at
  0.25) and ``headroom-winnow`` through ``WinnowCompressor.compress`` with all
  its gates and safety rules. ``squeez-raw`` is the bare model (lines touched
  by a span), i.e. Squeez's own ``verbatim_v2`` baseline. For the
  ``relevance_split`` rows the dropped tail is simply dropped; in Headroom it
  would be Kompressed, so their token reduction here is an upper bound.

  **Fixed budget.** Each method ranks lines by its own per-line score and keeps
  the top ``(1 - r)`` share for ``r`` in 50/70/90%. Same compression, so recall
  compares ranking quality head to head — this is the gate.

Usage::

    python benchmarks/compare.py --limit 50          # smoke run
    python benchmarks/compare.py                     # full test split (618)
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headroom.relevance import BM25Scorer, HybridScorer
from headroom.relevance.base import RelevanceScorer
from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.transforms.compressor_registry import CompressInput
from headroom.transforms.relevance_split import plan_relevance_split

from headroom_winnow.backends import HighlighterBackend, PooledBackend, lines_to_spans
from headroom_winnow.compressor import WinnowCompressor, WinnowSettings
from headroom_winnow.selection import mandatory_lines, split_lines

DATASET = "KRLabsOrg/tool-output-extraction-swebench"
RATIOS = (0.5, 0.7, 0.9)
#: Headroom's default ``relevance.relevance_threshold`` (headroom/config.py).
RS_THRESHOLD = 0.25
SIZE_BUCKETS = ((200, "<=200"), (2000, "<=2000"), (math.inf, ">2000"))

_TOKENS = EstimatingTokenCounter()
_OFFLOADED_RE = re.compile(r"<<ccr:[a-f0-9]{12,24} (\d+)_lines_offloaded>>")


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    """One benchmark example."""

    query: str
    tool_output: str
    gold: list[str]
    tool_type: str


def _parse_gold(response: str) -> list[str]:
    """Squeez's ``_parse_relevant_lines`` for the XML/raw-text case."""
    text = response.strip()
    m = re.search(r"<relevant_lines>\s*\n?(.*?)\n?\s*</relevant_lines>", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return [line.strip() for line in text.split("\n") if line.strip()]


def load_samples(split: str, limit: int | None) -> list[Sample]:
    """Load and parse ``split`` the way Squeez's evaluator does."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    samples: list[Sample] = []
    for row in ds:
        prompt = row["prompt"]
        q = re.search(r"<query>\n(.*?)\n</query>", prompt, re.DOTALL)
        out = re.search(r"<tool_output>\n(.*?)\n</tool_output>", prompt, re.DOTALL)
        if not q or not out or not out.group(1).strip():
            continue
        samples.append(
            Sample(
                query=q.group(1),
                tool_output=out.group(1),
                gold=_parse_gold(row["response"]),
                tool_type=row["metadata"]["tool_type"],
            )
        )
        if limit and len(samples) >= limit:
            break
    return samples


# ── metrics ──────────────────────────────────────────────────────────────────


def span_metrics(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    """Squeez's ``compute_span_metrics``: (precision, recall, f1) on line sets."""
    pred_set = {p.strip() for p in pred if p.strip()}
    gold_set = set(gold)
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    p, r = tp / len(pred_set), tp / len(gold_set)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


@dataclass
class Result:
    """What one method did to one sample."""

    kept: set[int]
    output: str
    seconds: float = 0.0


@dataclass
class Row:
    """Per-sample metrics for one method."""

    precision: float
    recall: float
    f1: float
    line_compression: float
    token_reduction: float
    mandatory_lost: int
    seconds: float
    n_lines: int


def score_result(sample: Sample, lines: list[str], res: Result) -> Row:
    """Turn a :class:`Result` into metrics."""
    kept_lines = [lines[i].rstrip("\n") for i in sorted(res.kept)]
    p, r, f = span_metrics(kept_lines, sample.gold)
    kept_text = "\n".join(kept_lines)
    compression = 1.0 - len(kept_text.split("\n")) / len(sample.tool_output.split("\n"))
    before = _TOKENS.count_text(sample.tool_output)
    after = _TOKENS.count_text(res.output)
    lost = sum(1 for i in mandatory_lines(lines) if i not in res.kept and lines[i].strip())
    return Row(p, r, f, compression, 1.0 - after / max(1, before), lost, res.seconds, len(lines))


# ── squeez model: one forward pass → spans (as process()) + per-line scores ──


class PooledRunner:
    """Runs a pooled line classifier once per sample: kept-line spans + line scores."""

    def __init__(self, backend: PooledBackend) -> None:
        self.backend = backend

    def run(
        self, query: str, content: str, lines: list[str]
    ) -> tuple[list[tuple[int, int]], list[float]]:
        probs = self.backend.line_probabilities(query, content)
        # The model splits on every newline; split_lines drops a trailing
        # empty line, so trim to align.
        return lines_to_spans(content, probs, self.backend.threshold), probs[: len(lines)]


class SqueezRunner:
    """Runs the highlighter once per sample and derives both views from it.

    Mirrors ``VerbatimRagHighlighter.process`` (revision pinned in
    ``headroom_winnow.backends``) step for step, but also keeps the per-token
    probabilities so lines can be ranked for the fixed-budget view without a
    second forward pass.
    """

    def __init__(self, backend: HighlighterBackend) -> None:
        self.backend = backend

    def run(
        self, query: str, content: str, lines: list[str]
    ) -> tuple[list[tuple[int, int]], list[float]]:
        import torch

        b = self.backend
        model = b._load()
        enc = model._tokenizer(
            query,
            content,
            return_offsets_mapping=True,
            max_length=b.max_length,
            truncation="only_second",
            stride=b.doc_stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        device = b.resolved_device()
        with torch.inference_mode():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits.float()
        positive = torch.softmax(logits, dim=-1)[..., 1:].sum(dim=-1).cpu()

        starts, pos = [], 0
        for line in lines:
            starts.append(pos)
            pos += len(line)
        line_scores = [0.0] * len(lines)

        raw: list[tuple[int, int, float]] = []
        for w in range(enc["input_ids"].size(0)):
            seq_ids = enc.sequence_ids(w)
            offsets = enc["offset_mapping"][w].tolist()
            cur: list[Any] | None = None
            for sid, (s, e), p in zip(seq_ids, offsets, positive[w].tolist()):
                is_ctx = sid == 1 and s != e
                if is_ctx:
                    li = bisect.bisect_right(starts, s) - 1
                    line_scores[li] = max(line_scores[li], p)
                if is_ctx and p >= b.threshold:
                    cur = [s, e, p] if cur is None else [cur[0], e, max(cur[2], p)]
                elif cur is not None:
                    raw.append((cur[0], cur[1], cur[2]))
                    cur = None
            if cur is not None:
                raw.append((cur[0], cur[1], cur[2]))

        raw.sort()
        merged: list[list[Any]] = []
        for s, e, p in raw:
            if merged and s - merged[-1][1] <= b.merge_gap_chars:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e, p])
        spans = [(s, e) for s, e, _ in merged if e - s >= b.min_span_chars]
        return spans, line_scores


class _FixedSpans:
    """Span backend that returns precomputed spans (so the model runs once)."""

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        self.spans = spans
        self.called = False

    def find_spans(self, query: str, content: str) -> list[tuple[int, int]]:
        self.called = True
        return self.spans


def _kept_from_output(output: str, n_lines: int) -> set[int]:
    """Recover kept input line indices by walking the compressor's output."""
    kept: set[int] = set()
    i = 0
    for line in output.splitlines():
        m = _OFFLOADED_RE.fullmatch(line.strip())
        if m:
            i += int(m.group(1))
        else:
            kept.add(i)
            i += 1
    return {k for k in kept if k < n_lines}


# ── relevance_split ──────────────────────────────────────────────────────────


def relevance_split_kept(content: str, query: str, scorer: RelevanceScorer) -> set[int]:
    """Line indices inside KEEP runs of Headroom's planner with its defaults."""
    runs = plan_relevance_split(content, query, scorer, threshold=RS_THRESHOLD, adaptive=True)
    kept: set[int] = set()
    line = 0
    for keep, text in runs:
        n = len(split_lines(text))
        if keep:
            kept.update(range(line, line + n))
        line += n
    return kept


def line_scores(scorer: RelevanceScorer, lines: list[str], query: str) -> list[float]:
    return [s.score for s in scorer.score_batch([ln.rstrip("\n") for ln in lines], query)]


def top_k(scores: list[float], ratio: float) -> set[int]:
    """Keep the top ``1 - ratio`` share of lines by score (ties: earlier line)."""
    k = max(1, round(len(scores) * (1 - ratio)))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return set(order[:k])


# ── driver ───────────────────────────────────────────────────────────────────


@dataclass
class Collector:
    rows: dict[str, list[Row]] = field(default_factory=dict)

    def add(self, method: str, row: Row) -> None:
        self.rows.setdefault(method, []).append(row)


def _timed(fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def run(
    samples: list[Sample], methods: set[str], backend: HighlighterBackend | PooledBackend
) -> Collector:
    col = Collector()
    bm25 = BM25Scorer()
    hybrid: RelevanceScorer | None = None
    if "rs-hybrid" in methods:
        from headroom.relevance.embedding import embedding_available

        if not embedding_available():
            sys.exit(
                "rs-hybrid needs fastembed (pip install fastembed); or pass --methods without it"
            )
        hybrid = HybridScorer()
    runner: SqueezRunner | PooledRunner | None = None
    if methods & {"squeez-raw", "headroom-winnow"}:
        runner = (
            SqueezRunner(backend)
            if isinstance(backend, HighlighterBackend)
            else PooledRunner(backend)
        )

    for n, s in enumerate(samples, 1):
        lines = split_lines(s.tool_output)

        def keep_text(kept: set[int], lines: list[str] = lines) -> str:
            return "".join(lines[i] for i in sorted(kept))

        for name, scorer in (("rs-bm25", bm25), ("rs-hybrid", hybrid)):
            if name not in methods or scorer is None:
                continue
            kept, sec = _timed(relevance_split_kept, s.tool_output, s.query, scorer)
            col.add(name, score_result(s, lines, Result(kept, keep_text(kept), sec)))
            scores = line_scores(scorer, lines, s.query)
            for r in RATIOS:
                kept = top_k(scores, r)
                col.add(
                    f"{name}@{int(r * 100)}", score_result(s, lines, Result(kept, keep_text(kept)))
                )

        if runner is not None:
            (spans, sq_scores), fwd = _timed(runner.run, s.query, s.tool_output, lines)
            if "squeez-raw" in methods:
                from headroom_winnow.selection import lines_touched

                kept = lines_touched(spans, lines)
                col.add("squeez-raw", score_result(s, lines, Result(kept, keep_text(kept), fwd)))
            if "headroom-winnow" in methods:
                # As shipped (token budget for this device), and unbounded to
                # show what the budget costs in recall and compression.
                for name, budget in (
                    ("headroom-winnow", backend.max_input_tokens),
                    ("headroom-winnow-unbounded", 10**9),
                ):
                    fixed = _FixedSpans(spans)
                    comp = WinnowCompressor(fixed, WinnowSettings(max_tokens=budget))
                    out, sel = _timed(
                        comp.compress, CompressInput(s.tool_output, "text/plain", s.query)
                    )
                    kept = (
                        _kept_from_output(out.content, len(lines))
                        if out.compressed
                        else set(range(len(lines)))
                    )
                    # The model only costs time when the compressor consulted it.
                    latency = sel + (fwd if fixed.called else 0.0)
                    col.add(name, score_result(s, lines, Result(kept, out.content, latency)))
            for r in RATIOS:
                kept = top_k(sq_scores, r)
                col.add(
                    f"squeez@{int(r * 100)}", score_result(s, lines, Result(kept, keep_text(kept)))
                )

        if n % 25 == 0 or n == len(samples):
            print(f"  {n}/{len(samples)}", file=sys.stderr, flush=True)
    return col


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(col: Collector) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method, rows in col.rows.items():
        summary: dict[str, Any] = {
            "n": len(rows),
            "precision": statistics.mean(r.precision for r in rows),
            "recall": statistics.mean(r.recall for r in rows),
            "f1": statistics.mean(r.f1 for r in rows),
            "line_compression": statistics.mean(r.line_compression for r in rows),
            "token_reduction": statistics.mean(r.token_reduction for r in rows),
            "mandatory_lost": sum(r.mandatory_lost for r in rows),
        }
        secs = [r.seconds for r in rows if r.seconds]
        if secs:
            summary["p50_ms"] = _pct(secs, 0.5) * 1000
            summary["p95_ms"] = _pct(secs, 0.95) * 1000
            for limit, label in SIZE_BUCKETS:
                lo = max((b for b, _ in SIZE_BUCKETS if b < limit), default=0)
                bucket = [r.seconds for r in rows if r.seconds and lo < r.n_lines <= limit]
                if bucket:
                    summary[f"p50_ms_{label}"] = _pct(bucket, 0.5) * 1000
        out[method] = summary
    return out


def to_markdown(summary: dict[str, dict[str, Any]]) -> str:
    head = "| method | recall | precision | F1 | line compr. | token red. | mandatory lost | p50 ms | p95 ms |"
    lines = [head, "|" + "---|" * 9]
    for method in sorted(summary):
        s = summary[method]
        ms = (f"{s['p50_ms']:.0f}", f"{s['p95_ms']:.0f}") if "p50_ms" in s else ("", "")
        lines.append(
            f"| {method} | {s['recall']:.3f} | {s['precision']:.3f} | {s['f1']:.3f} "
            f"| {s['line_compression']:.3f} | {s['token_reduction']:.3f} "
            f"| {s['mandatory_lost']} | {ms[0]} | {ms[1]} |"
        )
    lines.append("")
    lines.append("Gate 3 (fixed budget, recall): ")
    for r in RATIOS:
        tag = int(r * 100)
        sq = summary.get(f"squeez@{tag}", {}).get("recall")
        rivals = {
            m: summary[f"{m}@{tag}"]["recall"]
            for m in ("rs-bm25", "rs-hybrid")
            if f"{m}@{tag}" in summary
        }
        if sq is None or not rivals:
            continue
        best = max(rivals, key=lambda m: rivals[m])
        verdict = "PASS" if sq > rivals[best] else "FAIL"
        lines.append(f"- {tag}%: squeez {sq:.3f} vs {best} {rivals[best]:.3f} -> {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--methods",
        default="rs-bm25,rs-hybrid,squeez-raw,headroom-winnow",
        help="comma-separated subset of rs-bm25, rs-hybrid, squeez-raw, headroom-winnow",
    )
    ap.add_argument("--max-length", type=int, default=8192, help="model token window")
    ap.add_argument("--stride", type=int, default=256, help="token overlap between windows")
    ap.add_argument("--device", default=None, help="cpu, cuda, or auto (default: backend's)")
    ap.add_argument(
        "--backend", default="highlighter", choices=["highlighter", "pooled"], help="squeez model"
    )
    ap.add_argument("--model-path", default=None, help="pooled model dir or hub id")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args(argv)

    samples = load_samples(args.split, args.limit)
    print(f"{len(samples)} samples from {DATASET}:{args.split}", file=sys.stderr)
    backend: HighlighterBackend | PooledBackend
    if args.backend == "pooled":
        backend = PooledBackend(args.model_path, device=args.device)
    else:
        backend = HighlighterBackend(
            device=args.device, max_length=args.max_length, doc_stride=args.stride
        )
    col = run(samples, set(args.methods.split(",")), backend)
    summary = summarize(col)
    table = to_markdown(summary)
    print(table)

    args.out.mkdir(parents=True, exist_ok=True)
    model_tag = f"L{args.max_length}" if args.backend == "highlighter" else "pooled"
    stem = f"{args.split}-{len(samples)}-{model_tag}-{backend.resolved_device()}"
    (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / f"{stem}.md").write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
