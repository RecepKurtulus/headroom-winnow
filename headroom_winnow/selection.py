"""Turn model spans into a keep/drop plan over lines, then render it.

The model scores characters; agents read lines. This module is the bridge,
and it is deliberately biased toward keeping: dropping a line the agent needed
forces it to re-run a command (or fetch the marker), which costs far more than
the few tokens a spare line costs. So, whatever the model says:

  * lines that carry failure signal (errors, tracebacks, exit codes, pytest
    ``E`` assertion lines) are always kept, plus the body of every Python
    traceback;
  * every kept line pulls in ``context`` neighbours on each side;
  * the first and last ``edge`` lines are kept (command banner, summary line).

Rendering preserves the original bytes of every kept line — the split is on
``"\\n"`` only, line endings included — so kept content is byte-identical to
the input. Each maximal dropped run becomes one marker line; a run whose text
is not longer than its marker is kept instead, since replacing it would grow
the output.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .markers import content_hash, format_marker

__all__ = [
    "MANDATORY_RE",
    "PrunePlan",
    "expand_keep",
    "lines_touched",
    "mandatory_lines",
    "render",
    "split_lines",
]

#: Lines that carry failure signal and are never dropped.
MANDATORY_RE = re.compile(
    r"error|failed|traceback|panic|exception|exit code|exit status|fatal",
    re.IGNORECASE,
)

# pytest prints assertion details as ``E   <detail>``; they rarely contain any
# of the words above but are exactly what the agent needs next.
_PYTEST_DETAIL_RE = re.compile(r"^E\s{2,}\S")

_TRACEBACK_HEADER = "Traceback (most recent call last)"


@dataclass
class PrunePlan:
    """Result of :func:`render`.

    Attributes:
        content: The pruned text (kept lines verbatim, markers for dropped runs).
        recoverable: ``hash -> dropped text`` for every marker in ``content``.
        dropped_lines: Number of input lines replaced by markers.
    """

    content: str
    recoverable: dict[str, str] = field(default_factory=dict)
    dropped_lines: int = 0


def split_lines(content: str) -> list[str]:
    """Split ``content`` on ``"\\n"`` keeping line endings.

    ``"".join(split_lines(content)) == content`` always holds. Unlike
    ``str.splitlines`` this never splits on ``\\r``, form feeds or other
    Unicode separators, so character offsets from the model map onto lines
    exactly.

    Args:
        content: Raw tool output.

    Returns:
        The lines, each ending in ``"\\n"`` except possibly the last.
    """
    if not content:
        return []
    parts = content.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def lines_touched(spans: Iterable[tuple[int, int]], lines: list[str]) -> set[int]:
    """Return indices of lines that overlap any ``[start, end)`` span.

    Args:
        spans: Character spans over ``"".join(lines)``.
        lines: Output of :func:`split_lines`.

    Returns:
        The set of touched line indices.
    """
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line)

    touched: set[int] = set()
    for start, end in spans:
        if end <= start:
            continue
        first = max(0, bisect.bisect_right(starts, start) - 1)
        last = bisect.bisect_left(starts, end) - 1
        touched.update(range(first, min(last, len(lines) - 1) + 1))
    return touched


def mandatory_lines(lines: list[str]) -> set[int]:
    """Return indices of lines that must survive regardless of the model.

    A line is mandatory if it matches :data:`MANDATORY_RE` or looks like a
    pytest assertion detail. A Python traceback is kept whole: the header, its
    indented frame/source lines, and the first unindented line after them (the
    exception message).

    Args:
        lines: Output of :func:`split_lines`.

    Returns:
        The set of mandatory line indices.
    """
    keep: set[int] = set()
    in_traceback = False
    for i, line in enumerate(lines):
        if in_traceback:
            keep.add(i)
            if line.strip() and not line[0].isspace():
                in_traceback = False
            continue
        if MANDATORY_RE.search(line) or _PYTEST_DETAIL_RE.match(line):
            keep.add(i)
        if _TRACEBACK_HEADER in line:
            in_traceback = True
    return keep


def expand_keep(keep: set[int], n_lines: int, *, context: int, edge: int) -> set[int]:
    """Grow ``keep`` by ``context`` neighbours and the first/last ``edge`` lines.

    Args:
        keep: Line indices already kept.
        n_lines: Total number of lines.
        context: Neighbours kept on each side of every kept line.
        edge: Lines always kept at the start and at the end.

    Returns:
        A new set; ``keep`` is not modified.
    """
    grown: set[int] = set()
    for i in keep:
        grown.update(range(max(0, i - context), min(n_lines, i + context + 1)))
    grown.update(range(min(edge, n_lines)))
    grown.update(range(max(0, n_lines - edge), n_lines))
    return grown


def render(lines: list[str], keep: set[int]) -> PrunePlan:
    """Render kept lines verbatim and replace each dropped run with a marker.

    Args:
        lines: Output of :func:`split_lines`.
        keep: Line indices to keep.

    Returns:
        The :class:`PrunePlan`. Output is a pure function of the inputs.
    """
    out: list[str] = []
    recoverable: dict[str, str] = {}
    dropped = 0
    i = 0
    n = len(lines)
    while i < n:
        if i in keep:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < n and j not in keep:
            j += 1
        run = "".join(lines[i:j])
        ccr_hash = content_hash(run)
        marker = format_marker(ccr_hash, j - i)
        if run.endswith("\n"):
            marker += "\n"
        if len(marker) < len(run):
            out.append(marker)
            recoverable[ccr_hash] = run
            dropped += j - i
        else:
            out.append(run)
        i = j
    return PrunePlan(content="".join(out), recoverable=recoverable, dropped_lines=dropped)
