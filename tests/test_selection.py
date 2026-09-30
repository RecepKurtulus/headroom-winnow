"""Line mapping, safety rules and rendering."""

from __future__ import annotations

import pytest

from headroom_winnow.markers import MARKER_RE
from headroom_winnow.selection import (
    expand_keep,
    lines_touched,
    mandatory_lines,
    render,
    split_lines,
)


@pytest.mark.parametrize(
    "content",
    ["", "a", "a\n", "a\nb", "a\n\nb\n", "a\r\nb\r\n", "x\x0cy\n z"],
)
def test_split_lines_is_lossless(content: str) -> None:
    assert "".join(split_lines(content)) == content


def test_split_lines_splits_only_on_newline() -> None:
    assert split_lines("a\rb\x0cc\n") == ["a\rb\x0cc\n"]


def test_lines_touched_maps_spans_to_lines() -> None:
    lines = split_lines("aaa\nbbb\nccc\nddd\n")
    # "bbb\n" is chars 4-8; a span inside it, and one crossing c/d.
    assert lines_touched([(5, 6)], lines) == {1}
    assert lines_touched([(10, 13)], lines) == {2, 3}
    assert lines_touched([(0, 0), (7, 5)], lines) == set()
    assert lines_touched([(0, 10_000)], lines) == {0, 1, 2, 3}


def test_mandatory_keeps_failure_lines_and_whole_traceback() -> None:
    lines = split_lines(
        "ok 1\n"
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in f\n'
        "    g()\n"
        "ZeroDivisionError: division by zero\n"
        "ok 2\n"
        "E       assert 1 == 2\n"
        "build FAILED\n"
        "process exited with exit code 2\n"
        "ok 3\n"
    )
    assert mandatory_lines(lines) == {1, 2, 3, 4, 6, 7, 8}


def test_expand_keep_adds_context_and_edges() -> None:
    assert expand_keep({10}, 30, context=2, edge=2) == {0, 1, 8, 9, 10, 11, 12, 28, 29}
    assert expand_keep(set(), 3, context=2, edge=2) == {0, 1, 2}


def test_render_keeps_bytes_and_marks_dropped_runs() -> None:
    lines = [f"line number {i} with some padding text\n" for i in range(20)]
    plan = render(lines, {0, 1, 10, 19})
    out = plan.content.splitlines(keepends=True)
    assert out[0] == lines[0] and out[1] == lines[1]
    assert MARKER_RE.match(out[2])
    assert out[3] == lines[10]
    assert MARKER_RE.match(out[4])
    assert out[5] == lines[19]
    assert plan.dropped_lines == 16
    # Every marker resolves to exactly the lines it replaced.
    restored = "".join(
        plan.recoverable[m.group(1)] if (m := MARKER_RE.match(line)) else line for line in out
    )
    assert restored == "".join(lines)


def test_render_keeps_runs_shorter_than_their_marker() -> None:
    lines = ["keep this line\n", "\n", "keep this line too\n"]
    plan = render(lines, {0, 2})
    assert plan.content == "".join(lines)
    assert plan.recoverable == {}


def test_render_is_deterministic() -> None:
    lines = [f"row {i} lorem ipsum dolor sit amet\n" for i in range(50)]
    keep = {0, 25, 49}
    assert render(lines, keep) == render(lines, keep)
