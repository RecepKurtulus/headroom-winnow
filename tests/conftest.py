"""Shared fixtures: a deterministic fake span backend and sample tool output."""

from __future__ import annotations

import pytest

from headroom_squeez.backends import BackendUnavailableError


class FakeBackend:
    """Marks every occurrence of any keyword as a span. No model, no downloads."""

    def __init__(
        self,
        keywords: tuple[str, ...] = (),
        *,
        unavailable: bool = False,
        raises: bool = False,
    ) -> None:
        self.keywords = keywords
        self.unavailable = unavailable
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    def find_spans(self, query: str, content: str) -> list[tuple[int, int]]:
        self.calls.append((query, content))
        if self.unavailable:
            raise BackendUnavailableError("no torch here")
        if self.raises:
            raise RuntimeError("inference blew up")
        spans: list[tuple[int, int]] = []
        for kw in self.keywords:
            start = content.find(kw)
            while start != -1:
                spans.append((start, start + len(kw)))
                start = content.find(kw, start + 1)
        return sorted(spans)


def make_pytest_log(n_passed: int = 80) -> str:
    """A pytest run: many passing tests, one failure with a traceback, a summary."""
    lines = ["============================= test session starts =============================="]
    lines += [f"tests/test_views.py::test_case_{i:03d} PASSED" for i in range(n_passed // 2)]
    lines += [
        "tests/test_views.py::test_login_redirect FAILED",
        "Traceback (most recent call last):",
        '  File "app/views.py", line 88, in login',
        "    return redirect(next_url)",
        "AssertionError: expected 302, got 500",
    ]
    lines += [f"tests/test_models.py::test_model_{i:03d} PASSED" for i in range(n_passed // 2)]
    lines += ["=================== 1 failed, 80 passed in 3.21s ==================="]
    return "\n".join(lines) + "\n"


@pytest.fixture
def sample_log() -> str:
    return make_pytest_log()
