"""GitHub Actions annotations, Playwright's ``github`` reporter: on a GitHub Actions runner (``GITHUB_ACTIONS=true``)
each failed test becomes an ``::error`` at the line of the test that failed, and each flaky one a ``::warning``, so
GitHub shows them on the pull request's diff and the run's summary. Off with ``--rw-github off``; ``on`` writes them
anywhere (to see what a run would annotate)."""

from __future__ import annotations

import os

from .trace import _LIBRARY, _PACKAGE


def enabled(mode: str) -> bool:
    return mode == "on" or (mode == "auto" and os.environ.get("GITHUB_ACTIONS") == "true")


def _data(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _prop(s: str) -> str:
    return _data(s).replace(":", "%3A").replace(",", "%2C")


def _where(report, rootdir: str, invoked: str) -> tuple[str, int]:
    """The file and line in the test's own code where it failed, as an absolute path: the deepest frame of the
    traceback that is not robowright's, pytest's or the standard library's; else the test function's own line.
    Traceback paths are relative to where pytest was started (``invoked``), a test's location to the rootdir."""
    tb = getattr(getattr(report, "longrepr", None), "reprtraceback", None)
    for entry in reversed(getattr(tb, "reprentries", None) or []):
        loc = getattr(entry, "reprfileloc", None)
        if loc is None:
            continue
        path = os.path.abspath(os.path.join(invoked, loc.path))
        if not path.startswith(_PACKAGE) and not path.startswith(_LIBRARY):
            return path, int(loc.lineno)
    path, line, _ = report.location
    return os.path.abspath(os.path.join(rootdir, path)), (line or 0) + 1  # pytest's line is 0-based


def _message(report) -> str:
    crash = getattr(getattr(report, "longrepr", None), "reprcrash", None)
    text = crash.message if crash is not None else str(getattr(report, "longrepr", "") or "failed")
    return text.strip()


def annotation(report, rootdir: str, invoked: str, trace: str | None = None, failed_attempt=None) -> str:
    """One workflow command for a failed test's report, or for a flaky one's (``failed_attempt``: the report of the
    attempt that failed, whose line it points at). Its file is relative to the repository's root
    (``GITHUB_WORKSPACE``, a checkout's working directory), as GitHub matches it against the diff."""
    flaky = failed_attempt is not None
    path, line = _where(failed_attempt or report, rootdir, invoked)
    path = os.path.relpath(path, os.environ.get("GITHUB_WORKSPACE", invoked))
    kind, title = ("warning", f"flaky: {report.nodeid}") if flaky else ("error", report.nodeid)
    body = f"passed on a retry after failing: {_message(failed_attempt)}" if flaky else _message(report)
    if trace:
        body += f"\n\nrobowright show-trace {trace}"
    return f"::{kind} file={_prop(path)},line={line},title={_prop(title)}::{_data(body)}"
