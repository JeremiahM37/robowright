"""Named steps: ``with rw.step("pick the cube"): ...``, Playwright's ``test.step``.

A step groups what a test does under a name a person reads: the trace records it with its start
and end (the text timeline indents what ran inside it, steps inside steps too), and an error
raised inside it says which step it was in.
"""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def step(title: str, world=None):
    """``with rw.step("put the cube in the bin"):`` around part of a test. ``world`` defaults to the
    test's world (the one ``rw.launch`` or the ``world`` fixture made)."""
    from . import _current

    w = world or _current.get()
    tr = getattr(w, "trace", None) if w is not None else None
    stack = w.__dict__.setdefault("_rw_steps", []) if w is not None else []
    e = tr.event("step", title, status="running") if tr else None
    stack.append(title)
    try:
        yield
    except BaseException as ex:
        if e is not None:
            first = str(ex).splitlines()[0] if str(ex) else ""
            tr.finish_event(e, "failed", f"{type(ex).__name__}: {first}" if first else type(ex).__name__)
        if hasattr(ex, "add_note") and not getattr(ex, "_rw_step_noted", False):
            ex.add_note(f"in step {' > '.join(stack)!r}")  # the innermost step names its whole path, once
            ex._rw_step_noted = True
        raise
    else:
        if e is not None:
            tr.finish_event(e, "ok")
    finally:
        stack.pop()
