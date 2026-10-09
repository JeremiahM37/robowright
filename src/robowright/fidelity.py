"""How faithful the simulated robot is to the model it was read from, and a record of every change.

A test is only as good as what it runs. robowright reads a robot's own model file (MJCF or
URDF) and makes some changes to it before simulating it. Each change has a kind:

``sourced``
    A value from the robot maker's specification that the model leaves out or gets wrong,
    with the document it came from (a gripper's rated force, from its datasheet).
``repair``
    A defect without which the file does not simulate at all (link meshes that overlap at
    rest and lock their joint).
``interface``
    What it takes to command the robot through robowright's interface, by one rule the same
    for every robot: a gripper with several motors driven as one, legs with torque motors given
    the joint PD loop their firmware runs, a URDF (which has no actuators) given position
    servos from its effort limits, a model translated for another engine.
``adjusted``
    A change robowright chose because a test failed without it: stiffer contacts, a capped
    gripper, stiffened servos, extra motor inertia, a different integrator or friction cone.
    Each is a guess about the real robot, not a fact about it.

The fidelity mode decides which run. ``published`` (the default) applies the first three
kinds only, so a test runs the robot as its makers modelled it plus what they specified.
``adjusted`` applies everything, as robowright did before 0.2: more robots pass, on physics
that nobody measured. Choose it with ``--rw-fidelity`` (pytest), ``Settings(fidelity=...)``,
or ``ROBOWRIGHT_FIDELITY``. Every world records the changes it ran with
(``world.model_changes``), and the trace, the HTML report and ``robowright info --robot``
show them.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from dataclasses import asdict, dataclass

MODES = ("published", "adjusted")
KINDS = ("sourced", "repair", "interface", "adjusted")
ENV = "ROBOWRIGHT_FIDELITY"

_override: contextvars.ContextVar[str | None] = contextvars.ContextVar("robowright_fidelity", default=None)
_log: contextvars.ContextVar[list | None] = contextvars.ContextVar("robowright_changes", default=None)


@dataclass(frozen=True)
class Change:
    kind: str  # one of KINDS
    what: str  # a short name: "grip force", "resting contacts excluded", ...
    detail: str  # what changed, in numbers
    source: str = ""  # for "sourced": where the value comes from
    engine: str = ""  # the engine it applies to, when it is one engine's

    def describe(self) -> str:
        where = f" [{self.engine}]" if self.engine else ""
        src = f" (source: {self.source})" if self.source else ""
        return f"{self.kind}{where}: {self.what}: {self.detail}{src}"

    def as_dict(self) -> dict:
        return asdict(self)


def mode() -> str:
    """The fidelity mode in force: a ``using`` block's, else ``$ROBOWRIGHT_FIDELITY``, else published."""
    m = _override.get() or os.environ.get(ENV) or "published"
    if m not in MODES:
        raise ValueError(f"fidelity is one of {', '.join(MODES)}, not {m!r} (from ${ENV})")
    return m


def adjusted() -> bool:
    """Whether robowright's own adjustments apply (``adjusted`` mode)."""
    return mode() == "adjusted"


@contextlib.contextmanager
def using(m: str | None):
    """Build worlds in fidelity mode ``m`` inside the block (``None``: leave it as it is)."""
    if m is None:
        yield
        return
    if m not in MODES:
        raise ValueError(f"fidelity is one of {', '.join(MODES)}, not {m!r}")
    token = _override.set(m)
    try:
        yield
    finally:
        _override.reset(token)


def record(kind: str, what: str, detail: str, source: str = "", engine: str = "") -> None:
    """Note a change made to the model being built (kept only inside a ``recording`` block)."""
    if kind not in KINDS:
        raise ValueError(f"kind is one of {', '.join(KINDS)}")
    if kind == "adjusted" and not adjusted():
        return  # not applied in this mode, so not part of what ran
    log = _log.get()
    if log is not None:
        c = Change(kind, what, detail, source, engine)
        if c not in log:
            log.append(c)


@contextlib.contextmanager
def recording():
    """Collect every change ``record``-ed inside the block into the list it yields. Nested
    blocks (a model built while building another, to measure it) collect into their own."""
    log: list[Change] = []
    token = _log.set(log)
    try:
        yield log
    finally:
        _log.reset(token)


@contextlib.contextmanager
def silent():
    """Build something without recording it: a measurement copy, not the robot under test."""
    token = _log.set(None)
    try:
        yield
    finally:
        _log.reset(token)


def summary(changes) -> str:
    """The changes as text, one per line, grouped by kind."""
    if not changes:
        return "the model as published, unchanged"
    order = {k: i for i, k in enumerate(KINDS)}
    return "\n".join(c.describe() for c in sorted(changes, key=lambda c: order[c.kind]))
