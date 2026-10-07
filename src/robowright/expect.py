"""``expect``: assertions that wait.

A physical condition is rarely true the instant you check it - the cube is
still settling, the arm is still slowing down. Like Playwright's web-first
assertions, every matcher re-checks the condition each control step until
it holds (optionally for ``hold`` seconds) or the timeout expires, then
fails with the last observed state.

``expect(x).always.<matcher>()`` registers the condition as an invariant
instead: it is checked after every step for the rest of the test.
"""

from __future__ import annotations

import numpy as np

from .errors import ExpectationError
from .locators import ObjectHandle, Point, as_subject
from .robot import GRIPPER_CLOSED, Gripper, Robot, _jsonable

# Holding, without contact sensing: jaws closed on nothing stop at an opening of 0.031 at most
# (the 13 built-in arms, on MuJoCo), and on the 25 mm cube at 0.136 at least.
_GRASPED = 0.06

# Keep robowright internals out of pytest failure tracebacks (--full-trace shows them).
__tracebackhide__ = True


def _fmt(p) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in np.asarray(p).ravel()) + ")"


class Expectation:
    def __init__(self, subject, world=None, timeout=None, negate=False, mode="eventually", message=None, soft=False):
        self.raw = subject
        self.world = world or _world_of(subject)
        self.subject = subject if isinstance(subject, (Robot,)) else as_subject(self.world, subject)
        self.timeout = timeout
        self.negate = negate
        self.mode = mode
        self.message = message
        self.soft = soft

    # modifiers ---------------------------------------------------------------
    @property
    def not_(self) -> Expectation:
        return Expectation(self.raw, self.world, self.timeout, not self.negate, self.mode, self.message, self.soft)

    @property
    def always(self) -> Expectation:
        return Expectation(self.raw, self.world, self.timeout, self.negate, "always", self.message, self.soft)

    # core --------------------------------------------------------------------
    def _run(self, name: str, check, timeout=None, hold: float = 0.0, args=None):
        w = self.world
        label = f"expect({self.subject.name}){'.not_' if self.negate else ''}.{name}"

        def pred():
            ok, detail = check()
            return (not ok if self.negate else ok), detail

        rec = {
            "subject": self.subject.name,
            "matcher": name.split("(")[0],
            "negate": self.negate,
            "kwargs": {k: _jsonable(v) for k, v in (args or {}).items()},
        }
        if timeout is not None:
            rec["kwargs"]["timeout"] = timeout
        elif self.timeout is not None:  # given to expect(): the matcher takes it too, and codegen keeps it
            rec["kwargs"]["timeout"] = self.timeout
        if hold:
            rec["kwargs"]["hold"] = hold
        if self.mode == "always":
            w.add_invariant(pred, label, rec)
            return self
        t = timeout if timeout is not None else self.timeout
        t = w.settings.expect_timeout if t is None else t
        last = {"detail": ""}

        def p():
            ok, detail = pred()
            last["detail"] = detail
            return ok

        t0 = w.time
        ok = w.run_until(p, t, hold)
        tr = w.trace
        if ok:
            if tr:
                tr.event("expect", label, rec, "ok", last["detail"])
            return self
        msg = (
            f"{self.message + ': ' if self.message else ''}{label} failed after {w.time - t0:.2f}s "
            f"(timeout {t}s{f', hold {hold}s' if hold else ''})\n  {last['detail']}"
        )
        if tr:
            tr.event("expect", label, rec, "failed", last["detail"])
        if self.soft:
            w.soft_fail(msg)
            return self
        w.status = "failed"
        raise ExpectationError(msg)

    # position matchers -------------------------------------------------------
    def to_be_near(self, target, tol: float = 0.01, *, timeout=None, hold=0.0):
        """Within ``tol`` metres of ``target`` (a point, tuple or another handle)."""
        tgt = as_subject(self.world, target)

        def check():
            d = float(np.linalg.norm(self.subject.position - tgt.position))
            where = "the target" if isinstance(tgt, Point) else tgt.name  # a point's name is its position
            return (
                d <= tol,
                f"{self.subject.name} at {_fmt(self.subject.position)}, {where} at {_fmt(tgt.position)}, "
                f"distance {d * 1000:.1f} mm (tol {tol * 1000:.1f} mm)",
            )

        return self._run("to_be_near", check, timeout, hold, {"target": tgt, "tol": tol})

    def to_be_inside(self, container, margin: float = 0.0, *, timeout=None, hold=0.0):
        """Centre within ``container``'s bounds (shrunk by ``margin``)."""
        c = as_subject(self.world, container)

        def check():
            lo, hi = c.bounds()
            p = self.subject.position
            inside = bool(np.all(p >= lo + margin) and np.all(p <= hi - margin))
            return inside, f"{self.subject.name} at {_fmt(p)}; {c.name} spans {_fmt(lo)}..{_fmt(hi)}"

        return self._run("to_be_inside", check, timeout, hold, {"container": c, "margin": margin})

    def to_be_above(self, target, by: float = 0.0, *, timeout=None, hold=0.0):
        """Higher than ``target`` (a height in metres or a handle's top surface) by ``by``."""
        if isinstance(target, (int, float)):
            ref, name = (lambda: float(target)), f"z={target}"
        else:
            t = as_subject(self.world, target)
            ref, name = (lambda: t.top if isinstance(t, ObjectHandle) else float(t.position[2])), t.name

        def check():
            z = float(self.subject.position[2])
            return z >= ref() + by, f"{self.subject.name} z={z:.3f}, {name} reference {ref():.3f} (+{by})"

        return self._run("to_be_above", check, timeout, hold, {"target": target, "by": by})

    def to_have_position(self, xyz, tol: float = 0.01, **kw):
        return self.to_be_near(Point(xyz), tol, **kw)

    # state matchers -------------------------------------------------------------
    def to_be_at_rest(self, lin_tol: float = 0.01, ang_tol: float = 0.1, *, timeout=None, hold=0.1):
        """Linear speed below ``lin_tol`` m/s and angular below ``ang_tol`` rad/s."""

        def check():
            if isinstance(self.subject, Robot):
                v = np.abs(self.world.backend.qvel()).max()
                return v < ang_tol, f"max joint speed {v:.3f} rad/s"
            v = self.subject.velocity
            lin, ang = float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))
            return lin < lin_tol and ang < ang_tol, f"{self.subject.name} moving {lin:.4f} m/s, {ang:.3f} rad/s"

        return self._run("to_be_at_rest", check, timeout, hold, {"lin_tol": lin_tol, "ang_tol": ang_tol})

    def to_be_upright(self, tol_deg: float = 10.0, *, timeout=None, hold=0.0):
        def check():
            up = self.subject.up_axis
            ang = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
            return ang <= tol_deg, f"{self.subject.name} tilted {ang:.1f} deg (tol {tol_deg})"

        return self._run("to_be_upright", check, timeout, hold, {"tol_deg": tol_deg})

    def to_be_touching(self, other, *, timeout=None, hold=0.0):
        o = other if isinstance(other, str) else as_subject(self.world, other).name

        def check():
            touching = self.subject.contacts()
            return o in touching, f"{self.subject.name} touching {touching or 'nothing'}"

        return self._run("to_be_touching", check, timeout, hold, {"other": o})

    # gripper / robot matchers ------------------------------------------------------
    def to_be_holding(self, obj=None, *, timeout=None, hold=0.0):
        """Both jaws in contact with ``obj`` (or with anything, if omitted)."""
        g = self._gripper()
        name = None if obj is None else (obj if isinstance(obj, str) else as_subject(self.world, obj).name)

        def check():
            fixed, moving = g.touching()
            held = (fixed & moving) - {"floor"}
            ok = (name in held) if name else bool(held)
            return (
                ok,
                f"left finger touching {sorted(fixed) or 'nothing'}, right finger touching {sorted(moving) or 'nothing'}, "
                f"opening {g.opening:.2f}",
            )

        return self._run("to_be_holding", check if self.world.has_contacts else self._jaws_on(g, name), timeout, hold, {"obj": name})

    def _jaws_on(self, g: Gripper, name: str | None):
        """Holding, on a robot that cannot feel contacts (hardware): the jaws were told to close and
        stopped short on something, and the object (if its pose is known) is in the hand."""
        w = self.world
        r = g.robot

        def check():
            told = r._target[-1] <= GRIPPER_CLOSED + 0.05
            stopped = g.opening > _GRASPED
            ok, where = told and stopped, ""
            if name is not None and (w.has_ground_truth or name in w.perception):
                o = w.scene[name]
                d = float(np.linalg.norm(o.position - r.tcp.position))
                reach = float(np.max(o.bounds()[1] - o.position)) + 0.03
                ok, where = ok and d <= reach, f", {name} {d:.3f} m from the tool (within {reach:.3f} counts)"
            told_s = "told to close" if told else f"told to open to {r._target[-1]:.2f}"
            return ok, f"no contact sensing on {w.backend.name}: the jaws {told_s}, at opening {g.opening:.2f}{where}"

        return check

    def to_be_open(self, min_opening: float = 0.8, **kw):
        g = self._gripper()
        return self._run(
            "to_be_open",
            lambda: (g.opening >= min_opening, f"opening {g.opening:.2f}"),
            kw.get("timeout"),
            kw.get("hold", 0.0),
            {"min_opening": min_opening},
        )

    def to_be_closed(self, max_opening: float = 0.1, **kw):
        g = self._gripper()
        return self._run(
            "to_be_closed",
            lambda: (g.opening <= max_opening, f"opening {g.opening:.2f}"),
            kw.get("timeout"),
            kw.get("hold", 0.0),
            {"max_opening": max_opening},
        )

    def to_have_joint(self, joint: str, value: float, tol: float = 0.02, **kw):
        r = self._robot()

        def check():
            q = r.joints[joint]
            return abs(q - value) <= tol, f"{joint} = {q:.3f} rad (want {value:.3f} +/- {tol})"

        return self._run("to_have_joint", check, kw.get("timeout"), kw.get("hold", 0.0), {"joint": joint, "value": value, "tol": tol})

    def to_have_no_collisions(self, allow=("floor",), *, timeout=None, hold=0.0):
        """No arm link touches anything except ``allow`` (the fingers may touch objects)."""
        r = self._robot()
        allow = set(allow)

        def check():
            bad = []
            for c in r.world.backend.contacts():
                for me, other in ((c.a, c.b), (c.b, c.a)):
                    if me.startswith("robot:") and not other.startswith("robot:") and other not in allow:
                        if me in ("robot:left_finger", "robot:right_finger") and other != "floor":
                            continue
                        bad.append(f"{me}<->{other} ({c.force:.2f} N)")
            return not bad, ("collisions: " + ", ".join(sorted(set(bad)))) if bad else "no collisions"

        self.world.require("contacts", "collision checks")
        return self._run("to_have_no_collisions", check, timeout, hold, {"allow": sorted(allow)})

    def to_satisfy(self, predicate, description: str = "predicate", *, timeout=None, hold=0.0):
        """A custom condition: ``predicate(subject) -> bool``."""
        return self._run(f"to_satisfy({description})", lambda: (bool(predicate(self.subject)), description), timeout, hold)

    # helpers -----------------------------------------------------------------------
    def _gripper(self) -> Gripper:
        s = self.subject
        if isinstance(s, Gripper):
            return s
        if isinstance(s, Robot):
            return s.gripper
        raise TypeError(f"this matcher needs a gripper or robot, got {s!r}")

    def _robot(self) -> Robot:
        s = self.subject
        if isinstance(s, Robot):
            return s
        if isinstance(s, Gripper):
            return s.robot
        raise TypeError(f"this matcher needs a robot, got {s!r}")


def _world_of(subject):
    for attr in ("world",):
        if hasattr(subject, attr):
            return getattr(subject, attr)
    if hasattr(subject, "robot"):
        return subject.robot.world
    from . import _current

    w = _current.get()
    if w is None:
        raise ValueError(f"cannot find the world for {subject!r}; pass world=")
    return w


def expect(subject, *, timeout: float | None = None, message: str | None = None, world=None) -> Expectation:
    return Expectation(subject, world, timeout, message=message)


def _soft(subject, *, timeout=None, message=None, world=None) -> Expectation:
    """Like ``expect`` but records the failure and keeps going; the test fails at teardown."""
    return Expectation(subject, world, timeout, message=message, soft=True)


expect.soft = _soft


def condition(subject, matcher: str, *args, world=None, **kwargs):
    """A zero-argument predicate for ``run_policy(until=...)`` built from any matcher.

    ``condition(cube, "to_be_inside", "bin")`` is true whenever
    ``expect(cube).to_be_inside("bin")`` would pass right now.
    """
    e = Expectation(subject, world)
    captured = {}

    def fake_run(name, check, timeout=None, hold=0.0, args=None):
        captured["check"] = check
        return e

    e._run = fake_run
    getattr(e, matcher)(*args, **kwargs)
    check = captured["check"]

    def pred():
        return bool(check()[0])

    pred.__robowright__ = {
        "subject": e.subject.name,
        "matcher": matcher,
        "args": [_jsonable(a) for a in args],
        "kwargs": {k: _jsonable(v) for k, v in kwargs.items()},
    }
    return pred
