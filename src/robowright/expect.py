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

import contextvars
from pathlib import Path

import numpy as np

from .errors import ExpectationError
from .locators import ObjectHandle, Point, Subject, as_subject
from .robot import GRIPPER_CLOSED, Gripper, Robot, _jsonable

# Holding, without contact sensing: jaws closed on nothing stop at an opening of 0.031 at most
# (the 13 built-in arms, on MuJoCo), and on the 25 mm cube at 0.136 at least.
_GRASPED = 0.06

# Keep robowright internals out of pytest failure tracebacks (--full-trace shows them).
__tracebackhide__ = True


def _fmt(p) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in np.asarray(p).ravel()) + ")"


# Set while expect(block).to_pass() runs its block: each expectation in it checks once, now, instead of waiting.
_ONCE = contextvars.ContextVar("robowright_expect_once", default=False)


class Expectation:
    """What ``expect(subject)`` returns: one matcher call is one assertion.

    Every matcher waits. It checks at each control step until it holds, then for ``hold`` more seconds if given, and
    fails with what it last saw once ``timeout`` seconds of simulated time have gone (``expect(..., timeout=)``, or
    ``Settings.expect_timeout``). ``.not_`` asserts the opposite; ``.always`` makes the matcher an invariant, checked
    after every step for the rest of the test. A matcher returns the expectation, so calls can be chained.
    """

    def __init__(self, subject, world=None, timeout=None, negate=False, mode="eventually", message=None, soft=False):
        self.raw = subject
        self.world = world or _world_of(subject)
        if subject is self.world:
            self.subject = _WorldView()  # expect(world): what its cameras see (to_match_snapshot)
        elif isinstance(subject, _Polled):
            self.subject = subject
        elif callable(subject) and not isinstance(subject, Subject):
            self.subject = _Polled(subject)  # expect(block).to_pass(), or a value as expect.poll reads it
        else:
            self.subject = subject if isinstance(subject, (Robot,)) else as_subject(self.world, subject)
        self.timeout = timeout
        self.negate = negate
        self.mode = mode
        self.message = message
        self.soft = soft

    # modifiers ---------------------------------------------------------------
    @property
    def not_(self) -> Expectation:
        """The same expectation, inverted: ``expect(cube).not_.to_be_touching("floor")``."""
        return Expectation(self._subject_arg, self.world, self.timeout, not self.negate, self.mode, self.message, self.soft)

    @property
    def always(self) -> Expectation:
        """The matcher as an invariant: registered now, checked after every step, and the test fails at the step it
        stops holding (``InvariantViolation``). ``expect(robot).always.to_have_no_collisions()``."""
        return Expectation(self._subject_arg, self.world, self.timeout, self.negate, "always", self.message, self.soft)

    @property
    def _subject_arg(self):
        return self.subject if isinstance(self.subject, _Polled) else self.raw

    # core --------------------------------------------------------------------
    def _settles(self, check, timeout) -> bool:
        """Whether ``check`` comes to hold within the matcher's wait: a snapshot under ``--rw-update-snapshots changed``
        is written again only once it has had the time a test gives it to match."""
        if self.negate or self.mode == "always" or _ONCE.get():
            return True
        t = timeout if timeout is not None else self.timeout
        return self.world.run_until(lambda: check()[0], self.world.settings.expect_timeout if t is None else t, 0.0)

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
        if isinstance(self.subject, _Polled) or name.startswith(("to_satisfy", "to_have_count")) or name.split("(")[0] in _CUSTOM:
            rec["code"] = True  # it runs the test's own code, which a trace does not keep: codegen cannot write it again
        if self.mode != "always" and not _ONCE.get():  # once per assertion: not each try of a to_pass block
            hooks = list(getattr(w, "_action_hooks", ()))  # a debugger (--rw-debug) pausing before it
            if hooks:
                from .trace import _call

                what = _call({"type": "expect", "name": f"expect({self.subject.name}).{name}", "args": rec})
                for hook in hooks:
                    hook(w, what)
        if _ONCE.get():  # inside to_pass's block: checked once, now; to_pass waits and tries the block again
            if self.mode == "always":
                raise TypeError(f"{label}: an invariant (.always) is registered once, outside to_pass's block")
            ok, detail = pred()
            if ok:
                return self
            raise ExpectationError(f"{label}: {detail}")
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

    def to_be_between(self, lo: float, hi: float, *, timeout=None, hold=0.0):
        """A joint's value within ``[lo, hi]``: a hinge angle in rad or a slide in m (``world.joint(name)``)."""

        def check():
            v = self.subject.value
            return lo <= v <= hi, f"{self.subject.name} = {v:.3f} (want {lo:.3f}..{hi:.3f})"

        return self._run("to_be_between", check, timeout, hold, {"lo": lo, "hi": hi})

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
            margin = f"{by:g} m above" if by else "above"  # "must be 0 above z=0.12" read as a typo
            return z >= ref() + by, f"{self.subject.name} z={z:.3f}, must be {margin} {name}" + (
                "" if isinstance(target, (int, float)) else f" (at {ref():.3f})"
            )

        return self._run("to_be_above", check, timeout, hold, {"target": target, "by": by})

    def to_have_position(self, xyz, tol: float = 0.01, **kw):
        """At ``xyz`` (world frame, metres) to within ``tol``: ``to_be_near`` a point."""
        return self.to_be_near(Point(xyz), tol, **kw)

    def to_have_heading(self, yaw: float, tol: float = 0.05, *, timeout=None, hold=0.0):
        """Facing ``yaw`` (radians, about the vertical axis) to within ``tol``: a mobile robot's base."""
        if not hasattr(self.subject, "yaw"):
            raise TypeError(f"to_have_heading needs a base or a body with a yaw, got {self.subject!r}")

        def check():
            d = abs(float(np.arctan2(np.sin(self.subject.yaw - yaw), np.cos(self.subject.yaw - yaw))))
            return d <= tol, f"{self.subject.name} heading {np.degrees(self.subject.yaw):.1f} deg, target {np.degrees(yaw):.1f} deg"

        return self._run("to_have_heading", check, timeout, hold, {"yaw": yaw, "tol": tol})

    # topic matchers --------------------------------------------------------------
    def to_publish(self, hz: float | None = None, *, tol: float = 0.25, timeout=None, hold=0.0):
        """A ROS topic publishes (at ``hz``, within ``tol`` as a fraction, if given), measured over
        the last seconds of the world's clock. ``expect(world.ros.topic("/scan")).to_publish(hz=4)``."""
        t = self.subject
        if not hasattr(t, "hz"):
            raise TypeError(f"to_publish needs a ROS topic (world.ros.topic(...)), got {t!r}")

        def check():
            rate, n = t.hz, t.count
            if hz is None:
                return n > 0, f"{t.name}: {n} message(s), {rate:.2f} Hz"
            ok = n > 0 and abs(rate - hz) <= tol * hz
            return ok, f"{t.name} measured {rate:.2f} Hz ({n} messages so far), want {hz} Hz +/- {tol * 100:.0f}%"

        return self._run("to_publish", check, timeout, hold, {"hz": hz, "tol": tol})

    # image matchers -------------------------------------------------------------
    def to_match_snapshot(
        self,
        name: str,
        *,
        camera: str = "front",
        size: tuple[int, int] = (320, 240),
        threshold: float = 0.2,
        max_diff_pixel_ratio: float = 0.0,
        timeout: float = 0.0,
        mask=(),
        mask_margin: float = 0.01,
    ):
        """What ``camera`` sees matches the image kept for this test: Playwright's ``toHaveScreenshot``.
        ``expect(world).to_match_snapshot("tidy.png")``. A pixel differs when a channel changes by more
        than ``threshold`` of full scale; at most ``max_diff_pixel_ratio`` of them may. ``mask`` (scene objects or
        their names) leaves out where those objects are, now and in the baseline, as Playwright's ``mask`` does, with
        ``mask_margin`` metres around each for the shadow it casts. The baseline is
        ``__snapshots__/<test file>/<name>-<engine>-<robot>.png``; the first run writes it and fails
        (check it, then run again), and ``pytest --rw-update-snapshots`` writes it again. See
        robowright.snapshots.

        On a scene object, ``expect(cube).to_match_snapshot("cube.png")`` compares only the part of the image around it,
        as Playwright's element screenshot does: a window the size of the baseline, centred where the object is now,
        so it checks how the object looks wherever it has moved to."""
        from . import snapshots

        if self.negate:
            raise ValueError("to_match_snapshot cannot be negated: say what the view should look like")
        region = None if isinstance(self.subject, _WorldView) else self.subject
        if region is not None and not hasattr(region, "bounds"):
            raise TypeError(f"to_match_snapshot takes the world or a scene object, not {self.subject.name}")
        w = self.world
        w.require("render", "to_match_snapshot")
        names = [c.name for c in w.spec.cameras]
        if camera not in names:
            raise ValueError(f"unknown camera {camera!r}; this world has: {', '.join(names)}")
        path = snapshots.baseline_path(name, w.backend.name, w.spec.robot or "robot")
        width, height = size
        args = {"name": name, "camera": camera, "threshold": threshold, "max_diff_pixel_ratio": max_diff_pixel_ratio}
        if mask:
            args["mask"] = [getattr(m, "name", m) for m in mask]
        label = f"to_match_snapshot({name!r})"
        mask_path = path.with_name(path.stem + ".mask.png")  # where the masked objects were in the baseline

        def view(window=None):
            """What is compared: the whole image, or the window around ``region`` (the baseline's size, or the object's
            own box for a new baseline), with the masked area cut the same way."""
            img = w.backend.render(camera, width, height)[..., :3]
            masked = snapshots.mask_of(w, mask, camera, width, height, mask_margin) if mask else None
            if region is None:
                return img, masked
            box = snapshots.box_of(w, region, camera, width, height, margin=0.0)
            if box is None:
                raise ExpectationError(f"expect({region.name}).to_match_snapshot: {region.name} is behind the {camera} camera")
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            size = window or (box[2] - box[0], box[3] - box[1])
            return snapshots.crop(img, center, size), (None if masked is None else snapshots.crop(masked, center, size))

        mode = snapshots.update_mode()

        def write():
            img, masked = view()
            snapshots.write(path, img)
            if mask:
                snapshots.write(mask_path, np.repeat(masked[..., None], 3, -1).astype(np.uint8) * 255)
            else:
                mask_path.unlink(missing_ok=True)

        if not path.exists() and mode == "none":
            return self._run(label, lambda: (False, f"no snapshot {path} (--rw-update-snapshots none: not written)"), 0.0, 0.0, args)
        if mode == "all" or not path.exists():
            write()
            if mode in ("all", "changed"):
                return self._run(label, lambda: (True, f"wrote {path}"), timeout, 0.0, args)
            missing = f"no snapshot {path} yet: wrote it from this run; look at it, then run again"
            return self._run(label, lambda: (False, missing), 0.0, 0.0, args)
        expected = snapshots.read(path)
        was_masked = snapshots.read(mask_path)[..., 0] > 127 if mask and mask_path.exists() else None
        out = Path(w.settings.trace_dir) / "snapshots" / snapshots._slug(f"{w.name}-{Path(name).stem}")

        def check():
            actual, masked = view((expected.shape[1], expected.shape[0]) if region is not None else None)
            if masked is not None and was_masked is not None and was_masked.shape == masked.shape:
                masked |= was_masked  # a masked object that moved: where it is now and where it was
            n, diff = snapshots.compare(actual, expected, threshold, masked)
            if n < 0:
                got, want = f"{actual.shape[1]}x{actual.shape[0]}", f"{expected.shape[1]}x{expected.shape[0]}"
                return False, f"{camera} renders {got}, the snapshot is {want} ({path})"
            ratio = n / max(1, expected.shape[0] * expected.shape[1] - (int(masked.sum()) if masked is not None else 0))
            if ratio <= max_diff_pixel_ratio:
                return True, f"{n} pixels differ ({ratio:.2%}) from {path}"
            a = snapshots.write(Path(f"{out}-actual.png"), actual)
            d = snapshots.write(Path(f"{out}-diff.png"), diff)
            return False, f"{n} pixels differ ({ratio:.2%}, {max_diff_pixel_ratio:.2%} allowed) from {path}\n  actual: {a}\n  diff: {d}"

        if mode == "changed" and not self._settles(check, timeout):
            write()
            return self._run(label, lambda: (True, f"updated {path}: it no longer matched"), 0.0, 0.0, args)
        return self._run(label, check, timeout, 0.0, args)

    def to_match_state_snapshot(self, name: str, *, tol: float = 0.01, joint_tol: float = 0.05, timeout: float = 0.0):
        """The world's state matches the text kept for this test: where each object is (within ``tol`` m), the
        robot's joints (within ``joint_tol`` rad) and a mobile robot's base. Playwright's ``toMatchAriaSnapshot``
        for a world: ``expect(world).to_match_state_snapshot("tidy")``. The baseline is
        ``__snapshots__/<test file>/<name>-<engine>-<robot>.state.txt``, written by the first run (which fails: read
        it, then run again) and by ``pytest --rw-update-snapshots``. It is plain text: delete a line to stop checking
        that object or joint. See robowright.snapshots."""
        from . import snapshots

        if self.negate:
            raise ValueError("to_match_state_snapshot cannot be negated: say what the state should be")
        w = self.world
        w.require("ground_truth", "to_match_state_snapshot")
        path = snapshots.baseline_path(Path(name).stem + ".txt", w.backend.name, w.spec.robot or "robot")
        path = path.with_suffix(snapshots.STATE_SUFFIX)
        args = {"name": name, "tol": tol, "joint_tol": joint_tol}
        label = f"to_match_state_snapshot({name!r})"
        header = f"robowright state snapshot {name!r}: {w.backend.name}, {w.spec.robot or 'robot'}"
        mode = snapshots.update_mode()

        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(snapshots.format_state(snapshots.state_of(w), header))

        if not path.exists() and mode == "none":
            return self._run(label, lambda: (False, f"no snapshot {path} (--rw-update-snapshots none: not written)"), 0.0, 0.0, args)
        if mode == "all" or not path.exists():
            write()
            if mode in ("all", "changed"):
                return self._run(label, lambda: (True, f"wrote {path}"), timeout, 0.0, args)
            missing = f"no snapshot {path} yet: wrote it from this run; read it, then run again"
            return self._run(label, lambda: (False, missing), 0.0, 0.0, args)
        expected = snapshots.parse_state(path.read_text(), path)
        out = Path(w.settings.trace_dir) / "snapshots" / (snapshots._slug(f"{w.name}-{Path(name).stem}") + snapshots.STATE_SUFFIX)

        def check():
            actual = snapshots.state_of(w)
            diffs = snapshots.compare_state(actual, expected, tol, joint_tol)
            if not diffs:
                return True, f"matches {path}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(snapshots.format_state(actual, header))
            n = f"{len(diffs)} entries differ" if len(diffs) > 1 else "1 entry differs"
            return False, f"{n} from {path}:\n  " + "\n  ".join(diffs) + f"\n  this run's state: {out}"

        if mode == "changed" and not self._settles(check, timeout):
            # the entries the snapshot has, as they are now: a line deleted to stop checking it stays deleted
            actual = snapshots.state_of(w)
            kept = {sec: {k: actual[sec][k] for k in entries if k in actual.get(sec, {})} for sec, entries in expected.items()}
            path.write_text(snapshots.format_state(kept, header))
            return self._run(label, lambda: (True, f"updated {path}: it no longer matched"), 0.0, 0.0, args)
        return self._run(label, check, timeout, 0.0, args)

    # state matchers -------------------------------------------------------------
    def to_be_at_rest(self, lin_tol: float = 0.01, ang_tol: float = 0.1, *, timeout=None, hold=None):
        """Linear speed below ``lin_tol`` m/s and angular below ``ang_tol`` rad/s, for ``hold`` seconds (0.1 unless
        given; none where robowright cannot let time pass, an attached simulation with no ``advance``)."""
        if hold is None:
            hold = 0.1 if getattr(self.world, "can_wait", True) else 0.0

        def check():
            if isinstance(self.subject, Robot):
                v = np.abs(self.world.backend.qvel()).max()
                return v < ang_tol, f"max joint speed {v:.3f} rad/s"
            v = self.subject.velocity
            lin, ang = float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))
            return lin < lin_tol and ang < ang_tol, f"{self.subject.name} moving {lin:.4f} m/s, {ang:.3f} rad/s"

        return self._run("to_be_at_rest", check, timeout, hold, {"lin_tol": lin_tol, "ang_tol": ang_tol})

    def to_be_upright(self, tol_deg: float = 10.0, *, timeout=None, hold=0.0):
        """Its up axis within ``tol_deg`` degrees of vertical: a box that has not tipped over, a legged robot standing."""

        def check():
            up = self.subject.up_axis
            ang = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
            return ang <= tol_deg, f"{self.subject.name} tilted {ang:.1f} deg (tol {tol_deg})"

        return self._run("to_be_upright", check, timeout, hold, {"tol_deg": tol_deg})

    def to_be_touching(self, other, *, timeout=None, hold=0.0):
        """In contact with ``other`` (a name or handle). Needs an engine that reports contacts."""
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
        """A gripper (or a robot's) opened to at least ``min_opening`` of its range (0 closed, 1 fully open)."""
        g = self._gripper()
        return self._run(
            "to_be_open",
            lambda: (g.opening >= min_opening, f"opening {g.opening:.2f}"),
            kw.get("timeout"),
            kw.get("hold", 0.0),
            {"min_opening": min_opening},
        )

    def to_be_closed(self, max_opening: float = 0.1, **kw):
        """A gripper (or a robot's) closed to at most ``max_opening`` of its range (0 closed, 1 fully open)."""
        g = self._gripper()
        return self._run(
            "to_be_closed",
            lambda: (g.opening <= max_opening, f"opening {g.opening:.2f}"),
            kw.get("timeout"),
            kw.get("hold", 0.0),
            {"max_opening": max_opening},
        )

    def to_have_joint(self, joint: str, value: float, tol: float = 0.02, **kw):
        """The robot's ``joint`` at ``value`` (rad, or m for a slide) to within ``tol``."""
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
        """A custom condition: ``predicate(subject) -> bool``, checked every control step until it holds
        (Playwright's ``expect.poll``). ``expect(world).to_satisfy(lambda w: ...)`` is given the world."""
        target = self.world if isinstance(self.subject, _WorldView) else self.subject
        return self._run(f"to_satisfy({description})", lambda: (bool(predicate(target)), description), timeout, hold)

    # value matchers (expect.poll, a joint) ------------------------------------------
    def to_have_count(self, n: int, *, timeout=None, hold=0.0):
        """As many objects match as ``n``: ``expect(scene.filter(inside="bin")).to_have_count(3)`` (Playwright's
        ``toHaveCount``), the objects found again every control step."""
        from .locators import Objects

        if not isinstance(self.subject, Objects):
            raise TypeError(f"to_have_count takes scene.filter(...), not {self.subject.name}")

        def check():
            found = [h.name for h in self.subject.all()]
            return len(found) == n, f"{len(found)} match {found} (want {n})"

        return self._run("to_have_count", check, timeout, hold, {"n": n})

    def to_equal(self, expected, *, timeout=None, hold=0.0):
        """The value equals ``expected`` (arrays: every element)."""

        def check():
            v = self._value()
            same = np.array_equal(v, expected) if isinstance(v, np.ndarray) or isinstance(expected, np.ndarray) else v == expected
            return bool(same), f"{self.subject.name} = {v!r} (want {expected!r})"

        return self._run("to_equal", check, timeout, hold, {"expected": expected})

    def to_be_greater_than(self, bound, *, timeout=None, hold=0.0):
        """The value (``expect.poll``, or a joint's) is greater than ``bound``."""

        def check():
            v = self._value()
            return bool(v > bound), f"{self.subject.name} = {v!r} (want > {bound!r})"

        return self._run("to_be_greater_than", check, timeout, hold, {"bound": bound})

    def to_be_less_than(self, bound, *, timeout=None, hold=0.0):
        """The value (``expect.poll``, or a joint's) is less than ``bound``."""

        def check():
            v = self._value()
            return bool(v < bound), f"{self.subject.name} = {v!r} (want < {bound!r})"

        return self._run("to_be_less_than", check, timeout, hold, {"bound": bound})

    def to_be_close_to(self, expected, tol: float = 1e-6, *, timeout=None, hold=0.0):
        """Within ``tol`` of ``expected`` (arrays: every element)."""

        def check():
            v = self._value()
            err = float(np.max(np.abs(np.asarray(v, dtype=float) - np.asarray(expected, dtype=float))))
            return err <= tol, f"{self.subject.name} = {v!r} (want {expected!r} within {tol:g}, off by {err:.3g})"

        return self._run("to_be_close_to", check, timeout, hold, {"expected": expected, "tol": tol})

    def to_be_truthy(self, *, timeout=None, hold=0.0):
        """The value (``expect.poll``, or a joint's) is truthy."""

        def check():
            v = self._value()
            return bool(v), f"{self.subject.name} = {v!r}"

        return self._run("to_be_truthy", check, timeout, hold)

    def to_pass(self, *, timeout=None):
        """``expect(block).to_pass()``: run ``block()`` every control step until it raises no ExpectationError or
        AssertionError (Playwright's ``toPass``). Each expectation in it checks once, as things are at that step, and
        the failure says which one failed last. ``.always.to_pass()`` makes the block an invariant, run after every step.
        A block should check, not act: what it moves, it moves again at every try."""
        if not isinstance(self.subject, _Polled):
            raise TypeError("to_pass takes a block: expect(lambda: ...).to_pass()")
        if self.negate:
            raise TypeError("expect(block).not_.to_pass() says nothing a test can rely on: assert the opposite inside the block")
        block = self.subject.fn

        def check():
            token = _ONCE.set(True)
            try:
                block()
                return True, "passed"
            except (ExpectationError, AssertionError) as e:
                text = str(e).strip()
                return False, text.splitlines()[0] if text else type(e).__name__
            finally:
                _ONCE.reset(token)

        return self._run("to_pass", check, timeout)

    # helpers -----------------------------------------------------------------------
    def _value(self):
        s = self.subject
        if isinstance(s, _Polled):
            return s.value
        if hasattr(type(s), "value"):  # a joint: its angle or extension
            return s.value
        raise TypeError(f"value matchers read expect.poll(...) or a joint, not {s!r}")

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


class _Polled(Subject):
    """A value the test computes, read again every control step: ``expect.poll(fn)``, or the block of ``to_pass``."""

    def __init__(self, fn, name: str | None = None):
        self.fn = fn
        fn_name = getattr(fn, "__name__", "")
        self.name = name or (f"{fn_name}()" if fn_name and fn_name != "<lambda>" else "value")

    @property
    def value(self):
        return self.fn()

    @property
    def position(self) -> np.ndarray:
        raise TypeError("expect.poll(...) takes value matchers (to_equal, to_be_greater_than, ...), not position matchers")


class _WorldView(Subject):
    """The world as a subject: ``expect(world)`` asserts on what its cameras see, not on a position."""

    name = "world"

    @property
    def position(self) -> np.ndarray:
        raise TypeError("expect(world) takes image matchers (to_match_snapshot), not position matchers")


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
    """Start an assertion on ``subject``: a scene object or its name, the robot, its gripper or base, a joint, a ROS
    topic, the world itself (what its cameras see), or a block of assertions (``to_pass``).

    ``timeout`` overrides the matcher's wait (seconds of simulated time), ``message`` heads the failure, and
    ``world`` is needed only when the subject does not know its world. Two more forms:

    * ``expect.soft(subject)``: a failure is recorded and the test carries on; it fails at teardown.
    * ``expect.poll(fn, name=None)``: ``fn()`` is read again at every control step until the matcher holds.

    ``expect.extend(to_be_level=fn)`` adds your own matchers (see :func:`_extend`).
    """
    return Expectation(subject, world, timeout, message=message)


def _soft(subject, *, timeout=None, message=None, world=None) -> Expectation:
    """Like ``expect`` but records the failure and keeps going; the test fails at teardown."""
    return Expectation(subject, world, timeout, message=message, soft=True)


expect.soft = _soft


def _poll(fn, *, name: str | None = None, timeout=None, message=None, world=None) -> Expectation:
    """``expect.poll(fn).to_equal(3)``: ``fn()`` read again every control step until the matcher holds, in simulated
    time (Playwright's ``expect.poll``). The failure says the last value it read; ``name`` is what it calls the value."""
    return Expectation(_Polled(fn, name), world, timeout, message=message)


expect.poll = _poll


def _configure(*, timeout: float | None = None, soft: bool = False, message: str | None = None):
    """An ``expect`` with its own defaults, as Playwright's ``expect.configure``::

        slow_expect = expect.configure(timeout=10.0)   # a stack that takes its time
        soft_expect = expect.configure(soft=True)       # every failure recorded, the test goes on

    ``timeout`` (seconds of simulated time) is used where a matcher is not given one; ``soft`` makes every assertion
    soft; ``message`` heads every failure. What it returns takes the same arguments as ``expect``."""

    def configured(subject, *, timeout: float | None = timeout, message: str | None = message, world=None) -> Expectation:
        return Expectation(subject, world, timeout, message=message, soft=soft)

    configured.soft = lambda subject, **kw: _soft(subject, **{"timeout": timeout, "message": message, **kw})
    configured.poll = lambda fn, **kw: _poll(fn, **{"timeout": timeout, "message": message, **kw})
    configured.extend = _extend
    configured.configure = lambda **kw: _configure(**{"timeout": timeout, "soft": soft, "message": message, **kw})
    return configured


expect.configure = _configure

_CUSTOM: set[str] = set()  # matchers added with expect.extend


def _extend(**matchers) -> None:
    """Add your own matchers, as Playwright's ``expect.extend``::

        def to_be_level(obj, tol_deg=2.0):
            tilt = obj.tilt_deg()
            return tilt <= tol_deg, f"tilted {tilt:.1f} deg"

        expect.extend(to_be_level=to_be_level)
        expect(scene["tray"]).to_be_level(tol_deg=1.0, timeout=2.0)

    Each function takes what ``expect()`` was given (a scene object's handle, the robot, a joint, the world; the value
    for ``expect.poll``) and its own arguments, and returns ``(passed, detail)`` or a bool. It then behaves as the
    built-in matchers do: checked every control step until it holds or ``timeout`` runs out, ``hold=`` seconds, ``.not_``,
    ``.always``, ``expect.soft``, ``condition()``, in the trace and the debugger. Names start with ``to_`` and cannot
    replace a built-in matcher; a generated test (codegen) leaves them as comments, as it does ``to_satisfy``."""
    import inspect

    for name, fn in matchers.items():
        if not name.startswith("to_") or not name.isidentifier():
            raise ValueError(f"expect.extend: a matcher's name starts with to_ (got {name!r})")
        if hasattr(Expectation, name) and name not in _CUSTOM:
            raise ValueError(f"expect.extend: {name} is a built-in matcher")
        if not callable(fn):
            raise TypeError(f"expect.extend: {name} is {fn!r}, not a function")
        sig = inspect.signature(fn)

        def matcher(self, *args, timeout=None, hold=0.0, _fn=fn, _name=name, _sig=sig, **kwargs):
            bound = _sig.bind(None, *args, **kwargs).arguments
            recorded = {k: v for i, (k, v) in enumerate(bound.items()) if i}  # the subject is not an argument

            def check():
                subject = self.world if isinstance(self.subject, _WorldView) else self.subject
                if isinstance(subject, _Polled):
                    subject = subject.value
                out = _fn(subject, *args, **kwargs)
                ok, detail = out if isinstance(out, tuple) else (out, "")
                return bool(ok), str(detail) if detail else f"{_name} returned {ok!r}"

            return self._run(_name, check, timeout, hold, recorded)

        matcher.__name__ = matcher.__qualname__ = name
        matcher.__doc__ = fn.__doc__
        setattr(Expectation, name, matcher)
        _CUSTOM.add(name)


expect.extend = _extend


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
