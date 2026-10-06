"""The robot API: actions that wait until they are done.

Every action plans in joint space, streams targets one control period at a
time through the world, then waits until the robot has actually settled -
the robotics version of Playwright's actionability checks. Actions are
recorded in the trace with their arguments so codegen can replay them.
"""

from __future__ import annotations

import functools
import time as _time
from collections import deque

import numpy as np

from .errors import ActionTimeoutError, GraspError, UnreachableError
from .locators import ObjectHandle, Subject, as_subject
from .robots import Kinematics

# Keep robowright internals out of pytest failure tracebacks (--full-trace shows them).
__tracebackhide__ = True

# The gripper is commanded and read as an opening: 0 is closed, 1 is fully open.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
DOWN = (0.0, 0.0, -1.0)
# Fingertips must clear the table by this much at the bottom of a top-down grasp.
TABLE_CLEARANCE = 0.004


def _jsonable(v):
    if isinstance(v, Subject):
        return {"$ref": v.name} if isinstance(v, ObjectHandle) else {"$point": [float(x) for x in v.position]}
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.ravel()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "__robowright__"):
        return {"$condition": v.__robowright__}
    if hasattr(v, "to_config"):
        cls = type(v)
        return {"$policy": {"class": f"{cls.__module__}.{cls.__qualname__}", "kwargs": v.to_config()}}
    if callable(v):
        return {"$callable": getattr(v, "__qualname__", getattr(type(v), "__qualname__", repr(v)))}
    return repr(v)


def action(fn):
    """Record a robot method as a trace action event.

    Skills call other actions through ``__wrapped__`` so only the call the test
    made appears in the trace - that is what codegen turns back into code.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        world = self.world
        tr = world.trace
        ev = None
        if tr:
            names = fn.__code__.co_varnames[1 : fn.__code__.co_argcount]
            bound = dict(zip(names, args))
            bound.update(kwargs)
            ev = tr.event("action", f"{self._label}.{fn.__name__}", {k: _jsonable(v) for k, v in bound.items()}, status="running")
        try:
            out = fn(self, *args, **kwargs)
        except Exception as e:
            if ev:
                tr.finish_event(ev, "failed", f"{type(e).__name__}: {e}")
            world.status = "failed"
            raise
        if ev:
            tr.finish_event(ev, "ok")
        return out

    return wrapper


class TCP(Subject):
    """The tool centre point, computed from measured joint angles."""

    name = "tcp"

    def __init__(self, robot):
        self.robot = robot

    @property
    def position(self):
        r = self.robot
        return r.kin.tcp(r.true_qpos()[: r.n_arm])


class Arm:
    _label = "arm"

    def __init__(self, robot):
        self.robot = robot
        self.world = robot.world

    @property
    def home_q(self) -> np.ndarray:
        return self.robot.home_q

    @action
    def move_joints(self, q, speed: float | None = None, timeout: float | None = None):
        """Move the arm joints to ``q`` (radians) and wait until settled."""
        r = self.robot
        goal = np.asarray(q, float)
        start = r._target[: r.n_arm].copy()
        speed = speed or self.world.settings.max_joint_speed
        # Long arms sweep the tool fast for small joint motions; cap tool speed too
        # or a held object is flung out of the gripper. Measure the path the tool
        # actually sweeps: a joint-space move that changes arm configuration arcs far
        # from the straight line between its end points.
        pts = np.array([r.kin.tcp(start + (goal - start) * s) for s in np.linspace(0, 1, 17)])
        tool = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) / self.world.settings.max_tcp_speed
        duration = max(float(np.max(np.abs(goal - start))) / speed, tool, self.world.dt)
        r._stream(lambda s: r._set_arm(start + (goal - start) * _minjerk(s)), duration)
        r._settle(goal, timeout)

    @action
    def move_to(
        self,
        target,
        approach=DOWN,
        yaw: float | None = None,
        linear: bool = False,
        speed: float = 0.15,
        timeout: float | None = None,
        level: bool = False,
        plan: bool = False,
    ):
        """Move the tool centre point to ``target`` (a point, tuple or object handle).

        ``approach`` is the direction the fingers point (default straight down).
        ``linear=True`` follows a straight line in Cartesian space at ``speed`` m/s,
        which is what you want for the last few centimetres of a grasp. ``level``
        keeps the fingers closing horizontally (a side grasp). ``plan=True`` finds a
        path that keeps the arm (and anything held) off the table and the objects.
        """
        r = self.robot
        p = as_subject(self.world, target).position
        q_now = r._target[: r.n_arm].copy()
        if not linear:
            if plan:
                q = r._clear_ik(p, approach, yaw, level)
                self.follow.__wrapped__(self, r.planner.plan(q_now, q), timeout=timeout)
                return
            q, err = r._ik(p, q_now, approach, yaw, level=level)
            self.move_joints.__wrapped__(self, q, timeout=timeout)
            return
        p0 = r.kin.tcp(q_now)
        n = max(2, int(np.ceil(np.linalg.norm(p - p0) / 0.004)))
        qs, q = [], q_now
        for i in range(1, n + 1):
            # Hold the current posture along a straight line: pulling a redundant arm
            # toward home here makes its elbow and wrist drift while the hand moves.
            q, err = r._ik(p0 + (p - p0) * i / n, q, approach, yaw, rest=q_now, level=level)
            qs.append(q)
        qs = np.array([q_now, *qs])
        # A short straight line can still need a big wrist turn (to a new grasp yaw);
        # bound joint speed as well, or the wrist spins fast enough to fling what it holds.
        turn = float(np.max(np.abs(qs[-1] - qs[0]))) / self.world.settings.max_joint_speed
        duration = max(float(np.linalg.norm(p - p0)) / speed, turn, self.world.dt)

        def at(s):
            x = _minjerk(s) * (len(qs) - 1)
            i = min(int(x), len(qs) - 2)
            r._set_arm(qs[i] + (qs[i + 1] - qs[i]) * (x - i))

        r._stream(at, duration)
        r._settle(qs[-1], timeout)

    @action
    def follow(self, path, timeout: float | None = None):
        """Move the arm along joint-space waypoints ``path`` (from robot.planner), smoothly."""
        r = self.robot
        path = [r._target[: r.n_arm].copy(), *(np.asarray(q, float) for q in path)]
        seg = np.array([np.max(np.abs(b - a)) for a, b in zip(path, path[1:])])
        if seg.sum() < 1e-9:
            return
        at = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum()
        fine = np.array([r.kin.tcp(_along(path, at, s)) for s in np.linspace(0, 1, 8 * len(path) + 1)])
        tool = float(np.linalg.norm(np.diff(fine, axis=0), axis=1).sum()) / self.world.settings.max_tcp_speed
        duration = max(float(seg.sum()) / self.world.settings.max_joint_speed, tool, self.world.dt)
        r._stream(lambda s: r._set_arm(_along(path, at, _minjerk(s))), duration)
        r._settle(path[-1], timeout)

    @action
    def home(self, timeout: float | None = None):
        self.move_joints.__wrapped__(self, self.robot.home_q, timeout=timeout)


class Gripper(Subject):
    _label = "gripper"
    name = "gripper"

    def __init__(self, robot):
        self.robot = robot
        self.world = robot.world

    @property
    def position(self):
        return self.robot.tcp.position

    @property
    def opening(self) -> float:
        """0 = fully closed, 1 = fully open."""
        return float(np.clip(self.robot.true_qpos()[-1], 0, 1))

    def touching(self) -> tuple[set, set]:
        """Objects touching (left finger, right finger)."""
        self.world.require("contacts", "gripper contact sensing")
        fixed, moving = set(), set()
        for c in self.world.backend.contacts():
            for me, other in ((c.a, c.b), (c.b, c.a)):
                if me == "robot:left_finger" and not other.startswith("robot:"):
                    fixed.add(other)
                elif me == "robot:right_finger" and not other.startswith("robot:"):
                    moving.add(other)
        return fixed, moving

    def holding(self) -> str | None:
        fixed, moving = self.touching()
        both = (fixed & moving) - {"floor"}
        return sorted(both)[0] if both else None

    @action
    def open(self, amount: float = 1.0, timeout: float | None = None):
        self._go(GRIPPER_CLOSED + amount * (GRIPPER_OPEN - GRIPPER_CLOSED), timeout)

    @action
    def close(self, timeout: float | None = None):
        """Close until the jaws stop moving - on an object or fully shut."""
        self._go(GRIPPER_CLOSED, timeout, stall_ok=True)

    def _go(self, opening, timeout, stall_ok=False):
        r = self.robot
        start = r._target[-1]
        duration = max(abs(opening - start) * 0.35, self.world.dt)
        r._stream(lambda s: r._set_gripper(start + (opening - start) * _minjerk(s)), duration)
        timeout = timeout or self.world.settings.action_timeout

        # Stopped means the opening held within 0.5% of the stroke for 0.1 s: judged from
        # positions, as an encoder would, because a jaw squeezing hard reads a solver's
        # jitter of a few hundredths per second (Genesis, PhysX) though it does not move.
        recent = deque(maxlen=max(2, round(0.1 / self.world.dt) + 1))

        def done():
            now = r.true_qpos()[-1]
            recent.append(now)
            still = len(recent) == recent.maxlen and max(recent) - min(recent) < 0.005
            return still and (stall_ok or abs(now - opening) < 0.05)

        if not self.world.run_until(done, timeout):
            raise ActionTimeoutError(f"gripper did not reach opening {opening:.2f} within {timeout}s (at {r.true_qpos()[-1]:.2f})")


def _footprint(spec) -> float:
    """Radius of the circle an object covers on the floor."""
    if spec.kind in ("box", "bin"):
        return float(np.hypot(spec.size[0], spec.size[1]))
    return float(spec.size[0])


class Robot:
    _label = "robot"
    name = "robot"

    def __init__(self, world):
        self.world = world
        self.model = world.backend.robot_model
        self.kin = _kinematics(self.model.name)
        self.n_arm = self.model.n_arm
        self.arm = Arm(self)
        self.gripper = Gripper(self)
        self.tcp = TCP(self)
        b = world.backend
        self._target = b.qpos().copy()
        self._target[-1] = GRIPPER_OPEN
        self._home_q = None
        self._planner = None
        self._held_from = None  # after a side pick: (approach, yaw, TCP height above the object's underside)

    @property
    def base(self):
        raise AttributeError(
            f"{self.model.title} is a fixed-base arm: robot.base exists on legged robots. Arms have robot.arm, robot.gripper and robot.tcp"
        )

    @property
    def grip_yaw(self) -> float:
        """World angle of the finger-closing axis about z (modulo pi)."""
        T = self.kin.fk(self.true_qpos()[: self.n_arm])
        g = T[:3, :3] @ self.kin.grip_axis
        return float(np.arctan2(g[1], g[0]))

    @property
    def min_grasp_z(self) -> float:
        """Lowest TCP height for a top-down grasp that keeps the fingertips off the table."""
        return self.model.derived.finger_reach + TABLE_CLEARANCE

    @property
    def joint_names(self) -> list[str]:
        return self.world.backend.joint_names

    @property
    def home_q(self) -> np.ndarray:
        if self._home_q is None:
            self._home_q = home_q(self.model.name)
        return self._home_q

    def true_qpos(self) -> np.ndarray:
        return self.world.backend.qpos()

    def qpos(self) -> np.ndarray:
        """Joint readings as the robot sees them (with any injected sensor noise)."""
        return self.world.faults.filter_qpos(self.world.backend.qpos())

    @property
    def joints(self) -> dict[str, float]:
        return dict(zip(self.joint_names, map(float, self.qpos())))

    def reset_to(self, q_arm=None, gripper: float = GRIPPER_OPEN):
        """Teleport to a joint configuration (setup only; simulators with state support).

        ``gripper`` is an opening between 0 (closed) and 1 (open).
        """
        q = np.concatenate([self.home_q if q_arm is None else np.asarray(q_arm, float), [gripper]])
        self.world.backend.set_joint_positions(q)
        self._target = q.copy()
        if self.world.trace:
            args = {"q": [float(x) for x in q]}
            if q_arm is None and gripper == GRIPPER_OPEN:
                args["default"] = True  # codegen writes robot.reset_to(): same pose, readable
            self.world.trace.event("edit", "reset_to", args)

    # internals --------------------------------------------------------------
    def _set_arm(self, q):
        self._target[: self.n_arm] = q

    def _set_gripper(self, a):
        self._target[-1] = a

    def _stream(self, at, duration):
        n = max(1, int(round(duration / self.world.dt)))
        for i in range(1, n + 1):
            at(i / n)
            self.world.step()

    def _settle(self, goal, timeout, tol: float = 0.03):
        timeout = timeout or self.world.settings.action_timeout
        w = self.world
        n = self.n_arm

        def settled():
            # Stopped near the goal - or held well inside the tolerance while a servo
            # dithers (a held object can sustain a small limit cycle in a stiff wrist).
            q = self.qpos()[:n]
            err = np.max(np.abs(q - goal))
            return err < tol / 4 or err < tol and np.max(np.abs(w.backend.qvel()[:n])) < 0.08

        if not w.run_until(settled, timeout, hold=0.06):
            err = np.abs(self.true_qpos()[:n] - goal)
            j = self.joint_names[int(np.argmax(err))]
            what = "arm" if self.model.family == "arm" else "robot"
            raise ActionTimeoutError(f"{what} did not settle within {timeout}s; worst joint {j} is {err.max():.3f} rad off target")

    @property
    def planner(self):
        """Collision checking and path search for this arm in this scene (see robowright.planning)."""
        if self._planner is None:
            from .planning import Planner

            self._planner = Planner(self.world)
        return self._planner

    def _clear_ik(self, p, approach, yaw, level=False) -> np.ndarray:
        """Joint angles reaching ``p`` with the arm (and what it holds) clear of the table and
        the objects, nearest the arm's present ones."""
        return self._clear_iks(p, approach, yaw, level)[0]

    def _clear_iks(self, p, approach, yaw, level=False, want: int = 3) -> list[np.ndarray]:
        """Up to ``want`` such joint configurations, nearest first."""
        held = self.gripper.holding() if self.world.has_contacts and self.model.has_gripper else None
        self.planner.sync(held)
        seed = self._target[: self.n_arm].copy()
        yaws = [None] if yaw is None else [yaw] if level else [yaw + k * np.pi / 2 for k in (0, 1, -1, 2)]
        found = []
        for s in [seed, self.home_q, *_far_seeds(self.kin, 16)]:
            for y in yaws:
                q, err = self.kin.ik(p, s, approach, y, rest=seed, level=level)
                if err < 1e-3 and (y is None or _yaw_error(self.kin, q, y) < np.radians(2)) and self.planner.clear(q):
                    found.append(q)
            if len(found) >= want:
                break
        if not found:
            raise UnreachableError(f"no joint configuration reaches {np.round(p, 3).tolist()} clear of the table and objects")
        return sorted(found, key=lambda q: float(np.abs(q - seed).sum()))

    def _straight(self, q0, p1, approach, yaw, level, allow=()) -> list[np.ndarray] | None:
        """The joint angles a straight move of the tool from where ``q0`` puts it to ``p1``
        passes through (as Arm.move_to steps it), or None if they jump (IK turning to another
        solution at a joint limit) or come near the table or an object (except ``allow``)."""
        p0 = self.kin.tcp(q0)
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / 0.004)))
        qs, q = [], q0
        allowed = {(g, o) for g in self.planner.hand_geoms for o in allow}
        for i in range(1, n + 1):
            nxt, err = solve_ik(self.kin, p0 + (p1 - p0) * i / n, q, self.home_q, approach, yaw, q0, level)
            if err > 1e-3 or np.max(np.abs(nxt - q)) > 0.15 or self.planner.contacts(nxt) - allowed - self.planner.base_contacts:
                return None
            qs.append(q := nxt)
        return qs

    def _ik(self, p, seed, approach, yaw, rest=None, level=False):
        q, err = solve_ik(self.kin, p, seed, self.home_q, approach, yaw, rest, level)
        if err > 5e-3:
            raise UnreachableError(f"no joint configuration reaches {np.round(p, 3).tolist()} (closest {err * 1000:.1f} mm)")
        return q, err

    # skills ------------------------------------------------------------------
    @action
    def pick(self, obj, lift: float = 0.05, timeout: float | None = None, approach="top"):
        """Grasp ``obj``, then lift. Returns once the grasp is checked.

        ``approach`` is ``"top"`` (from above), ``"side"`` (horizontally: for a tall object, or
        one under something), or the horizontal direction to come in along. A side grasp
        plans its way to the object (see :attr:`planner`), the hand turned to the side away
        from the table and the objects.
        """
        o = as_subject(self.world, obj)
        if approach not in (None, "top"):
            return self._pick_from_side(o, approach, lift, timeout)
        if isinstance(o, ObjectHandle) and o.spec.kind != "bin":
            width = grasp_width(o.spec)
            opens = widest_gap(self.model)
            if width > opens:
                raise GraspError(f"pick({o.name!r}): {o.name} is {width * 1000:.0f} mm across and this gripper opens {opens * 1000:.0f} mm")
        p = o.position
        yaw = getattr(o, "yaw", 0.0)
        self.gripper.open.__wrapped__(self.gripper)
        grasp = p.copy()
        if isinstance(o, ObjectHandle):
            grasp[2] = max(p[2], self.min_grasp_z)
        self.arm.move_to.__wrapped__(self.arm, grasp + [0, 0, 0.05], yaw=yaw, timeout=timeout)
        self.arm.move_to.__wrapped__(self.arm, grasp, yaw=yaw, linear=True, timeout=timeout)
        self.gripper.close.__wrapped__(self.gripper)
        self.arm.move_to.__wrapped__(self.arm, grasp + [0, 0, lift], yaw=yaw, linear=True, timeout=timeout)
        w = self.world
        if isinstance(o, ObjectHandle) and w.has_contacts:
            # A pick that lifted nothing must say so here, not leave a later check to puzzle over
            # an empty gripper. Already holding costs no time; a jaw chattering on the object
            # gets a moment to settle.
            if not w.run_until(lambda: self.gripper.holding() == o.name, 0.2):
                fixed, moving = self.gripper.touching()
                raise GraspError(
                    f"pick({o.name!r}) lifted without it: {o.name} is at {np.round(o.position, 3).tolist()}, "
                    f"the tool at {np.round(self.tcp.position, 3).tolist()}, the jaws at opening {self.gripper.opening:.2f}, "
                    f"touching {sorted(fixed) or 'nothing'} / {sorted(moving) or 'nothing'}"
                )

    def _pick_from_side(self, o, approach, lift, timeout):
        if isinstance(o, ObjectHandle):
            width, opens = grasp_width(o.spec), widest_gap(self.model)
            if width > opens:
                raise GraspError(f"pick({o.name!r}): {o.name} is {width * 1000:.0f} mm across and this gripper opens {opens * 1000:.0f} mm")
        p = o.position
        grasp = p.copy()
        # High on the object (2.5 cm under its top): the arm then stays well off the table.
        top = o.top if isinstance(o, ObjectHandle) else p[2]
        grasp[2] = max(p[2], top - 0.025, self.model.derived.side_reach + TABLE_CLEARANCE)
        drop = float(grasp[2] - _bottom(o)) if isinstance(o, ObjectHandle) else 0.0  # the TCP above the underside
        if isinstance(approach, str):
            if approach != "side":
                raise ValueError(f"approach must be 'top', 'side' or a horizontal direction, not {approach!r}")
            v = p[:2] - np.asarray(self.model.base_pos[:2], float)
            base = float(np.arctan2(v[1], v[0]))
            # From the robot's side of the object first; turned 45 then 90 degrees either way if
            # the arm cannot fold its hand in between itself and the object.
            angles = [base + t for t in (0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2)]
        else:
            angles = [float(np.arctan2(approach[1], approach[0]))]
        self.gripper.open.__wrapped__(self.gripper)
        geoms = {g for g in range(self.planner.m.ngeom) if self.planner.m.geom_bodyid[g] == self.planner.objects.get(o.name, -1)}
        why = []
        for angle in angles:
            a = np.array([np.cos(angle), np.sin(angle), 0.0])
            yaw = angle + np.pi / 2  # the fingers close across the approach
            along = f"along {np.round(a[:2], 2).tolist()}"
            try:
                backs = self._clear_iks(grasp - a * 0.06, a, yaw, level=True, want=4)
            except UnreachableError:
                why.append(f"{along}: no pose holds the hand level there clear of the table and objects")
                continue
            # In and up must be straight, smooth moves clear of everything but the object itself.
            q_back = next(
                (
                    q
                    for q in backs
                    if (inn := self._straight(q, grasp, a, yaw, True, geoms))
                    and self._straight(inn[-1], grasp + [0, 0, lift], a, yaw, True, geoms)
                ),
                None,
            )
            if q_back is None:
                why.append(f"{along}: no smooth straight way in and up")
                continue
            kw = dict(approach=a, yaw=yaw, level=True, timeout=timeout)
            self.arm.follow.__wrapped__(self.arm, self.planner.plan(self._target[: self.n_arm], q_back), timeout=timeout)
            self.arm.move_to.__wrapped__(self.arm, grasp, linear=True, **kw)
            self.gripper.close.__wrapped__(self.gripper)
            self.arm.move_to.__wrapped__(self.arm, grasp + [0, 0, lift], linear=True, **kw)
            self._held_from = (a, yaw, drop) if isinstance(o, ObjectHandle) else None
            self._check_held(o)
            return
        raise UnreachableError(f"pick({o.name!r}) from the side at {np.round(grasp, 3).tolist()}: " + "; ".join(why))

    def _check_held(self, o):
        w = self.world
        if isinstance(o, ObjectHandle) and w.has_contacts:
            if not w.run_until(lambda: self.gripper.holding() == o.name, 0.2):
                fixed, moving = self.gripper.touching()
                raise GraspError(
                    f"pick({o.name!r}) lifted without it: {o.name} is at {np.round(o.position, 3).tolist()}, "
                    f"the tool at {np.round(self.tcp.position, 3).tolist()}, the jaws at opening {self.gripper.opening:.2f}, "
                    f"touching {sorted(fixed) or 'nothing'} / {sorted(moving) or 'nothing'}"
                )

    @action
    def place(self, on, height: float | None = None, yaw: float | None = None, timeout: float | None = None):
        """Carry the held object above ``on`` (object or point), lower it and release.

        ``height`` is the TCP's height above ``on``'s top at release; by default
        the fingertips stop just clear of it. ``yaw`` turns the grip to that angle
        on the way; by default the hand keeps its orientation, because turning the
        wrist under load is what most often shakes a weakly held object loose.
        """
        t = as_subject(self.world, on)
        p = t.position.copy()
        top = t.top if isinstance(t, ObjectHandle) else p[2]
        if self._held_from is not None:
            return self._place_from_side(t, p, top, height, timeout)
        if yaw is None:
            yaw = self.grip_yaw
        if height is None:
            height = max(0.015, self.model.derived.finger_reach + 0.006)
        if isinstance(t, ObjectHandle) and t.spec.kind == "bin":
            p[:2] = self._free_spot(t, top + height + 0.03, yaw)
        above = np.array([p[0], p[1], top + height + 0.03])
        # Rise straight up to the transit height first: a joint-space move from a low
        # lift dips on its way across and drags the held object through the target's rim.
        # Small arms cannot always reach that high where they are; then go direct.
        here = self.tcp.position
        if here[2] < above[2] - 0.005:
            try:
                self._ik([here[0], here[1], above[2]], self._target[: self.n_arm], DOWN, yaw)
            except UnreachableError:
                pass
            else:
                self.arm.move_to.__wrapped__(self.arm, [here[0], here[1], above[2]], yaw=yaw, linear=True, timeout=timeout)
        self.arm.move_to.__wrapped__(self.arm, above, yaw=yaw, timeout=timeout)
        self.arm.move_to.__wrapped__(self.arm, [p[0], p[1], top + height], yaw=yaw, linear=True, timeout=timeout)
        self.gripper.open.__wrapped__(self.gripper)
        self.arm.move_to.__wrapped__(self.arm, above, yaw=yaw, linear=True, timeout=timeout)

    def _place_from_side(self, t, p, top, height, timeout):
        """Set down what a side grasp holds: planned to above the spot, down, let go, back out."""
        a, yaw, drop = self._held_from
        lower_in = height is None and isinstance(t, ObjectHandle) and t.spec.kind == "bin"
        if height is None:
            height = drop + 0.01  # the object's underside 1 cm above the surface
        if isinstance(t, ObjectHandle) and t.spec.kind == "bin":
            p[:2] = self._free_spot(t, top + height + 0.03, yaw)
        down = np.array([p[0], p[1], top + height])
        above = down + [0, 0, 0.04]
        kw = dict(approach=a, yaw=yaw, level=True, timeout=timeout)
        held = self.gripper.holding() if self.world.has_contacts else None
        ignore = {g for g in range(self.planner.m.ngeom) if self.planner.m.geom_bodyid[g] == self.planner.objects.get(held, -1)}
        self.planner.sync(held)
        q_now = self._target[: self.n_arm].copy()
        carry = self._straight(q_now, above, a, yaw, True)
        if lower_in:
            # Down into the bin as far as the hand and the object stay clear of its walls and
            # floor, so the object is let go of near the bottom rather than dropped from the rim.
            floor = t.position[2] + 0.006 + drop + 0.01
            for z in np.arange(down[2] - 0.01, floor - 1e-9, -0.01):
                if not self._straight(carry[-1] if carry else q_now, [down[0], down[1], z], a, yaw, True):
                    break
                down = np.array([down[0], down[1], z])
        if carry and self._straight(carry[-1], down, a, yaw, True, ignore):
            # Carried straight, the hand level all the way: a joint-space path tilts it between
            # two level poses, and an object held high up swings out of the fingers.
            self.arm.move_to.__wrapped__(self.arm, above, linear=True, **kw)
        else:
            q = next((q for q in self._clear_iks(above, a, yaw, level=True, want=4) if self._straight(q, down, a, yaw, True, ignore)), None)
            if q is None:
                raise UnreachableError(f"place: no clear straight way down to {np.round(down, 3).tolist()} holding from the side")
            self.arm.follow.__wrapped__(self.arm, self.planner.plan(q_now, q), timeout=timeout)
        self.arm.move_to.__wrapped__(self.arm, down, linear=True, **kw)
        self.gripper.open.__wrapped__(self.gripper)
        self._held_from = None
        self.planner.sync()
        # Back out the way the fingers came in, if that way is clear; else straight up off the object.
        q_now = self._target[: self.n_arm].copy()
        if self._straight(q_now, down - a * 0.06, a, yaw, True, ignore):
            self.arm.move_to.__wrapped__(self.arm, down - a * 0.06, linear=True, **kw)
            down = down - a * 0.06
        for rise in (0.06, 0.04, 0.02):  # up off the object, as far as the arm reaches
            if self._straight(self._target[: self.n_arm].copy(), down + [0, 0, rise], a, yaw, True, ignore):
                self.arm.move_to.__wrapped__(self.arm, down + [0, 0, rise], linear=True, **kw)
                break

    def _free_spot(self, bin, z: float, yaw: float) -> np.ndarray:
        """Where in ``bin`` to drop the held object: its centre while empty, else the reachable
        spot inside it farthest from what is already there, so objects are not stacked."""
        w = self.world
        centre = bin.position[:2].copy()
        if not w.has_ground_truth:
            return centre
        lo, hi = bin.bounds()
        held = self.gripper.holding() if w.has_contacts else None
        inside = []
        for name in w.object_names:
            o = w.scene[name]
            if name in (bin.name, held) or o.spec.static:
                continue
            q = o.position
            if np.all(q[:2] > lo[:2]) and np.all(q[:2] < hi[:2]) and q[2] < hi[2] + 0.05:
                inside.append((q[:2], _footprint(o.spec)))
        if not inside:
            return centre
        r = _footprint(w.spec.object(held)) if held else 0.02
        xs = np.linspace(lo[0] + r + 0.004, hi[0] - r - 0.004, 9)
        ys = np.linspace(lo[1] + r + 0.004, hi[1] - r - 0.004, 9)
        if xs[0] > xs[-1] or ys[0] > ys[-1]:  # the bin is barely wider than the object
            return centre
        best, best_score = centre, -np.inf
        for x in xs:
            for y in ys:
                c = np.array([x, y])
                clear = min(float(np.linalg.norm(c - q)) - rq for q, rq in inside) - r
                score = clear - 1e-3 * float(np.linalg.norm(c - centre))  # ties: nearer the centre
                if score <= best_score:
                    continue
                try:
                    self._ik([x, y, z], self._target[: self.n_arm], DOWN, yaw)
                except UnreachableError:
                    continue
                best, best_score = c, score
        return best

    def observe(self, cameras=(), privileged: bool = False, task: str | None = None) -> dict:
        """What a policy sees: joint readings, optional camera images, optional ground truth."""
        w = self.world
        obs = {"qpos": self.qpos(), "t": w.time, "task": task, "robot": self.model.name}
        if cameras:
            obs["images"] = {c: w.faults.filter_image(w.backend.render(c, *w.settings.image_size)) for c in cameras}
        if privileged:
            obs["objects"] = {n: w.backend.object_pose(n) for n in w.object_names}
            obs["tcp"] = self.tcp.position
        return obs

    @action
    def run_policy(
        self,
        policy,
        task: str | None = None,
        until=None,
        timeout: float = 20.0,
        cameras=(),
        privileged: bool = False,
        hold: float = 0.0,
    ):
        """Run a policy (``obs -> joint targets`` or an action chunk) until ``until`` or ``timeout``.

        ``until`` is a zero-argument predicate, typically built with
        :func:`robowright.condition`; with ``hold`` it must stay true that long
        while the policy keeps running (an object carried into a bin is "inside"
        before it is let go). Returns a :class:`Rollout` summary; it does not
        raise when the condition is not met - assert on it.
        """
        w = self.world
        if hasattr(policy, "reset"):
            policy.reset()
        queue: list = []
        t0, wall0, steps, infer_ms = w.time, _time.perf_counter(), 0, []
        met = False
        since = None
        while w.time - t0 < timeout - 1e-9:
            if until is not None and until():
                since = w.time if since is None else since
                if w.time - since >= hold - 1e-9:
                    met = True
                    break
            else:
                since = None
            if not queue:
                obs = self.observe(cameras, privileged, task)
                ti = _time.perf_counter()
                act = np.asarray(policy(obs), float)
                infer_ms.append((_time.perf_counter() - ti) * 1000)
                queue = list(act) if act.ndim == 2 else [act]
            a = queue.pop(0)
            self._target[: len(a)] = a
            w.step()
            steps += 1
        else:
            met = bool(until()) if until is not None else True
        return Rollout(met, steps, w.time - t0, _time.perf_counter() - wall0, infer_ms)


class Rollout:
    def __init__(self, success, steps, sim_seconds, wall_seconds, infer_ms):
        self.success = success
        self.steps = steps
        self.sim_seconds = sim_seconds
        self.wall_seconds = wall_seconds
        self.infer_ms = infer_ms

    def __bool__(self):
        return self.success

    def __repr__(self):
        return f"<Rollout success={self.success} steps={self.steps} sim={self.sim_seconds:.2f}s>"


@functools.cache
def _kinematics(name: str) -> Kinematics:
    from . import robots

    return Kinematics(robots.get(name))


def solve_ik(kin: Kinematics, p, seed, home, approach=DOWN, yaw=None, rest=None, level=False):
    """IK over the grasp yaw's symmetric variants and two seeds.

    Among the solutions that reach the target, take the one that moves the
    joints least: a square object can be gripped (or set down) at any multiple
    of 90 degrees, and turning the wrist further than needed while holding it
    shakes it loose. ``rest`` is the posture redundant arms drift toward
    (default ``home``).
    """
    rest = home if rest is None else rest
    seed = np.asarray(seed, float)
    # Level (a side grasp), the fingers close across the approach: no quarter turns.
    yaws = [None] if yaw is None else [yaw] if level else [yaw + k * np.pi / 2 for k in (0, 1, -1, 2)]
    cands = []
    for y in yaws:
        for s in (seed, home):
            q, err = kin.ik(p, s, approach, y, rest=rest, level=level)
            # A solution reaches the target only if it also turns the grip as asked: IK
            # gives up on orientation before position, so a candidate can arrive with the
            # wrist short of the yaw (and would win here for having moved least).
            cands.append((float(np.abs(q - seed).sum()), err, q, y is None or _yaw_error(kin, q, y) < np.radians(2)))
    reached = [c for c in cands if c[1] < 1e-3 and c[3]] or [c for c in cands if c[1] < 1e-3]
    if not reached:
        # IK is local: from where the arm is and from home it can stall against a joint limit
        # short of a target other poses reach. Try from mid-range and a few fixed random poses.
        for s in _far_seeds(kin):
            for y in yaws:
                q, err = kin.ik(p, s, approach, y, rest=rest, level=level)
                if err < 1e-3 and (y is None or _yaw_error(kin, q, y) < np.radians(2)):
                    cands.append((float(np.abs(q - seed).sum()), err, q, True))
            if any(c[1] < 1e-3 for c in cands):
                break
        reached = [c for c in cands if c[1] < 1e-3]
    if not reached and yaw is not None:
        # An arm without a wrist roll (four joints, say) cannot choose its grip angle: the base
        # turning towards the target sets it. Grip at whatever angle reaches.
        for s in (seed, home):
            q, err = kin.ik(p, s, approach, None, rest=rest)
            cands.append((float(np.abs(q - seed).sum()), err, q, False))
        reached = [c for c in cands if c[1] < 1e-3]
    _, err, q, _ = min(reached, key=lambda c: c[0]) if reached else min(cands, key=lambda c: c[1])
    return q, err


def _far_seeds(kin: Kinematics, n: int = 4) -> list[np.ndarray]:
    """IK starting poses away from the arm's own: mid-range and ``n`` fixed random ones (with
    the default, the same as robots.detect places the arm with, so what placement reached, a
    move reaches)."""
    rng = np.random.default_rng(0)
    lo, hi = np.maximum(kin.lower, -np.pi), np.minimum(kin.upper, np.pi)
    return [(lo + hi) / 2, *(rng.uniform(lo, hi) for _ in range(n))]


def _yaw_error(kin: Kinematics, q, yaw: float) -> float:
    """How far the grip axis is turned from ``yaw`` about the vertical (modulo pi: fingers are symmetric)."""
    g = kin.fk(q)[:3, :3] @ kin.grip_axis
    if g[0] ** 2 + g[1] ** 2 < 1e-6:
        return 0.0
    return abs((np.arctan2(g[1], g[0]) - yaw + np.pi / 2) % np.pi - np.pi / 2)


@functools.cache
def home_q(name: str) -> np.ndarray:
    from . import robots

    m = robots.get(name)
    kin = _kinematics(name)
    seed = np.asarray(m.seed, float) if m.seed is not None else m.keyframe_q()
    if seed is None:
        seed = np.clip(np.zeros(m.n_arm), kin.lower, kin.upper)
    q, err = kin.ik(m.home, seed, DOWN, yaw=0.0, rest=seed)
    if err > 1e-3:
        # IK is local: a model's own pose can sit in a basin that does not reach home. Try from
        # mid-range and a few fixed random poses (as robots.detect does when it places the arm).
        rng = np.random.default_rng(0)
        lo, hi = np.maximum(kin.lower, -np.pi), np.minimum(kin.upper, np.pi)
        for y in (0.0, None):  # None: an arm with no wrist roll grips at the angle it reaches with
            for s in [seed, (lo + hi) / 2, *(rng.uniform(lo, hi) for _ in range(4))]:
                q, err = kin.ik(m.home, s, DOWN, yaw=y, rest=seed)
                if err <= 1e-3:
                    break
            if err <= 1e-3:
                break
    if err > 1e-3:
        raise UnreachableError(f"{name}: home pose {m.home} unreachable (closest {err * 1000:.1f} mm)")
    q.setflags(write=False)
    return q


def _bottom(o) -> float:
    """Height of an object's underside."""
    return float(o.position[2] - o.spec.half_height)


def widest_gap(model) -> float:
    """The widest object the gripper opens around: its aperture at its widest, less where its
    fingers meet. (A jaw that swings past upright is widest before fully open.)"""
    curve = model.derived.apertures
    return float(max(curve) - curve[0]) if curve else float(model.derived.max_aperture)


def grasp_width(spec) -> float:
    """How wide an object is where a top-down grasp closes on it."""
    return 2 * (min(spec.size[0], spec.size[1]) if spec.kind == "box" else spec.size[0])


def _along(path, at, s: float) -> np.ndarray:
    """The point a fraction ``s`` along piecewise-linear ``path``, its knots at fractions ``at``."""
    i = min(int(np.searchsorted(at, s, side="right")) - 1, len(path) - 2)
    i = max(i, 0)
    f = (s - at[i]) / max(at[i + 1] - at[i], 1e-12)
    return path[i] + (path[i + 1] - path[i]) * min(max(f, 0.0), 1.0)


def _minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5
