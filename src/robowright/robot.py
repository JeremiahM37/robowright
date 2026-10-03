"""The robot API: actions that wait until they are done.

Every action plans in joint space, streams targets one control period at a
time through the world, then waits until the robot has actually settled -
the robotics version of Playwright's actionability checks. Actions are
recorded in the trace with their arguments so codegen can replay them.
"""

from __future__ import annotations

import functools
import time as _time

import numpy as np

from .errors import ActionTimeoutError, UnreachableError
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
        self, target, approach=DOWN, yaw: float | None = None, linear: bool = False, speed: float = 0.15, timeout: float | None = None
    ):
        """Move the tool centre point to ``target`` (a point, tuple or object handle).

        ``approach`` is the direction the fingers point (default straight down).
        ``linear=True`` follows a straight line in Cartesian space at ``speed`` m/s,
        which is what you want for the last few centimetres of a grasp.
        """
        r = self.robot
        p = as_subject(self.world, target).position
        q_now = r._target[: r.n_arm].copy()
        if not linear:
            q, err = r._ik(p, q_now, approach, yaw)
            self.move_joints.__wrapped__(self, q, timeout=timeout)
            return
        p0 = r.kin.tcp(q_now)
        n = max(2, int(np.ceil(np.linalg.norm(p - p0) / 0.004)))
        qs, q = [], q_now
        for i in range(1, n + 1):
            # Hold the current posture along a straight line: pulling a redundant arm
            # toward home here makes its elbow and wrist drift while the hand moves.
            q, err = r._ik(p0 + (p - p0) * i / n, q, approach, yaw, rest=q_now)
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

        def done():
            v = abs(self.world.backend.qvel()[-1])
            return v < 0.04 and (stall_ok or abs(r.true_qpos()[-1] - opening) < 0.05)

        if not self.world.run_until(done, timeout, hold=0.1):
            raise ActionTimeoutError(f"gripper did not reach opening {opening:.2f} within {timeout}s (at {r.true_qpos()[-1]:.2f})")


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
            self.world.trace.event("edit", "reset_to", {"q": [float(x) for x in q]})

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

    def _ik(self, p, seed, approach, yaw, rest=None):
        q, err = solve_ik(self.kin, p, seed, self.home_q, approach, yaw, rest)
        if err > 5e-3:
            raise UnreachableError(f"no joint configuration reaches {np.round(p, 3).tolist()} (closest {err * 1000:.1f} mm)")
        return q, err

    # skills ------------------------------------------------------------------
    @action
    def pick(self, obj, lift: float = 0.05, timeout: float | None = None):
        """Top-down grasp of ``obj``, then lift. Returns once the grasp is checked."""
        o = as_subject(self.world, obj)
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
        if yaw is None:
            yaw = self.grip_yaw
        if height is None:
            height = max(0.015, self.model.derived.finger_reach + 0.006)
        above = np.array([p[0], p[1], top + height + 0.03])
        # Rise straight up to the transit height first: a joint-space move from a low
        # lift dips on its way across and drags the held object through the target's rim.
        here = self.tcp.position
        if here[2] < above[2] - 0.005:
            self.arm.move_to.__wrapped__(self.arm, [here[0], here[1], above[2]], linear=True, timeout=timeout)
        self.arm.move_to.__wrapped__(self.arm, above, yaw=yaw, timeout=timeout)
        self.arm.move_to.__wrapped__(self.arm, [p[0], p[1], top + height], yaw=yaw, linear=True, timeout=timeout)
        self.gripper.open.__wrapped__(self.gripper)
        self.arm.move_to.__wrapped__(self.arm, above, yaw=yaw, linear=True, timeout=timeout)

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
    def run_policy(self, policy, task: str | None = None, until=None, timeout: float = 20.0, cameras=(), privileged: bool = False):
        """Run a policy (``obs -> joint targets`` or an action chunk) until ``until`` or ``timeout``.

        ``until`` is a zero-argument predicate, typically built with
        :func:`robowright.condition`. Returns a :class:`Rollout` summary;
        it does not raise when the condition is not met - assert on it.
        """
        w = self.world
        if hasattr(policy, "reset"):
            policy.reset()
        queue: list = []
        t0, wall0, steps, infer_ms = w.time, _time.perf_counter(), 0, []
        met = False
        while w.time - t0 < timeout - 1e-9:
            if until is not None and until():
                met = True
                break
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


def solve_ik(kin: Kinematics, p, seed, home, approach=DOWN, yaw=None, rest=None):
    """IK over the grasp yaw's symmetric variants and two seeds.

    Among the solutions that reach the target, take the one that moves the
    joints least: a square object can be gripped (or set down) at any multiple
    of 90 degrees, and turning the wrist further than needed while holding it
    shakes it loose. ``rest`` is the posture redundant arms drift toward
    (default ``home``).
    """
    rest = home if rest is None else rest
    seed = np.asarray(seed, float)
    yaws = [None] if yaw is None else [yaw + k * np.pi / 2 for k in (0, 1, -1, 2)]
    cands = []
    for y in yaws:
        for s in (seed, home):
            q, err = kin.ik(p, s, approach, y, rest=rest)
            cands.append((float(np.abs(q - seed).sum()), err, q))
    reached = [c for c in cands if c[1] < 1e-3]
    _, err, q = min(reached, key=lambda c: c[0]) if reached else min(cands, key=lambda c: c[1])
    return q, err


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
        raise UnreachableError(f"{name}: home pose {m.home} unreachable (closest {err * 1000:.1f} mm)")
    q.setflags(write=False)
    return q


def _minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5
