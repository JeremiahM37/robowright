"""An interactive session: one simulated world an AI agent (or a person) drives step by step.

This is what the MCP server exposes, kept free of any MCP code so it can be
used and tested on its own. It plays the part a browser tab plays for
Playwright MCP:

* :meth:`Session.snapshot` describes the world as text - the robot's state and
  every object with its name (the *ref* other calls take), shape, pose,
  contacts and what it sits in - the counterpart of an accessibility snapshot.
* Actions (``pick``, ``place``, ``move_to``, ...) go through robowright's own
  API, so they are planned, collision-checked and timed exactly as in a test,
  and each returns the snapshot after it.
* Everything is recorded in a trace, so :meth:`Session.generate_test` turns
  the session into a pytest test that reproduces it bit for bit.

Physics is deterministic: the same launch and the same calls give the same
world, so whatever an agent finds can be replayed and turned into a test.
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path

import numpy as np

from . import robots
from .errors import RobowrightError
from .scene import CameraSpec, ObjectSpec, default_scene
from .world import Settings, World, launch

# Matcher arguments that name another object or subject.
_SUBJECT_ARGS = ("container", "target", "other", "obj")
_MATCHERS = (
    "to_be_near",
    "to_be_inside",
    "to_be_above",
    "to_have_position",
    "to_be_at_rest",
    "to_be_upright",
    "to_be_touching",
    "to_be_holding",
    "to_be_open",
    "to_be_closed",
    "to_have_joint",
    "to_have_no_collisions",
)


def _f(x, n=3) -> str:
    return "[" + ", ".join(f"{float(v):.{n}f}" for v in np.ravel(x)) + "]"


def _half_extents(o: ObjectSpec) -> tuple:
    if o.kind in ("box", "bin"):
        return tuple(o.size)
    if o.kind == "cylinder":
        return (o.size[0], o.size[0], o.size[1])
    return (o.size[0],) * 3


def _views(cam: CameraSpec, objects) -> list[CameraSpec]:
    """A top and a side view of the task area (the objects, else what the default camera sees).

    The top view leans 15 degrees off vertical: straight down, a shiny floor whites out
    under the overhead light and the arm hides what it is reaching for.
    """
    look = np.asarray(cam.lookat, float)
    d = float(np.linalg.norm(np.asarray(cam.pos, float) - look))
    if objects:
        look = np.mean([o.initial_pos for o in objects], axis=0)
        look[2] = 0.0
    tilt = math.radians(15)
    return [
        CameraSpec("top", pos=tuple(look + [0.0, -d * math.sin(tilt), d * math.cos(tilt)]), lookat=tuple(look), fovy=cam.fovy),
        CameraSpec("side", pos=tuple(look + [0.0, -d, 0.15 * d]), lookat=tuple(look), fovy=cam.fovy),
    ]


class Session:
    """One world at a time, launched, driven and closed by calls."""

    def __init__(self, trace_dir: str | Path | None = None):
        self.world: World | None = None
        self.trace_dir = Path(trace_dir or tempfile.mkdtemp(prefix="robowright-session-"))
        self._n = 0
        self.last_trace: Path | None = None

    # --- lifecycle -------------------------------------------------------------
    def launch(self, robot: str = "so101", backend: str = "mujoco", seed: int = 0, objects: list[dict] | None = None) -> str:
        """Start a fresh world (closing any open one) and return its snapshot."""
        if self.world is not None:
            self.close()
        model = robots.get(robot)
        scene = default_scene(model.name)
        if objects is not None:
            scene.objects = [self._object(o) for o in objects]
        scene.cameras = list(scene.cameras) + _views(scene.cameras[0], scene.objects)
        self._n += 1
        # No camera frames in the trace: screenshots are taken on request, and rendering
        # every few steps would cost more than the physics.
        settings = Settings(trace="on", trace_dir=str(self.trace_dir), trace_cameras=[])
        self.world = launch(scene, backend=backend, seed=seed, name=f"session_{self._n}", settings=settings)
        self.world.robot.reset_to()
        return self.snapshot()

    @staticmethod
    def _object(o: dict) -> ObjectSpec:
        o = dict(o)
        if "pos" in o:
            p = list(o["pos"]) + [None] * (3 - len(o["pos"]))
            o["pos"] = tuple(p[:3])
        if "size" in o:
            o["size"] = tuple(float(v) for v in o["size"])
        return ObjectSpec(**o)

    def close(self) -> str:
        """Close the world and keep its trace; returns where it was saved."""
        w = self._w()
        self.world = None
        self.last_trace = w.close(failed=w.status == "failed")
        return f"closed; trace saved to {self.last_trace}"

    def _w(self) -> World:
        if self.world is None:
            raise RobowrightError("no world is open: call launch first")
        return self.world

    # --- observation -----------------------------------------------------------
    def snapshot(self) -> str:
        """The world as text: robot state, every object (by ref), contacts and faults."""
        w = self._w()
        r, b = w.robot, w.backend
        m = b.robot_model
        lines = [f"world: {m.title} ({m.name}) on {b.name}, t={w.time:.3f}s, seed={w.seed}, status={w.status}"]
        if m.family == "legged":
            base = r.base
            tilt = math.degrees(math.acos(float(np.clip(base.up_axis[2], -1, 1))))
            lines += [
                "robot (legged):",
                f"  base: position {_f(base.position)}, height {base.height:.3f} m (standing {m.stand_height:.3f}), "
                f"tilt {tilt:.1f} deg, yaw {math.degrees(base.yaw):.1f} deg",
                f"  base velocity: {_f(base.velocity[:3])} m/s",
            ]
            if w.has_contacts:
                lines.append(f"  base touching: {', '.join(sorted(base.contacts())) or 'nothing'}")
        else:
            g = r.gripper
            lines += ["robot (arm):", f"  tcp: {_f(r.tcp.position)}"]
            if b.has_gripper:
                state = "open" if g.opening > 0.8 else "closed" if g.opening < 0.1 else "partly open"
                line = f"  gripper: opening {g.opening:.2f} ({state})"
                if w.has_contacts:
                    held = g.holding()
                    line += f", holding {held}" if held else ", holding nothing"
                lines.append(line)
        lines.append("  joints: " + ", ".join(f"{k}={v:.3f}" for k, v in zip(b.joint_names, r.qpos())))
        lines.append("objects (refs for other calls):")
        for o in w.spec.objects:
            lines.append("  - " + self._describe(o))
        if not w.spec.objects:
            lines.append("  (none)")
        active = [f.describe() for f in getattr(w.faults, "active", [])]
        if active:
            lines.append("faults: " + "; ".join(", ".join(f"{k}={v}" for k, v in d.items()) for d in active))
        lines.append("cameras: " + ", ".join(c.name for c in w.spec.cameras))
        return "\n".join(lines)

    def _describe(self, o: ObjectSpec) -> str:
        w = self.world
        h = w.scene[o.name]
        dims = "x".join(f"{2000 * h:.0f}" for h in _half_extents(o))
        s = f"{o.name}: {o.color} {o.kind} {dims} mm"
        if w.has_ground_truth:
            s += f", at {_f(h.position)}, yaw {math.degrees(h.yaw):.0f} deg"
            if not o.static:
                v = float(np.linalg.norm(h.velocity[:3]))
                s += ", at rest" if v < 0.01 else f", moving {v:.2f} m/s"
                for c in w.spec.objects:
                    if c.kind == "bin" and c.name != o.name and self._inside(h, w.scene[c.name]):
                        s += f", inside {c.name}"
        if w.has_contacts:
            s += f", touching {', '.join(sorted(h.contacts())) or 'nothing'}"
        return s

    @staticmethod
    def _inside(obj, container) -> bool:
        lo, hi = container.bounds()
        p = obj.position
        return bool(np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and lo[2] - 0.01 < p[2] < hi[2] + 0.05)

    def screenshot(self, camera: str = "front", width: int = 640, height: int = 480) -> bytes:
        """A PNG from one of the world's cameras."""
        from PIL import Image

        w = self._w()
        w.require("render", "screenshots")
        names = [c.name for c in w.spec.cameras]
        if camera not in names:
            raise RobowrightError(f"unknown camera {camera!r}; this world has: {', '.join(names)}")
        buf = io.BytesIO()
        Image.fromarray(w.backend.render(camera, int(width), int(height))).save(buf, format="PNG")
        return buf.getvalue()

    # --- actions -----------------------------------------------------------------
    def _act(self, fn, *args, **kwargs) -> str:
        """Run an action; on failure say why, and still show where things ended up."""
        try:
            self._attempt(fn, *args, **kwargs)
        except RobowrightError as e:
            return f"FAILED: {type(e).__name__}: {e}\n\n{self.snapshot()}"
        return self.snapshot()

    def _attempt(self, fn, *args, **kwargs):
        """Call ``fn``; if it fails, keep the session going and keep the recording reproducible.

        A failed attempt is something for the agent to learn from, not the end of the session.
        If it moved nothing (planning failed), it is dropped from the trace; if the world ran
        during it, codegen replays it inside ``pytest.raises``, so the time it took still passes.
        """
        w = self._w()
        n, step, status = len(w.trace.events), w.step_count, w.status
        try:
            return fn(*args, **kwargs)
        except RobowrightError:
            w.status = status
            new = w.trace.events[n:]
            if w.step_count == step:
                del w.trace.events[n:]
            else:
                for e in new:
                    if e.status == "failed":
                        e.status = "raised"
            raise

    def _target(self, target):
        if isinstance(target, str):
            return self._w().scene[target]
        return tuple(float(v) for v in target)

    def _arm(self):
        r = self._w().robot
        if self.world.backend.robot_model.family != "arm":
            raise RobowrightError(f"{self.world.backend.robot_model.title} is a legged robot: use stand, crouch or move_joints")
        return r

    def pick(self, object: str, approach: str = "top") -> str:
        return self._act(self._arm().pick, self._w().scene[object], approach=approach)

    def place(self, on, height: float | None = None) -> str:
        return self._act(self._arm().place, self._target(on), height=height)

    def move_to(self, target, linear: bool = False, speed: float = 0.15) -> str:
        return self._act(self._arm().arm.move_to, self._target(target), linear=linear, speed=speed)

    def gripper(self, action: str, amount: float = 1.0) -> str:
        g = self._arm().gripper
        if action == "open":
            return self._act(g.open, amount)
        if action == "close":
            return self._act(g.close)
        raise RobowrightError(f"gripper action is 'open' or 'close', not {action!r}")

    def home(self) -> str:
        return self._act(self._arm().arm.home)

    def move_joints(self, joints: dict[str, float]) -> str:
        """Move named joints (radians) to new targets; the others keep theirs."""
        w = self._w()
        r, legged = w.robot, w.backend.robot_model.family == "legged"
        names = list(w.backend.joint_names)
        if not legged:
            names = names[: r.n_arm]  # the gripper has its own call
        q = np.asarray(r.qpos(), float)[: len(names)].copy()
        for k, v in joints.items():
            if k not in names:
                raise RobowrightError(f"unknown joint {k!r}; the joints are: {', '.join(names)}")
            q[names.index(k)] = float(v)
        return self._act((r if legged else r.arm).move_joints, q)

    def stand(self) -> str:
        return self._act(self._legged().stand)

    def crouch(self, depth: float = 0.5) -> str:
        return self._act(self._legged().crouch, depth)

    def _legged(self):
        w = self._w()
        if w.backend.robot_model.family != "legged":
            raise RobowrightError(f"{w.backend.robot_model.title} is an arm: use pick, place, move_to or gripper")
        return w.robot

    def wait(self, seconds: float) -> str:
        return self._act(self._w().wait, float(seconds))

    def push(self, target: str, force, duration: float = 0.1) -> str:
        """A force (newtons, world frame) on an object or ``robot`` (legged base), for ``duration`` seconds."""
        w = self._w()
        if target != "robot":
            w.scene[target]  # unknown names fail here, with the list of objects
        w.faults.push(target, force=tuple(float(f) for f in force), duration=float(duration))
        return self._act(w.wait, float(duration))

    def move_object(self, object: str, position, yaw: float | None = None) -> str:
        w = self._w()
        w.scene[object]
        quat = None if yaw is None else (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        return self._act(w.move_object, object, tuple(float(v) for v in position), quat)

    def fault(self, kind: str, **params) -> str:
        """Inject a fault: joint_noise, action_delay, weak_joint, jitter or camera_dropout."""
        w = self._w()
        allowed = ("joint_noise", "action_delay", "weak_joint", "jitter", "camera_dropout")
        if kind not in allowed:
            raise RobowrightError(f"unknown fault {kind!r}; choose from {', '.join(allowed)}")
        getattr(w.faults, kind)(**params)
        return self.snapshot()

    # --- assertions --------------------------------------------------------------
    def expect(self, subject: str, matcher: str, args: dict | None = None, negate: bool = False, timeout: float | None = None) -> str:
        """Check a robowright matcher; returns PASS or FAIL with the reason, and never ends the session."""
        from .errors import ExpectationError
        from .expect import expect

        w = self._w()
        if matcher not in _MATCHERS:
            raise RobowrightError(f"unknown matcher {matcher!r}; choose from {', '.join(_MATCHERS)}")
        subj = self._subject(subject)
        kw = dict(args or {})
        for k in _SUBJECT_ARGS:
            if isinstance(kw.get(k), str):
                kw[k] = self._subject(kw[k])
        e = expect(subj, timeout=timeout, world=w)
        if negate:
            e = e.not_
        try:
            self._attempt(getattr(e, matcher), **kw)
        except ExpectationError as err:
            return f"FAIL: {err}"
        return f"PASS: expect({subject}){'.not_' if negate else ''}.{matcher}({', '.join(f'{k}={v!r}' for k, v in (args or {}).items())})"

    def _subject(self, name: str):
        r = self._w().robot
        if name in ("robot", "base") and self.world.backend.robot_model.family == "legged":
            return r.base  # a legged robot's pose, velocity and contacts are its base's
        special = {"robot": r, "gripper": getattr(r, "gripper", None), "tcp": getattr(r, "tcp", None)}
        if name == "base":
            return r.base
        if special.get(name) is not None:
            return special[name]
        return self.world.scene[name]

    # --- output ------------------------------------------------------------------
    def generate_test(self, test_name: str = "test_session", path: str | None = None) -> str:
        """The session so far as a pytest test that reproduces it; written to ``path`` if given."""
        from .codegen import generate

        w = self._w()
        tmp = self.trace_dir / f"{w.name}_codegen.zip"
        w.trace.save(tmp)
        code = generate(tmp, test_name=test_name)
        if path:
            Path(path).write_text(code)
        return code

    def crosscheck(self, backend: str) -> str:
        """Make the session's calls again on another engine and say whether the outcome holds there."""
        from .crosscheck import crosscheck

        w = self._w()
        tmp = self.trace_dir / f"{w.name}_crosscheck.zip"
        w.trace.save(tmp)
        return crosscheck(tmp, backend).summary()

    def save_trace(self, path: str | None = None) -> str:
        w = self._w()
        p = w.trace.save(path or self.trace_dir / f"{w.name}.zip")
        return str(p)
