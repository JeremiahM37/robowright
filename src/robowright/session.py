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
        self.view = None  # a LiveView, once watch() is called
        self.bridge = None  # a Ros2Bridge, once serve_ros2() is called
        self.process = None  # the code under test, once start_process() is called
        self.trace_dir = Path(trace_dir or tempfile.mkdtemp(prefix="robowright-session-"))
        self._n = 0
        self.last_trace: Path | None = None

    # --- lifecycle -------------------------------------------------------------
    def launch(
        self,
        robot: str = "so101",
        backend: str = "mujoco",
        seed: int = 0,
        objects: list[dict] | None = None,
        fidelity: str | None = None,
        ros2: dict | None = None,
        robot_options: dict | None = None,
    ) -> str:
        """Start a fresh world (closing any open one) and return its snapshot.

        ``backend="ros2"`` connects to a robot behind ROS 2 instead (a real arm, Gazebo, Isaac
        Sim's bridge), with ``ros2`` its settings (topics, controllers, TF frames, ``gazebo``: see
        ``robowright.backends.ros2_backend``). ``robot_options``: for a robot given as a model
        file, how to read it (``{"gripper": false, "base_pos": [0, 0, 0]}``: see ``robots.load``)."""
        if self.world is not None:
            self.close()
        if robot_options and not robots.is_file(robot):
            raise RobowrightError(f"robot_options are for a robot given as a model file; {robot!r} is a built-in robot")
        opts = {k: tuple(v) if isinstance(v, list) else v for k, v in (robot_options or {}).items()}
        model = robots.load(robot, **opts) if opts else robots.get(robot)
        scene = default_scene(model.name)
        if objects is not None:
            scene.objects = [self._object(o) for o in objects]
        scene.cameras = list(scene.cameras) + _views(scene.cameras[0], scene.objects)
        self._n += 1
        # No camera frames in the trace: screenshots are taken on request, and rendering
        # every few steps would cost more than the physics.
        settings = Settings(trace="on", trace_dir=str(self.trace_dir), trace_cameras=[], fidelity=fidelity)
        if backend == "ros2" and not (ros2 or {}).get("gazebo"):
            scene.cameras = scene.cameras[:1]  # a ROS 2 robot has the cameras its settings name (in Gazebo, any)
        self.world = launch(scene, backend=backend, seed=seed, name=f"session_{self._n}", settings=settings, **(ros2 or {}))
        self.world.robot.reset_to()
        if self.view is not None:
            self.view.watch(self.world)
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
        self.stop_process()
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
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

    def pick(self, object: str, approach: str | list = "top") -> str:
        approach = approach if isinstance(approach, str) else tuple(float(v) for v in approach)
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

    # --- the code under test ------------------------------------------------------
    def run_policy(
        self,
        policy: str,
        kwargs: dict | None = None,
        until: dict | None = None,
        timeout: float = 20.0,
        hold: float = 0.0,
    ) -> str:
        """Run a controller or policy (``"package.module:name"``: a callable ``obs -> joint targets``,
        a class taking ``kwargs``, or a learned policy a LearnedPolicy loads) until ``until`` holds
        (``{"subject": "cube", "matcher": "to_be_inside", "args": {"container": "bin"}}``) or
        ``timeout`` passes. Returns whether it succeeded, how long it took, and the snapshot."""
        from .expect import condition

        w = self._w()
        pol = _load_policy(policy, kwargs or {})
        done = None
        if until:
            kw = dict(until.get("args") or {})
            for k in _SUBJECT_ARGS:
                if isinstance(kw.get(k), str):
                    kw[k] = self._subject(kw[k])
            done = condition(self._subject(until["subject"]), until["matcher"], **kw)
        try:
            rollout = self._attempt(w.robot.run_policy, pol, until=done, timeout=float(timeout), hold=float(hold))
        except RobowrightError as e:
            return f"FAILED: {type(e).__name__}: {e}\n\n{self.snapshot()}"
        if done is None:  # nothing to judge it by
            head = f"ran {rollout.steps} steps ({rollout.sim_seconds:.2f} s simulated); no goal was given, so no verdict"
        else:
            verdict = "SUCCEEDED" if rollout.success else "did not succeed"
            head = f"{verdict}: {rollout.steps} steps, {rollout.sim_seconds:.2f} s simulated"
        return f"{head}\n\n{self.snapshot()}"

    def set_targets(self, joints: dict[str, float] | None = None, gripper: float | None = None) -> str:
        """Set joint targets (radians, by name) and/or the gripper opening (0 closed .. 1 open)
        without waiting for them: the low-level command a controller sends each control step.
        Time does not pass; follow with ``step``."""
        w = self._w()
        r = w.robot
        names = list(w.backend.joint_names)[: r.n_arm] if hasattr(r, "n_arm") else list(w.backend.joint_names)
        for k, v in (joints or {}).items():
            if k not in names:
                raise RobowrightError(f"unknown joint {k!r}; the joints are: {', '.join(names)}")
            r._target[names.index(k)] = float(v)
        if gripper is not None:
            if not w.backend.has_gripper:
                raise RobowrightError("this robot has no gripper")
            r._target[-1] = float(np.clip(gripper, 0.0, 1.0))
        if w.trace:
            w.trace.event("edit", "set_targets", {"joints": dict(joints or {}), "gripper": gripper})
        return self.snapshot()

    def step(self, steps: int = 1) -> str:
        """Advance ``steps`` control periods with the current targets."""
        return self._act(self._w().step, int(steps))

    # --- the simulator itself (Gazebo, when the ROS 2 backend has gazebo = true) --------
    def _gazebo(self):
        gz = getattr(self._w().backend, "gazebo", None)
        if gz is None:
            raise RobowrightError("no simulator to drive: launch with backend='ros2' and ros2={'gazebo': true} for a robot in Gazebo")
        return gz

    def sim_state(self) -> str:
        gz = self._gazebo()
        return (
            f"Gazebo world {gz.world!r}: t={gz.time:.3f} s, {gz.iterations} iterations, "
            f"{'paused' if gz.paused else 'running'} at {gz.real_time_factor:.2f}x real time\n"
            f"models: {', '.join(gz.models())}"
        )

    def sim_pause(self) -> str:
        self._gazebo().pause()
        return self.sim_state()

    def sim_play(self) -> str:
        self._gazebo().play()
        return self.sim_state()

    def sim_step(self, iterations: int = 1) -> str:
        """Advance the paused simulator exactly ``iterations`` physics steps (pausing it first)."""
        self._gazebo().step(int(iterations))
        return self.sim_state()

    def sim_spawn(self, object: dict) -> str:
        """Put a new object into the simulator (the same fields as launch's ``objects``)."""
        o = self._object(object)
        self._gazebo().spawn_object(o)
        w = self._w()
        w.spec.objects.append(o)
        w.object_names.append(o.name)
        if w.trace:
            w.trace.event("edit", "sim_spawn", {"object": object})
        return self.snapshot()

    def sim_remove(self, name: str) -> str:
        self._gazebo().remove(name)
        w = self._w()
        w.spec.objects = [o for o in w.spec.objects if o.name != name]
        w.object_names = [n for n in w.object_names if n != name]
        if w.trace:
            w.trace.event("edit", "sim_remove", {"name": name})
        return self.snapshot()

    def serve_ros2(
        self,
        namespace: str = "",
        arm_controller: str = "arm_controller",
        gripper_controller: str = "gripper_controller",
    ) -> str:
        """Serve the simulated world as a ROS 2 robot (see ``robowright.ros2_bridge``), so a ROS 2
        stack (yours) can drive it. The world runs while ``wait``/``expect`` step it, in real time."""
        from .ros2_bridge import Ros2Bridge

        w = self._w()
        if self.bridge is None:
            self.bridge = Ros2Bridge(w, namespace=namespace, arm_controller=arm_controller, gripper_controller=gripper_controller)
        a = f"{namespace.rstrip('/')}/{arm_controller}/follow_joint_trajectory"
        return f"serving the world as a ROS 2 robot: /joint_states, /clock, TF for the objects, {a} and the gripper's GripperCommand"

    def start_process(self, command: str, cwd: str | None = None) -> str:
        """Start the code under test (a launch file, a node, a script) as its own process. Its
        output is kept; ``stop_process`` ends it and returns the output's tail."""
        import shlex
        import subprocess

        if self.process is not None:
            self.stop_process()
        log = self.trace_dir / f"process_{self._n}.log"
        proc = subprocess.Popen(shlex.split(command), stdout=log.open("w"), stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True)
        self.process = (proc, log)
        w = self._w()
        if w.trace:
            w.trace.event("edit", "start_process", {"command": command})
        return f"started pid {proc.pid}: {command} (output in {log})"

    def stop_process(self) -> str:
        if self.process is None:
            return "no process is running"
        from .ros2_bridge import _stop

        proc, log = self.process
        self.process = None
        _stop(proc)
        tail = log.read_text()[-3000:] if log.exists() else ""
        return f"stopped (exit {proc.returncode}); its output ended:\n{tail}"

    def watch(self, port: int = 8765) -> str:
        """Serve the world live in a browser: the camera streaming, joints, objects, contacts."""
        from .live import LiveView

        if self.view is None:
            self.view = LiveView(port=port, quiet=True)
        if self.world is not None:
            self.view.watch(self.world)
        return f"watch live at {self.view.url}"

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


def _load_policy(ref: str, kwargs: dict):
    """A policy from ``"package.module:name"``: a callable, a class (built with ``kwargs``), or a
    reference a LearnedPolicy loader understands."""
    import importlib

    mod, _, name = ref.partition(":")
    if not name:
        from .learned import LearnedPolicy

        return LearnedPolicy(ref, **kwargs)
    obj = getattr(importlib.import_module(mod), name)
    return obj(**kwargs) if isinstance(obj, type) else obj
