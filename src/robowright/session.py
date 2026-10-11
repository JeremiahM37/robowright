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

import difflib
import functools
import inspect
import io
import math
import re
import tempfile
from pathlib import Path

import numpy as np

from . import robots
from .errors import CapabilityError, RobowrightError
from .scene import CameraSpec, ObjectSpec, SceneSpec, default_scene
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


_LIST_CAP = 60  # names per line in sim_subjects: a humanoid's bodies would otherwise fill the reply


def _built_only(reason: str):
    """A call that needs a world robowright built: on an attached simulation it answers why, not a traceback."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(self, *args, **kwargs):
            if self.env is not None:
                raise RobowrightError(f"{fn.__name__} is not available on an attached simulation because {reason}")
            return fn(self, *args, **kwargs)

        return inner

    return wrap


def _close_env(env) -> None:
    """Close an environment robowright made, if it can be closed (a bare reset-and-step env may not have close)."""
    close = getattr(env, "close", None)
    if callable(close):
        close()


# Environment packages whose ids gymnasium only knows once their module is handed to gymnasium or imported.
# How each one registers follows its own examples in examples/environments/. Only gymnasium-robotics is
# verified against the real package; the others register on import and are not installed here.
_ENV_PACKAGES = (
    ("gymnasium_robotics", "register"),  # gymnasium.register_envs(module), as its examples do
    ("mani_skill", "import"),  # ManiSkill registers its ids when mani_skill is imported
    ("panda_gym", "import"),
    ("pybullet_envs_gymnasium", "import"),
    ("gym_aloha", "import"),
)
_suites: tuple[list[str], dict[str, str]] | None = None  # per process: packages registered, and imports that failed


def _register_installed_suites(gymnasium) -> tuple[list[str], dict[str, str]]:
    """Import every installed environment package once, so its ids are registered. A package that fails to import
    is reported and does not stop the others."""
    global _suites
    if _suites is None:
        import importlib
        import importlib.util

        loaded, failed = [], {}
        for name, how in _ENV_PACKAGES:
            if importlib.util.find_spec(name) is None:
                continue
            try:
                module = importlib.import_module(name)
                if how == "register":
                    gymnasium.register_envs(module)
                loaded.append(name)
            except Exception as e:
                failed[name] = f"{type(e).__name__}: {e}"
        _suites = (loaded, failed)
    return _suites


def _registered_like(query: str) -> str:
    """The registered gymnasium ids that look like ``query``, so an agent can pick one that exists."""
    try:
        import gymnasium
    except ImportError:
        return ""
    ids = sorted(gymnasium.registry)
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", query) if len(t) >= 3]
    hits = [i for i in ids if any(t in i.lower() for t in tokens)]
    near = difflib.get_close_matches(query, ids, n=5, cutoff=0.5)
    found = list(dict.fromkeys(near + hits))[:10]
    if not found:
        return f"{len(ids)} environment ids are registered; none looks like it"
    return "registered ids that look like it: " + ", ".join(found)


def _suite_note(loaded: list[str], failed: dict[str, str]) -> str:
    """What was registered from installed packages, and which imports failed."""
    text = "packages registered: " + (", ".join(loaded) or "none (no known environment package is installed)")
    if failed:
        text += "; failed to import: " + "; ".join(f"{k} ({v})" for k, v in failed.items())
    return text


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
        self.env = None  # the environment someone else built, once sim_attach() has attached to it
        self.env_label = ""

    # --- lifecycle -------------------------------------------------------------
    def launch(
        self,
        robot: str | None = None,
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
        ``robowright.backends.ros2_backend``; ``robot=None`` reads the description the stack publishes);
        it gets no objects unless ``objects`` names some, and the robot is not moved.
        ``robot_options``: for a robot given as a model
        file, how to read it (``{"gripper": false, "base_pos": [0, 0, 0]}``: see ``robots.load``)."""
        if self.world is not None:
            self.close()
        if backend != "ros2":
            robot = robot or "so101"
        if backend == "ros2":
            from .world import _ros_robot

        opts = {k: tuple(v) if isinstance(v, list) else v for k, v in (robot_options or {}).items()}
        if backend == "ros2" and robot is None:
            robot, opts = _ros_robot(None, ros2 or {}, opts), {}  # the stack's own description, read with them
        elif robot_options and not robots.is_file(robot):
            raise RobowrightError(f"robot_options are for a robot given as a model file; {robot!r} is a built-in robot")
        model = robots.load(robot, **opts) if opts else robots.get(robot)
        # A mobile robot's scene is its own: no table, cube or bin to place (its world is the stack's).
        scene = SceneSpec(robot=model.name, objects=[]) if model.family == "mobile" else default_scene(model.name)
        if objects is not None:
            scene.objects = [self._object(o) for o in objects]
        elif backend == "ros2":  # a robot behind ROS 2 is someone else's world: no objects unless given
            scene.objects = []
        scene.cameras = list(scene.cameras) + _views(scene.cameras[0], scene.objects)
        self._n += 1
        # No camera frames in the trace: screenshots are taken on request, and rendering
        # every few steps would cost more than the physics.
        settings = Settings(trace="on", trace_dir=str(self.trace_dir), trace_cameras=[], fidelity=fidelity)
        if backend == "ros2" and not (ros2 or {}).get("gazebo"):
            scene.cameras = scene.cameras[:1]  # a ROS 2 robot has the cameras its settings name (in Gazebo, any)
        self.world = launch(scene, backend=backend, seed=seed, name=f"session_{self._n}", settings=settings, **(ros2 or {}))
        self.world.source = "session"
        if backend != "ros2":  # a robot behind ROS 2 is already where its own stack has it: never move it
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

    def ros2_inspect(self, namespace: str = "", timeout: float = 5.0) -> str:
        """What is running behind ROS 2, and the settings to connect to it with (as JSON for ``launch``)."""
        import json

        from . import ros2_discover as discover

        graph = discover.inspect(namespace, timeout=timeout)
        found = discover.suggest(graph)
        return f"{discover.report(graph)}\n\nros2 settings for launch: {json.dumps(found)}"

    def close(self) -> str:
        """Close the world and keep its trace; returns where it was saved."""
        w = self._w()
        self.stop_process()
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        self.world = None
        env, self.env = self.env, None
        self.last_trace = w.close(failed=w.status == "failed")
        _close_env(env)
        return f"closed; trace saved to {self.last_trace}"

    def _w(self) -> World:
        if self.world is None:
            raise RobowrightError("no world is open: call launch first")
        return self.world

    def _attached(self) -> World:
        if self.env is None:
            raise RobowrightError("no attached simulation: call sim_attach first")
        return self._w()

    def sim_attach(self, env_id: str = "", kwargs: dict | None = None, seed: int | None = 0, entry_point: str | None = None) -> str:
        """Attach to a simulation someone else built (see ``robowright.attach``): a gymnasium environment by
        ``env_id``, or a ``module:callable`` that returns one (called with ``kwargs``). It is reset with ``seed``
        and then watched. Their loop is not robowright's: time moves when ``sim_env_step`` steps it."""
        from .attached import attach

        if self.world is not None:
            self.close()
        env = self._make_env(env_id, dict(kwargs or {}), entry_point)
        label = type(getattr(env, "unwrapped", env)).__name__
        try:
            self._n += 1
            settings = Settings(trace="on", trace_dir=str(self.trace_dir), trace_cameras=[])
            name = f"session_{self._n}"
            try:
                world = attach(env, name=name, seed=seed or 0, settings=settings)
            except ValueError:  # an environment that builds its simulation on its first reset
                env.reset()
                world = attach(env, name=name, seed=seed or 0, settings=settings)
            # reset after attaching, so the trace has the reset and a generated test repeats it
            env.reset() if seed is None else env.reset(seed=seed)
        except Exception as e:
            _close_env(env)
            hint = (
                " robowright reads MuJoCo (gymnasium, dm_control, robosuite), SAPIEN (ManiSkill 3) and PyBullet (panda-gym) environments."
                if isinstance(e, ValueError)
                else ""
            )
            raise RobowrightError(f"cannot attach to {label}: {type(e).__name__}: {e}.{hint}") from e
        world.source = "session"
        spec_id = getattr(getattr(env, "spec", None), "id", None)  # the id gymnasium resolved, e.g. FetchPickAndPlace -> -v4
        self.world, self.env = world, env
        self.env_label = f"{spec_id}, {label}" if spec_id else label
        return self.snapshot()

    @staticmethod
    def _make_env(env_id: str, kwargs: dict, entry_point: str | None):
        import importlib

        if entry_point:
            mod, _, name = entry_point.partition(":")
            if not mod or not name:
                raise RobowrightError(f"entry_point is 'module:callable', e.g. 'my_pkg.envs:make_arm'; got {entry_point!r}")
            try:
                factory = getattr(importlib.import_module(mod), name)
            except (ImportError, AttributeError) as e:
                raise RobowrightError(f"cannot import entry_point {entry_point!r}: {type(e).__name__}: {e}") from e
            try:
                return factory(**kwargs)
            except Exception as e:
                raise RobowrightError(f"{entry_point} raised {type(e).__name__}: {e}") from e
        if not env_id:
            raise RobowrightError("give env_id (a gymnasium id) or entry_point ('module:callable')")
        try:
            import gymnasium
        except ImportError as e:
            raise RobowrightError(
                f"cannot make {env_id!r}: gymnasium is not installed here. Install it and the environment's package, "
                "or pass entry_point='module:callable'"
            ) from e
        try:
            return gymnasium.make(env_id, **kwargs)  # also "module:EnvId": gymnasium imports the module first
        except Exception:
            # Not registered yet: the installed suites register their ids, then the lookup is retried.
            loaded, failed = _register_installed_suites(gymnasium)
            try:
                return gymnasium.make(env_id, **kwargs)
            except Exception as e:
                reason = f"{type(e).__name__}: {str(e).rstrip('.')}"
                # gymnasium's own errors are about the id; anything else was raised while building a registered one
                lookup = getattr(getattr(gymnasium, "error", None), "Error", None)
                found = lookup is not None and not isinstance(e, lookup)
                if found:
                    # The id was found and the environment's own code raised: where, since an assert says nothing
                    raise RobowrightError(f"{env_id!r} is registered, but building it failed in its own code: {_where(e)}") from e
                raise RobowrightError(f"cannot make {env_id!r}: {reason}. {_registered_like(env_id)}. {_suite_note(loaded, failed)}") from e

    # --- observation -----------------------------------------------------------
    def snapshot(self) -> str:
        """The world as text: robot state, every object (by ref), contacts and faults."""
        if self.env is not None:
            return self._attached_snapshot()
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
        elif m.family == "mobile":
            base = r.base
            lines += [
                "robot (mobile):",
                f"  base: position {_f(base.position)}, yaw {math.degrees(base.yaw):.1f} deg",
                f"  base velocity: {_f(base.velocity[:3])} m/s",
            ]
            try:  # the localizer's belief: a robot with no [ros2] base frames has none to show
                est = base.estimate
                lines.append(f"  belief (estimate): position {_f(est.position)}, yaw {math.degrees(est.yaw):.1f} deg")
            except Exception as e:
                lines.append(f"  belief (estimate): unavailable ({e})")
            if r.joints:
                lines.append(f"  wheels: {', '.join(f'{k} {v:.3f}' for k, v in r.joints.items())}")
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

    def _attached_snapshot(self) -> str:
        """An attached simulation as text: its time, and every free body by name with its pose and contacts."""
        w = self._w()
        lines = [
            f"world: attached {self.env_label} ({w.backend.engine}), t={w.time:.3f}s, seed={w.seed}, status={w.status}",
            f"steps taken: {w.step_count} (time moves only when sim_env_step steps the environment)",
            "objects (free bodies, refs for other calls):",
        ]
        for name in w.object_names:
            h = w.scene[name]
            s = f"  - {name}: at {_f(h.position)}, yaw {math.degrees(h.yaw):.0f} deg"
            v = float(np.linalg.norm(h.velocity[:3]))
            s += ", at rest" if v < 0.01 else f", moving {v:.2f} m/s"
            if w.has_contacts:
                s += f", touching {', '.join(sorted(h.contacts())) or 'nothing'}"
            lines.append(s)
        if not w.object_names:
            lines.append("  (none)")
        try:  # sites are where a simulation marks its gripper and its targets: named here with where they are
            sites = list(w.backend.names("site"))
        except (NotImplementedError, KeyError, ValueError, AttributeError):
            sites = []
        if sites:
            lines.append("sites (points the simulation names, refs too):")
            lines += [f"  - {n}: at {_f(w.body(n).position)}" for n in sites[:12]]
            if len(sites) > 12:
                lines.append(f"  (+{len(sites) - 12} more: sim_subjects)")
        lines.append("other named bodies and joints: sim_subjects")
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
        if self.env is None:
            w.require("render", "screenshots")
            names = [c.name for c in w.spec.cameras]
            if camera not in names:
                raise RobowrightError(f"unknown camera {camera!r}; this world has: {', '.join(names)}")
            frame = w.backend.render(camera, int(width), int(height))
        else:  # an attached simulation's cameras are its own: the engine names them if the one asked for is missing
            try:
                frame = w.backend.render(camera, int(width), int(height))
            except NotImplementedError as e:
                raise RobowrightError(
                    f"robot_screenshot is not available on this attached simulation: its engine ({w.backend.engine}) renders no cameras"
                ) from e
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
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

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def pick(self, object: str, approach: str | list = "top") -> str:
        approach = approach if isinstance(approach, str) else tuple(float(v) for v in approach)
        return self._act(self._arm().pick, self._w().scene[object], approach=approach)

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def place(self, on, height: float | None = None) -> str:
        return self._act(self._arm().place, self._target(on), height=height)

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def move_to(self, target, linear: bool = False, speed: float = 0.15) -> str:
        return self._act(self._arm().arm.move_to, self._target(target), linear=linear, speed=speed)

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def gripper(self, action: str, amount: float = 1.0) -> str:
        g = self._arm().gripper
        if action == "open":
            return self._act(g.open, amount)
        if action == "close":
            return self._act(g.close)
        raise RobowrightError(f"gripper action is 'open' or 'close', not {action!r}")

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def home(self) -> str:
        return self._act(self._arm().arm.home)

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
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

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def stand(self) -> str:
        return self._act(self._legged().stand)

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def crouch(self, depth: float = 0.5) -> str:
        return self._act(self._legged().crouch, depth)

    def _mobile(self):
        w = self._w()
        if w.backend.robot_model.family != "mobile":
            raise RobowrightError(f"{w.backend.robot_model.title} is not a mobile robot: use pick, place, move_to or gripper")
        return w.robot.base

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def drive(self, linear: float, angular: float, duration: float) -> str:
        """Mobile: drive the base at ``linear`` m/s and ``angular`` rad/s for ``duration`` seconds, then stop."""
        return self._act(self._mobile().drive, linear, angular, duration)

    @_built_only("robowright drives only the robots it built, and the environment's own loop moves its robot")
    def go_to(self, x: float, y: float, yaw: float | None = None) -> str:
        """Mobile: send the base to (x, y) in the map frame, with heading ``yaw``, and wait for its navigation stack."""
        return self._act(self._mobile().go_to, (x, y), yaw=yaw)

    def _legged(self):
        w = self._w()
        if w.backend.robot_model.family != "legged":
            raise RobowrightError(f"{w.backend.robot_model.title} is an arm: use pick, place, move_to or gripper")
        return w.robot

    def wait(self, seconds: float) -> str:
        if self.env is not None:
            raise RobowrightError(
                "wait is not available on an attached simulation: time passes only when the environment's own step runs; "
                "use sim_env_step(steps=N)"
            )
        return self._act(self._w().wait, float(seconds))

    def push(self, target: str, force, duration: float = 0.1) -> str:
        """A force (newtons, world frame) on an object or ``robot`` (legged base), for ``duration`` seconds."""
        w = self._w()
        if self.env is not None:  # the force acts on the steps sim_env_step takes while it lasts
            if target not in w.object_names:
                raise RobowrightError(f"push moves free bodies; {target!r} is not one (free bodies: {', '.join(w.object_names) or 'none'})")
            w.faults.push(target, force=tuple(float(f) for f in force), duration=float(duration))
            when = "the steps sim_env_step takes in that time"
            return f"queued a push of {list(force)} N on {target} for {duration} s; it acts on {when}\n\n{self.snapshot()}"
        if target != "robot":
            w.scene[target]  # unknown names fail here, with the list of objects
        w.faults.push(target, force=tuple(float(f) for f in force), duration=float(duration))
        return self._act(w.wait, float(duration))

    def move_object(self, object: str, position, yaw: float | None = None) -> str:
        w = self._w()
        w.scene[object]
        quat = None if yaw is None else (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

        def move():
            try:
                w.move_object(object, tuple(float(v) for v in position), quat)
            except ValueError as e:  # an engine refuses to move a body that no free joint moves
                raise RobowrightError(str(e)) from e

        return self._act(move)

    @_built_only("its faults act on the robot robowright drives")
    def fault(self, kind: str, **params) -> str:
        """Inject a fault: joint_noise, action_delay, weak_joint, jitter or camera_dropout."""
        w = self._w()
        allowed = ("joint_noise", "action_delay", "weak_joint", "jitter", "camera_dropout")
        if kind not in allowed:
            raise RobowrightError(f"unknown fault {kind!r}; choose from {', '.join(allowed)}")
        getattr(w.faults, kind)(**params)
        return self.snapshot()

    # --- the code under test ------------------------------------------------------
    @_built_only("robowright runs policies on the robot it built, and an attached environment brings its own robot")
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

    @_built_only("robowright sends targets only to robots it built, and the environment's own loop sends its own")
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

    @_built_only("robowright sends targets only to robots it built, and the environment's own loop sends its own")
    def step(self, steps: int = 1) -> str:
        """Advance ``steps`` control periods with the current targets."""
        return self._act(self._w().step, int(steps))

    def sim_env_step(self, action: list | None = None, steps: int = 1, sample: bool = False) -> str:
        """Step the attached environment's own ``step`` ``steps`` times with ``action`` (None: the zero, no-op
        action; ``sample``: a random one from its action space). Each step goes through the environment's
        ``step``, so robowright records and checks it. Returns the reward summed over the steps, whether the
        episode ended, and which named bodies moved (more than 0.1 mm)."""
        self._attached()
        env = self.env
        space = getattr(env, "action_space", None)
        if space is None:
            raise RobowrightError("this environment has no action_space, so robowright cannot choose its action")
        if sample and action is not None:
            raise RobowrightError("pass action or sample=true, not both")
        if action is not None:
            a = np.asarray(action, dtype=space.dtype)
            if a.shape != tuple(space.shape):
                raise RobowrightError(f"action has shape {a.shape}; this environment's action space is {tuple(space.shape)}")
        elif sample:
            a = space.sample()
        else:
            a = np.zeros(space.shape, dtype=space.dtype)
        steps = int(steps)
        if steps < 1:
            raise RobowrightError("steps must be at least 1")
        before = self._body_positions()
        reward, taken, ended = 0.0, 0, ""
        for _ in range(steps):
            try:
                out = env.step(a)
            except RobowrightError as e:
                return f"FAILED after {taken} of {steps} steps: {type(e).__name__}: {e}\n\n{self.snapshot()}"
            reward += float(out[1])
            taken += 1
            term, trunc = out[2], (out[3] if len(out) == 5 else False)
            if term or trunc:
                ended = "terminated" if term else "truncated"
                break
        lines = [f"stepped {taken} of {steps}: reward {reward:.4g}, " + (f"episode {ended}" if ended else "episode continues")]
        lines.append("moved: " + _listing(self._moved(before), ()))
        if ended:
            lines.append("the episode has ended: call sim_attach again to reset it before stepping further")
        return "\n".join(lines) + "\n\n" + self.snapshot()

    def sim_subjects(self) -> str:
        """The names an attached simulation's tools take: bodies (a free body is an object, starred), sites,
        joints and cameras. An engine that does not name a kind leaves its line empty."""
        w = self._attached()
        b = w.backend

        def named(kind: str) -> list[str]:
            try:
                return list(b.names(kind))
            except (NotImplementedError, KeyError, ValueError):
                return []

        cameras = list(b.names_of_cameras()) if hasattr(b, "names_of_cameras") else []
        return "\n".join(
            [
                f"bodies (* = free body, an object): {_listing(w.bodies(), set(w.object_names))}",
                f"sites: {_listing(named('site'))}",
                f"joints (hinge and slide): {_listing(list(b.named_joints()))}",
                f"cameras (robot_screenshot camera=): {_listing(cameras)}",
            ]
        )

    def _body_positions(self) -> dict:
        w = self._w()
        return {n: w.body(n).position.copy() for n in w.bodies()}

    def _moved(self, before: dict) -> list[str]:
        after = self._body_positions()
        return [n for n, p in after.items() if float(np.linalg.norm(p - before[n])) > 1e-4]

    # --- the simulator itself (Gazebo, when the ROS 2 backend has gazebo = true) --------
    def _gazebo(self):
        gz = getattr(self._w().backend, "gazebo", None)
        if gz is None:
            raise RobowrightError("no simulator to drive: launch with backend='ros2' and ros2={'gazebo': true} for a robot in Gazebo")
        return gz

    def sim_state(self) -> str:
        gz = self._gazebo()
        w = self._w()
        return (
            f"Gazebo world {gz.world!r}: its clock t={gz.time:.3f} s, {gz.iterations} iterations, "
            f"{'paused' if gz.paused else 'running'} at {gz.real_time_factor:.2f}x real time\n"
            f"robowright's world time (the t in snapshots) is {w.time:.3f} s, counted from launch; "
            "Gazebo's clock is its own simulation time and need not match it\n"
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

    @_built_only("the ROS 2 bridge serves a robot robowright built")
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

    @_built_only("the live view draws a robowright robot's joints and cameras, which an attached environment does not have")
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
        attached_matchers = ("to_be_between",) if self.env is not None else ()  # a joint's value
        if matcher not in _MATCHERS + attached_matchers:
            raise RobowrightError(f"unknown matcher {matcher!r}; choose from {', '.join(_MATCHERS + attached_matchers)}")
        subj = self._subject(subject)
        kw = dict(args or {})
        for k in _SUBJECT_ARGS:
            if isinstance(kw.get(k), str):
                kw[k] = self._subject(kw[k])
        e = expect(subj, timeout=timeout, world=w)
        if negate:
            e = e.not_
        # With no advance callable nothing steps the world: the state is checked once, and a hold (time the
        # state must stay true) cannot pass, so it is not waited for.
        once = getattr(w, "_advance", 1) is None
        note = " (checked once: this simulation has no advance callable, so time did not pass and no hold was waited for)" if once else ""
        if once and "hold" in inspect.signature(getattr(e, matcher)).parameters:
            kw.setdefault("hold", 0.0)
        try:
            self._attempt(getattr(e, matcher), **kw)
        except ExpectationError as err:
            return f"FAIL: {err}{note}"
        except CapabilityError as err:  # a hold where time cannot pass: not checked, and why
            return f"FAIL: expect({subject}).{matcher} was not checked: {err}"
        shown = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
        return f"PASS: expect({subject}){'.not_' if negate else ''}.{matcher}({shown}){note}"

    def _subject(self, name: str):
        if self.env is not None:  # an attached simulation: its named bodies and joints are the only subjects
            w = self._w()
            if name in w.backend.named_joints():
                return w.joint(name)
            try:
                return w.body(name)
            except KeyError as e:
                raise RobowrightError(str(e.args[0])) from e
        r = self._w().robot
        if name in ("robot", "base") and self.world.backend.robot_model.family in ("legged", "mobile"):
            return r.base  # a legged or mobile robot's pose, velocity and contacts are its base's
        special = {"robot": r, "gripper": getattr(r, "gripper", None), "tcp": getattr(r, "tcp", None)}
        if name == "base":
            return r.base
        if special.get(name) is not None:
            return special[name]
        return self.world.scene[name]

    # --- output ------------------------------------------------------------------
    def generate_test(self, test_name: str = "test_session", path: str | None = None, keep_failed: bool = False) -> str:
        """The session so far as a pytest test that reproduces it; written to ``path`` if given.

        A step that raised is left out, as a comment (the first line of the result says how many), unless
        ``keep_failed``: then it is asserted to fail again. A failed step can still have moved the world, so
        without it the test may not reproduce the session exactly."""
        from .codegen import companion, generate

        w = self._w()
        tmp = self.trace_dir / f"{w.name}_codegen.zip"
        w.trace.save(tmp)
        code = generate(tmp, test_name=test_name, keep_failed=keep_failed)
        if path:
            Path(path).write_text(code)
            # an attached run's long action list is a file beside the test, which loads it by name
            if self.env is not None and (beside := companion(tmp, test_name)) is not None:
                Path(path).with_name(beside[0]).write_bytes(beside[1])
        return code

    @_built_only("it replays the session's calls on another engine, which needs a world robowright built")
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


def _where(e: BaseException) -> str:
    """An exception with the line that raised it: ``AssertionError at pkg/file.py:142 in fn: `assert ...```."""
    import traceback

    last = traceback.extract_tb(e.__traceback__)[-1]
    file = "/".join(Path(last.filename).parts[-3:])
    text = f": {e}" if str(e) else ""
    return f"{type(e).__name__}{text} at {file}:{last.lineno} in {last.name}: `{last.line}`"


def _listing(names, mark=()) -> str:
    """Names for one line: those in ``mark`` starred, and the rest past ``_LIST_CAP`` counted, not listed."""
    shown = [n + ("*" if n in mark else "") for n in names[:_LIST_CAP]]
    more = len(names) - _LIST_CAP
    return (", ".join(shown) or "none") + (f" (+{more} more)" if more > 0 else "")


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
