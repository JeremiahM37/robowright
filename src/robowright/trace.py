"""Trace recording and loading.

A trace is a zip archive holding everything needed to inspect, replay or
regenerate a run:

- ``trace.json``   metadata, scene spec, events (actions, expectations, faults, logs)
- ``steps.npz``    per-control-step arrays: time, qpos, ctrl, object poses, forces
- ``state0.npy``   full simulator state before the first step (when supported)
- ``contacts.json`` contact pairs per step
- ``frames/<camera>/<step>.jpg`` camera frames at ``frame_every`` steps, only if
  ``Settings.trace_cameras`` asked for frames captured during the run; otherwise the
  viewer draws them from the recorded state (robowright.render)
"""

from __future__ import annotations

import io
import json
import os
import sys
import sysconfig
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


BEHAVIOUR_SETTINGS = ("expect_timeout", "action_timeout", "max_joint_speed", "max_tcp_speed")

# Not the caller's own code: robowright itself, installed packages (pytest, an engine) and the standard library.
_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep
_LIBRARY = tuple(
    {os.path.abspath(p) + os.sep for k in ("stdlib", "platstdlib", "purelib", "platlib") if (p := sysconfig.get_paths().get(k))}
)
MAX_SOURCE = 256_000  # bytes of one source file a trace keeps


def _caller() -> tuple[str, int] | None:
    """The file and line of the code that called into robowright: a test, or the user's own helper it called."""
    f = sys._getframe(1)
    while f is not None:
        raw = f.f_code.co_filename
        name = os.path.abspath(raw)
        if not raw.startswith("<") and not name.startswith(_PACKAGE) and not name.startswith(_LIBRARY):  # <frozen runpy>, <string>
            return name, f.f_lineno
        f = f.f_back
    return None


@dataclass
class Event:
    type: str  # action | expect | invariant | fault | log | edit | step
    name: str
    step: int
    t: float
    end_step: int | None = None
    end_t: float | None = None
    status: str = "ok"  # ok | failed | running
    args: dict = field(default_factory=dict)
    detail: str = ""
    depth: int = 0  # how many rw.step()s the event is inside
    loc: list | None = None  # [file, line] of the code that made it, its text in the trace's "sources"

    def to_dict(self):
        d = dict(self.__dict__)
        if not d["depth"]:
            del d["depth"]  # a trace with no steps reads exactly as before
        if d["loc"] is None:
            del d["loc"]
        return d


class Recorder:
    def __init__(self, world, frame_every: int = 5, cameras=None, image_size=(320, 240)):
        self.world = world
        self._perceived = {} if world.has_ground_truth else world.backend.perception()  # recorded each step
        self.frame_every = frame_every
        self.cameras = cameras
        self.image_size = image_size
        self.events: list[Event] = []
        self.sources: dict[str, str] = {}  # file -> its text, for each file an event came from (the viewer's Source tab)
        self.t: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.led: list[bool] = []  # whether the backend led its servos along a smooth move that step
        self.obj_pos: list[np.ndarray] = []
        self.obj_quat: list[np.ndarray] = []
        self.forces: list[np.ndarray] = []
        # External forces are recorded per object, plus the robot's floating base if it has one.
        self.force_names = list(world.object_names) + (["robot"] if world.backend.robot_model.floating else [])
        self.contacts: list[list] = []
        self.frames: dict[str, dict[int, bytes]] = {}
        # ROS 2 topics and log lines seen during the run (ros2_topics.TopicLog.trace_fields): read from ros_source when
        # saved, or kept in ros_fields by the bridge that recorded them, which is closed before the world saves its trace.
        self.ros_fields: dict | None = None
        self.ros_source = None
        self.base: list[np.ndarray] = []
        self.state0 = None
        self.begin_index = 0
        self.started = time.time()

    @property
    def can_render(self):
        from .backends.base import RENDER

        return RENDER in self.world.backend.capabilities and bool(self.cameras)

    def begin(self, state0=None):
        self.begin_index = len(self.events)
        self.state0 = state0
        self._snapshot(0)

    def _snapshot(self, step: int):
        w = self.world
        b = w.backend
        names = w.object_names
        if names and w.has_ground_truth:
            poses = [b.object_pose(n) for n in names]
            self.obj_pos.append(np.array([p for p, _ in poses]))
            self.obj_quat.append(np.array([q for _, q in poses]))
        elif names and self._perceived:
            # Hardware: what the robot's own sensors saw (a ROS 2 robot's TF), so a failure's trace
            # shows the objects where the robot thought they were. Sources a test registers itself
            # (a detector, say) are not called every step; objects without a source stay at zero.
            pos, quat = np.zeros((len(names), 3)), np.zeros((len(names), 4))
            for i, n in enumerate(names):
                if n in self._perceived:
                    try:
                        pos[i], quat[i] = self._perceived[n]()
                    except Exception:  # noqa: BLE001 - not seen this step (a frame not yet published)
                        pass
            self.obj_pos.append(pos)
            self.obj_quat.append(quat)
        else:
            self.obj_pos.append(np.zeros((len(names), 3)))
            self.obj_quat.append(np.zeros((len(names), 4)))
        self.t.append(b.time)
        self.qpos.append(b.qpos())
        if b.robot_model.floating:  # with the joints, enough to redraw the robot (robowright.render)
            pos, quat = b.base_pose()
            self.base.append(np.concatenate([pos, quat]))
        self.ctrl.append(b.ctrl())
        self.led.append(False)
        self.forces.append(np.zeros((len(self.force_names), 3)))
        if w.has_contacts:
            self.contacts.append([[c.a, c.b, round(c.force, 4)] for c in b.contacts()])
        else:
            self.contacts.append([])
        if self.can_render and step % self.frame_every == 0:
            self._frame(step)

    def _frame(self, step: int):
        from PIL import Image

        w, h = self.image_size
        for cam in self.cameras:
            try:
                img = self.world.backend.render(cam, w, h)
            except Exception as e:  # no GL available: keep tracing, just without frames
                import warnings

                warnings.warn(f"robowright: camera frames disabled, rendering failed ({type(e).__name__}: {e})", stacklevel=2)
                self.cameras = []
                return
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=80)
            self.frames.setdefault(cam, {})[step] = buf.getvalue()

    def record_step(self, step: int, applied_ctrl: np.ndarray, forces: dict):
        self._snapshot(step)
        # ctrl stored at index i is what was applied during step i-1 -> i
        self.ctrl[-1] = applied_ctrl.copy()
        self.led[-1] = bool(getattr(self.world.backend, "feedforward", False))
        for n, f in forces.items():
            self.forces[-1][self.force_names.index(n)] = f

    def event(self, type, name, args=None, status="ok", detail="") -> Event:
        w = self.world
        e = Event(type, name, w.step_count, w.time, args=args or {}, status=status, detail=detail)
        e.depth = len(w.__dict__.get("_rw_steps", ()))
        loc = _caller()
        if loc is not None:
            e.loc = [loc[0], loc[1]]
            if loc[0] not in self.sources:
                try:
                    with open(loc[0], encoding="utf-8", errors="replace") as fh:
                        self.sources[loc[0]] = fh.read(MAX_SOURCE)
                except OSError:
                    self.sources[loc[0]] = ""
        if type not in ("action", "step"):  # these two last a while: finish_event closes them
            e.end_step, e.end_t = e.step, e.t
        self.events.append(e)
        return e

    def finish_event(self, e: Event, status="ok", detail=""):
        e.end_step, e.end_t, e.status = self.world.step_count, self.world.time, status
        if detail:
            e.detail = detail

    def meta(self) -> dict:
        from .robots.detect import recorded

        w = self.world
        ros = self.ros_source() if self.ros_source is not None else self.ros_fields
        return {
            **{k: v for k, v in (ros or {}).items() if v},
            "format": FORMAT_VERSION,
            "robowright": _version(),
            "name": w.name,
            "source": w.source,
            "backend": w.backend.name,
            # A robot loaded from a model file, and the options its backend was created with, so the
            # run can be rebuilt by codegen, replay and crosscheck (see Trace.scene).
            "robot_file": recorded(w.backend.robot_model),
            "backend_options": json.loads(json.dumps(getattr(w, "backend_options", {}), default=_json_default)),
            "fidelity": getattr(w, "fidelity", None),
            "model_changes": [c.as_dict() for c in getattr(w, "model_changes", [])],
            "seed": w.seed,
            "scene": w.spec.to_dict(),
            "joint_names": list(w.backend.joint_names),
            "object_names": list(w.object_names),
            "force_names": self.force_names,
            "control_dt": w.backend.control_dt,
            "frame_every": self.frame_every,
            "cameras": sorted(self.frames),
            "started": self.started,
            "wall_seconds": time.time() - self.started,
            "steps": w.step_count,
            "begin_event_index": self.begin_index,
            "status": w.status,
            "faults": [f.describe() for f in w.faults.active],
            # The settings that shape motion and timing, so codegen can rebuild the same run.
            "settings": {k: getattr(w.settings, k) for k in BEHAVIOUR_SETTINGS},
            "events": [e.to_dict() for e in self.events],
            "sources": {k: v for k, v in self.sources.items() if v},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("trace.json", json.dumps(self.meta(), default=_json_default))
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                t=np.array(self.t),
                qpos=np.array(self.qpos),
                ctrl=np.array(self.ctrl),
                obj_pos=np.array(self.obj_pos),
                obj_quat=np.array(self.obj_quat),
                forces=np.array(self.forces),
                **({"base": np.array(self.base)} if self.base else {}),
                **({"led": np.array(self.led)} if any(self.led) else {}),
            )
            z.writestr("steps.npz", buf.getvalue())
            if self.state0 is not None:
                b = io.BytesIO()
                np.save(b, self.state0)
                z.writestr("state0.npy", b.getvalue())
            z.writestr("contacts.json", json.dumps(self.contacts))
            for cam, frames in self.frames.items():
                for step, data in frames.items():
                    z.writestr(f"frames/{cam}/{step:06d}.jpg", data)
        return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def _version():
    from . import __version__

    return __version__


def _call(e: dict) -> str:
    """An event as the call that made it: ``robot.pick(cube)``, ``expect(cube).to_be_inside(bin)``."""

    def val(v):
        if isinstance(v, dict) and "$ref" in v:
            return v["$ref"]
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v) + "]"
        return f"{v:.3g}" if isinstance(v, float) else repr(v)

    args = dict(e.get("args") or {})
    if e["type"] == "step":
        return f"step {e['name']!r}"
    if e["type"] == "expect":
        kw = {k: v for k, v in (args.get("kwargs") or {}).items() if v is not None}
        inner = ", ".join(f"{k}={val(v)}" for k, v in kw.items())
        return f"{e['name'].replace('expect(', 'expect(' + ('not ' if args.get('negate') else ''), 1)}({inner})"
    return f"{e['name']}(" + ", ".join(f"{k}={val(v)}" for k, v in args.items()) + ")"


class Trace:
    """A loaded trace archive."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as z:
            self.meta = json.loads(z.read("trace.json"))
            with z.open("steps.npz") as f:
                arr = np.load(io.BytesIO(f.read()))
                self.arrays = {k: arr[k] for k in arr.files}
            self.state0 = np.load(io.BytesIO(z.read("state0.npy"))) if "state0.npy" in z.namelist() else None
            self.contacts = json.loads(z.read("contacts.json"))
            self.frame_names = sorted(n for n in z.namelist() if n.startswith("frames/"))
            # Attached runs only: every physics step as the environment's loop took it (see attached.AttachedWorld).
            self.actions = None
            if "actions.npz" in z.namelist():
                with z.open("actions.npz") as f:
                    arr = np.load(io.BytesIO(f.read()))
                    self.actions = {k: arr[k] for k in arr.files}

    @property
    def events(self) -> list[dict]:
        return self.meta["events"]

    @property
    def controls(self) -> list[tuple[int, str, np.ndarray | None]]:
        """Each physics step of an attached run in order: ``(step, "action", action)`` for ``env.step(action)`` made
        by the test's loop, or ``(step, "observe", None)`` for a step the loop took below ``env.step``."""
        if not self.actions:
            return []
        rows = iter(self.actions["action"])
        steps, observe = self.actions["steps"], self.actions["observe"]
        return [(int(s), "observe", None) if o else (int(s), "action", next(rows)) for s, o in zip(steps, observe)]

    @property
    def failed(self) -> bool:
        return self.meta.get("status") == "failed"

    def scene(self):
        from .scene import SceneSpec

        self._load_robot()
        return SceneSpec.from_dict(self.meta["scene"])

    def _load_robot(self):
        """Load the robot this trace was recorded with, from its model file, if it is not registered."""
        from .robots import REGISTRY
        from .robots.detect import reload

        rec = self.meta.get("robot_file")
        if not rec or rec["name"] in REGISTRY:
            return
        if not Path(rec["file"]).exists():
            raise FileNotFoundError(
                f"this trace's robot {rec['name']!r} was loaded from {rec['file']}, which no longer exists: "
                "put the model back there, or register the robot under that name, to replay or regenerate the trace"
            )
        reload(rec)

    def frame(self, name: str) -> bytes:
        with zipfile.ZipFile(self.path) as z:
            return z.read(name)

    def __len__(self):
        return len(self.arrays["t"])

    def _code_line(self, loc) -> str:
        """``file:line`` and that line of the code, as the trace kept it."""
        file, line = loc
        text = (self.meta.get("sources") or {}).get(file, "").splitlines()
        code = text[line - 1].strip() if 0 < line <= len(text) else ""
        try:
            file = os.path.relpath(file)
        except ValueError:  # another drive (Windows)
            pass
        return f"{file}:{line}" + (f": {code}" if code else "")

    def summary(self, events: int = 40) -> str:
        """The trace as text, for a log, a terminal or an AI agent: what ran, what failed and why,
        and the state of the robot and the objects at the moment it failed (or at the end)."""
        m, a = self.meta, self.arrays
        failed = [e for e in self.events if e["status"] == "failed"]
        at = min(failed[0]["step"], len(self) - 1) if failed else len(self) - 1
        lines = [
            f"{m.get('name') or self.path.name}: {'FAILED' if self.failed else m.get('status', 'passed').upper()}",
            f"  {(m.get('scene') or {}).get('robot', 'robot')} on {m['backend']}, seed {m['seed']}, {m['steps']} steps"
            f" ({float(a['t'][-1]):.2f} s simulated)",
        ]
        if m.get("fidelity"):
            changes = m.get("model_changes") or []
            kinds = sorted({c["kind"] for c in changes})
            lines.append(
                f"  fidelity: {m['fidelity']}; the model "
                + (
                    f"with {len(changes)} change(s) ({', '.join(kinds)}): " + "; ".join(c["what"] for c in changes)
                    if changes
                    else "as published"
                )
            )
        if m.get("faults"):
            lines.append("  faults: " + "; ".join(_fault_text(f) for f in m["faults"]))
        lines.append("timeline:")
        shown = [e for e in self.events if e["type"] != "edit" or e["name"] != "reset_to"]
        for e in shown[:events]:
            span = f"{e['t']:6.2f}s" if e.get("end_t") in (None, e["t"]) else f"{e['t']:6.2f}-{e['end_t']:.2f}s"
            mark = {"ok": "  ", "failed": "✗ ", "running": "… "}.get(e["status"], "  ")
            pad = "  " * e.get("depth", 0)  # inside rw.step()s: indented under them
            lines.append(f"  {mark}{span}  {pad}{_call(e)}" + (f"\n        {pad}{e['detail']}" if e.get("detail") else ""))
            if e["status"] == "failed" and e.get("loc"):
                lines.append(f"        {pad}at {self._code_line(e['loc'])}")
        if len(shown) > events:
            lines.append(f"  ... {len(shown) - events} more")
        lines.append(f"{'at the failure' if failed else 'at the end'} (t={float(a['t'][at]):.2f} s, step {at}):")
        names = m.get("joint_names", [])
        q = a["qpos"][at]
        lines.append("  joints: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, q)))
        for i, name in enumerate(m.get("object_names", [])):
            p, quat = a["obj_pos"][at][i], a["obj_quat"][at][i]
            yaw = float(np.degrees(2 * np.arctan2(quat[3], quat[0])))
            lines.append(f"  {name}: at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), yaw {yaw:.0f} deg")
        if self.contacts and at < len(self.contacts):
            pairs = {}
            for x, y, f in self.contacts[at]:
                pairs[(x, y)] = pairs.get((x, y), 0.0) + f
            lines.append("  contacts: " + (", ".join(f"{x}-{y} {f:.2f} N" for (x, y), f in sorted(pairs.items())) or "none"))
        if m.get("ros_topics") or m.get("ros_log"):
            from .ros2_topics import log_lines, summary_lines

            lines += summary_lines(m.get("ros_topics") or []) + log_lines(m.get("ros_log") or [], m.get("ros_log_dropped", 0))
        return "\n".join(lines)


def _fault_text(fault) -> str:
    """One fault of a trace's record as text: ``Push(object=cube, force=[0, 1.5, 0], at=3.0, duration=0.1)``."""
    if not isinstance(fault, dict):
        return str(fault)
    return f"{fault.get('type', 'fault')}(" + ", ".join(f"{k}={v}" for k, v in fault.items() if k != "type") + ")"
