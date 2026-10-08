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
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


BEHAVIOUR_SETTINGS = ("expect_timeout", "action_timeout", "max_joint_speed", "max_tcp_speed")


@dataclass
class Event:
    type: str  # action | expect | invariant | fault | log | edit
    name: str
    step: int
    t: float
    end_step: int | None = None
    end_t: float | None = None
    status: str = "ok"  # ok | failed | running
    args: dict = field(default_factory=dict)
    detail: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class Recorder:
    def __init__(self, world, frame_every: int = 5, cameras=None, image_size=(320, 240)):
        self.world = world
        self._perceived = {} if world.has_ground_truth else world.backend.perception()  # recorded each step
        self.frame_every = frame_every
        self.cameras = cameras
        self.image_size = image_size
        self.events: list[Event] = []
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
        if type not in ("action",):
            e.end_step, e.end_t = e.step, e.t
        self.events.append(e)
        return e

    def finish_event(self, e: Event, status="ok", detail=""):
        e.end_step, e.end_t, e.status = self.world.step_count, self.world.time, status
        if detail:
            e.detail = detail

    def meta(self) -> dict:
        w = self.world
        return {
            "format": FORMAT_VERSION,
            "robowright": _version(),
            "name": w.name,
            "backend": w.backend.name,
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

    @property
    def events(self) -> list[dict]:
        return self.meta["events"]

    @property
    def failed(self) -> bool:
        return self.meta.get("status") == "failed"

    def scene(self):
        from .scene import SceneSpec

        return SceneSpec.from_dict(self.meta["scene"])

    def frame(self, name: str) -> bytes:
        with zipfile.ZipFile(self.path) as z:
            return z.read(name)

    def __len__(self):
        return len(self.arrays["t"])

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
        if m.get("faults"):
            lines.append("  faults: " + "; ".join(m["faults"]))
        lines.append("timeline:")
        shown = [e for e in self.events if e["type"] != "edit" or e["name"] != "reset_to"]
        for e in shown[:events]:
            span = f"{e['t']:6.2f}s" if e.get("end_t") in (None, e["t"]) else f"{e['t']:6.2f}-{e['end_t']:.2f}s"
            mark = {"ok": "  ", "failed": "✗ ", "running": "… "}.get(e["status"], "  ")
            lines.append(f"  {mark}{span}  {_call(e)}" + (f"\n        {e['detail']}" if e.get("detail") else ""))
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
        return "\n".join(lines)
