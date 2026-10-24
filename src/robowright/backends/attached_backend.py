"""A MuJoCo simulation that robowright did not build.

Environments such as Gymnasium's MuJoCo tasks, dm_control and robosuite own an ``MjModel`` and
``MjData``, their own scene and their own step loop. This backend is a view onto those two
objects: it reads them and sets nothing in them, except the pose a test asks to move and the
forces a test asks to push with, which are undone after the step that applied them. Stepping
is not its business: the environment's loop steps, and :mod:`robowright.attached` observes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np

from ..errors import CapabilityError
from .attached_engine import AttachedEngine
from .base import CONTACTS, FORCES, GROUND_TRUTH, RENDER, STATE, Contact
from .mujoco_backend import renderer

_FREE = mujoco.mjtJoint.mjJNT_FREE
_HINGE = mujoco.mjtJoint.mjJNT_HINGE
_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
_KINDS = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE, "geom": mujoco.mjtObj.mjOBJ_GEOM}


class AttachedBackend(AttachedEngine):
    """A MuJoCo model and data, as the environment holds them. Handles are ``(kind, id)`` pairs."""

    engine = "mujoco"
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, STATE, FORCES})

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, seed: int = 0):
        super().__init__(seed, dt=float(model.opt.timestep))
        self.model, self.data = model, data
        self.joint_names = _qpos_names(model)
        # MuJoCo allows bodies with no name: each gets a stable generated name, "body#<id>", so a world lists
        # as many bodies as the engine has and every one can be asked for by name.
        self._label = {b: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or _unnamed(b)) for b in range(model.nbody)}
        self._label[0] = "world"
        self._free = {}  # body name -> (qpos address, dof address)
        for b in range(1, model.nbody):
            if model.body_jntnum[b] == 1 and model.jnt_type[model.body_jntadr[b]] == _FREE and not _is_unnamed(self._label[b]):
                j = model.body_jntadr[b]
                self._free[self._label[b]] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
        self.free_bodies = list(self._free)

    # --- what the environment's own loop reads and writes --------------------------
    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def qvel(self) -> np.ndarray:
        return self.data.qvel.copy()

    def ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()

    def set_ctrl(self, target) -> None:
        raise CapabilityError("the environment's own loop sets its controls; robowright only observes them")

    def step(self) -> None:
        raise CapabilityError("time advances when the environment's loop steps it; robowright observes those steps")

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def control_dt(self) -> float:
        return self._dt

    def set_control_dt(self, dt: float) -> None:
        """Measured from the first environment step, which is what the environment's control rate is."""
        self._dt = float(dt)

    # --- bodies, sites and geoms by name ------------------------------------------------
    def resolve(self, name: str):
        """``(kind, id)`` of the body, site or geom called ``name``; a body wins over a site over a geom."""
        for kind, objtype in _KINDS.items():
            i = mujoco.mj_name2id(self.model, objtype, name)
            if i >= 0:
                return kind, i
        if _is_unnamed(name):  # a body with no name in the engine, by the generated name names() lists
            i = _unnamed_id(name)
            if i is not None and 0 < i < self.model.nbody:
                return "body", i
        return None

    def names(self, kind: str = "body") -> list[str]:
        if kind == "body":  # every body, the world's included: the same count as the engine's nbody
            return [self._label[b] for b in range(self.model.nbody)]
        n = {"site": self.model.nsite, "geom": self.model.ngeom}[kind]
        out = [mujoco.mj_id2name(self.model, _KINDS[kind], i) for i in range(n)]
        return [x for x in out if x]

    def label(self, handle) -> str:
        kind, i = handle
        return self._label[i] if kind == "body" else (mujoco.mj_id2name(self.model, _KINDS[kind], i) or f"{kind}{i}")

    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        kind, i = handle
        d = self.data
        if kind == "body":
            return d.xpos[i].copy(), d.xquat[i].copy()
        if kind == "site":
            pos, mat = d.site_xpos[i], d.site_xmat[i]
        else:
            pos, mat = d.geom_xpos[i], d.geom_xmat[i]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(mat, float))
        return pos.copy(), quat

    def velocity(self, handle) -> np.ndarray:
        """World-frame linear then angular velocity."""
        kind, i = handle
        res = np.zeros(6)
        # mj_objectVelocity reads the body velocities cached by the last physics step, which lag the
        # joint state by a substep (0.7 mm/s on a 3.8 mm/s push); comVel refreshes them from qpos/qvel.
        mujoco.mj_comVel(self.model, self.data)
        mujoco.mj_objectVelocity(self.model, self.data, _KINDS[kind], i, res, 0)
        return np.concatenate([res[3:], res[:3]])

    def geom_name(self, g: int) -> str | None:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)

    def body_of(self, kind: str, i: int) -> int:
        if kind == "body":
            return i
        return int(self.model.site_bodyid[i] if kind == "site" else self.model.geom_bodyid[i])

    def contacts_of(self, handle) -> list[str]:
        """What this touches, by body name and by geom name: a geom is named as the environment names it."""
        kind, index = handle
        body = self.body_of(kind, index)
        out = set()
        for b1, b2, _, g1, g2 in self.contact_pairs():
            if kind == "geom":
                if index not in (g1, g2):
                    continue
                other_body, other_geom = (b2, g2) if g1 == index else (b1, g1)
            else:
                if body not in (b1, b2):
                    continue
                other_body, other_geom = (b2, g2) if b1 == body else (b1, g1)
            if other_body == body:
                continue
            out.add(self._label[other_body])
            if self.geom_name(other_geom):
                out.add(self.geom_name(other_geom))
        return sorted(out)

    # --- joints -------------------------------------------------------------------------
    def named_joints(self) -> list[str]:
        """Hinge and slide joints by name: the ones ``world.joint`` takes a value from."""
        out = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
            if int(self.model.jnt_type[i]) in (int(_HINGE), int(_SLIDE))
        ]
        return [n for n in out if n]

    def joint_kinds(self) -> dict[str, str]:
        """Every named joint and its kind: ``hinge``, ``slide``, ``ball`` or ``free``. Never raises."""
        out = {}
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n:
                out[n] = _JOINT_KIND.get(int(self.model.jnt_type[i]), "?")
        return out

    def joint_qadr(self, name: str) -> int:
        """The qpos address of a hinge or slide joint, for the core's ``joint()``; the errors name what exists."""
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            named = [n for n in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)) if n]
            raise KeyError(f"no joint named {name!r}; joints: {', '.join(named) or 'none'}")
        # int(): numpy's int32 does not compare equal to MuJoCo's enum members
        if int(self.model.jnt_type[j]) == int(_FREE):
            body = self._label[int(self.model.jnt_bodyid[j])]
            raise CapabilityError(
                f"{name!r} is the free joint of body {body!r}: it has 7 values (position and quaternion), not one angle; "
                f"use world.body({body!r}) for its pose"
            )
        if int(self.model.jnt_type[j]) not in (int(_HINGE), int(_SLIDE)):
            raise CapabilityError(f"joint {name!r} is not a hinge or slide, so it has one scalar value; use world.body() for its pose")
        return int(self.model.jnt_qposadr[j])

    def joint_value(self, name: str) -> float:
        return float(self.data.qpos[self.joint_qadr(name)])

    # --- ground truth for free bodies (what the object locators read) -------------------
    def _body_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError(f"no body named {name!r} in this simulation")
        return i

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        if name not in self._free:
            raise ValueError(f"{name!r} is not a free body, so its pose cannot be set (free bodies: {', '.join(self.free_bodies)})")
        a, dof = self._free[name]
        self.data.qpos[a : a + 3] = pos
        if quat is not None:
            self.data.qpos[a + 3 : a + 7] = quat
        self.data.qvel[dof : dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_joint(self, name: str, value: float) -> None:
        """A hinge or slide joint moved to ``value``, at rest; for a body built from slide joints, which no free joint moves."""
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0 or int(self.model.jnt_type[j]) not in (int(_HINGE), int(_SLIDE)):
            raise ValueError(f"{name!r} is not a hinge or slide joint in this simulation")
        self.data.qpos[self.model.jnt_qposadr[j]] = value
        self.data.qvel[self.model.jnt_dofadr[j]] = 0
        mujoco.mj_forward(self.model, self.data)

    def apply_force(self, name: str, force) -> None:
        b = self._body_id(name)
        self.data.xfrc_applied[b, :3] += force

    # --- contacts -----------------------------------------------------------------------
    def contact_pairs(self) -> list[tuple]:
        """Every contact: ``(body1, body2, normal force, geom1, geom2)``, within one body excluded."""
        m, d = self.model, self.data
        f6 = np.zeros(6)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 == b2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            out.append((b1, b2, float(abs(f6[0])), int(c.geom1), int(c.geom2)))
        return out

    def contacts(self):
        return [Contact(self._label[a], self._label[b], f) for a, b, f, _, _ in self.contact_pairs()]

    # --- rendering, state ------------------------------------------------------------------
    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        # Their model sets the offscreen buffer; a larger picture is drawn at the size that fits, same shape
        fit = min(1.0, self.model.vis.global_.offwidth / width, self.model.vis.global_.offheight / height)
        width, height = max(1, int(width * fit)), max(1, int(height * fit))
        r = self._renderers.get((width, height))
        if r is None:
            r = self._renderers[(width, height)] = renderer(self.model, height, width)
        try:
            r.update_scene(self.data, camera=camera)
        except Exception as e:
            cams = ", ".join(self.names_of_cameras()) or "none"
            raise CapabilityError(f"no camera {camera!r} in this simulation; its cameras: {cams}") from e
        return r.render()

    def names_of_cameras(self) -> list[str]:
        return [n for n in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)) if n]

    def get_state(self) -> np.ndarray:
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        st = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, st, spec)
        return st

    def set_state(self, state) -> None:
        mujoco.mj_setState(self.model, self.data, np.asarray(state, float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.model, self.data)

    def trace_meta(self) -> dict:
        # No "engine" key: a trace without one is a MuJoCo trace, as those were recorded before the adapters.
        return {"model": "model.mjb", "nq": int(self.model.nq)}

    def trace_files(self) -> dict[str, bytes]:
        return {"model.mjb": model_bytes(self.model)}


class PendingBackend(AttachedEngine):
    """A MuJoCo environment that has not built its model yet: safety-gymnasium builds it in ``reset``.

    Until the first reset there is nothing to read, and every read says so. The world binds to the real model
    at that reset (or the first step) and replaces this with an :class:`AttachedBackend`.
    """

    engine = "mujoco"
    capabilities = frozenset()
    model = None  # no model yet: the world's rebind check sees None and binds at the first reset
    data = None

    def __init__(self, seed: int = 0, dt: float | None = None):
        super().__init__(seed, dt=dt)

    def _unbuilt(self):
        raise CapabilityError("this environment builds its simulation in reset(), which has not run yet: call env.reset() first")

    def qpos(self) -> np.ndarray:
        return np.zeros(0)

    def qvel(self) -> np.ndarray:
        return np.zeros(0)

    def ctrl(self) -> np.ndarray:
        return np.zeros(0)

    def set_ctrl(self, target) -> None:
        self._unbuilt()

    def step(self) -> None:
        self._unbuilt()

    @property
    def time(self) -> float:
        return 0.0

    @property
    def control_dt(self) -> float:
        return self._dt

    def set_control_dt(self, dt: float) -> None:
        self._dt = float(dt)

    def names(self, kind: str = "body") -> list[str]:
        return []

    def resolve(self, name: str):
        return None

    def label(self, handle) -> str:
        self._unbuilt()

    def pose(self, handle):
        self._unbuilt()

    def velocity(self, handle):
        self._unbuilt()

    def contacts_of(self, handle) -> list[str]:
        self._unbuilt()

    def contact_pairs(self) -> list[tuple]:
        return []

    def joint_kinds(self) -> dict[str, str]:
        return {}

    def joint_qadr(self, name: str) -> int:
        self._unbuilt()

    def joint_value(self, name: str) -> float:
        self._unbuilt()

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        self._unbuilt()

    def set_joint(self, name: str, value: float) -> None:
        self._unbuilt()

    def apply_force(self, name: str, force) -> None:
        self._unbuilt()

    def trace_meta(self) -> dict:
        return {"model": "model.mjb", "nq": 0}

    def trace_files(self) -> dict[str, bytes]:
        return {}


def model_bytes(model: mujoco.MjModel) -> bytes:
    """The model as a self-contained MuJoCo binary (``.mjb``), which the trace keeps for redrawing."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.mjb"
        mujoco.mj_saveModel(model, str(p), None)
        return p.read_bytes()


_JOINT_KIND = {int(_FREE): "free", int(_HINGE): "hinge", int(_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_BALL): "ball"}


def _unnamed(body_id: int) -> str:
    return f"body#{body_id}"


def _is_unnamed(name: str) -> bool:
    return name.startswith("body#")


def _unnamed_id(name: str) -> int | None:
    digits = name[len("body#") :]
    return int(digits) if digits.isdigit() else None


def _qpos_names(model: mujoco.MjModel) -> list[str]:
    """One name per qpos entry: the joint's name (or ``joint<i>``), with a suffix for each component of a free or ball joint."""
    names = []
    for j in range(model.njnt):
        base = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        t = model.jnt_type[j]
        if t == _FREE:
            names += [f"{base}.{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            names += [f"{base}.{k}" for k in ("qw", "qx", "qy", "qz")]
        else:
            names.append(base)
    return names
