"""A MuJoCo simulation that robowright did not build.

Environments such as Gymnasium's MuJoCo tasks, dm_control and robosuite own an ``MjModel`` and
``MjData``, their own scene and their own step loop. This backend is a view onto those two
objects: it reads them and sets nothing in them, except the pose a test asks to move and the
forces a test asks to push with, which are undone after the step that applied them. Stepping
is not its business: the environment's loop steps, and :mod:`robowright.attached` observes.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..errors import CapabilityError
from ..scene import SceneSpec
from .base import CONTACTS, FORCES, GROUND_TRUTH, RENDER, STATE, Backend, Contact

_FREE = mujoco.mjtJoint.mjJNT_FREE
_HINGE = mujoco.mjtJoint.mjJNT_HINGE
_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
_KINDS = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE, "geom": mujoco.mjtObj.mjOBJ_GEOM}


class _UnknownRobot:
    """The fields of a robot model the core reads. The robot here belongs to the environment,
    so robowright describes none of it: no arm, no gripper, no floating base."""

    floating = False
    arm_joints: tuple = ()
    title = "attached"
    family = "attached"
    n_arm = 0
    has_gripper = False


class AttachedBackend(Backend):
    name = "attached"
    builds = False  # the environment built it: robowright changes nothing in it
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, STATE, FORCES})

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, seed: int = 0):
        # Backend.__init__ reads a robot model from a SceneSpec; there is none, so set what it sets.
        self.model, self.data = model, data
        self.seed = seed
        self.spec = SceneSpec(robot="attached", objects=[], cameras=[])
        self.robot_model = _UnknownRobot()
        self.n_arm, self.has_gripper = 0, False
        self.joint_names = _qpos_names(model)
        self._dt = float(model.opt.timestep)
        self._renderers: dict = {}
        self._label = {b: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}") for b in range(model.nbody)}
        self._label[0] = "world"
        self._free = {}  # body name -> (qpos address, dof address)
        for b in range(1, model.nbody):
            if model.body_jntnum[b] == 1 and model.jnt_type[model.body_jntadr[b]] == _FREE and self._label[b] != f"body{b}":
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
        return None

    def names(self, kind: str = "body") -> list[str]:
        n = {"body": self.model.nbody, "site": self.model.nsite, "geom": self.model.ngeom}[kind]
        out = [mujoco.mj_id2name(self.model, _KINDS[kind], i) for i in range(n)]
        return [x for x in out if x]

    def pose(self, kind: str, i: int) -> tuple[np.ndarray, np.ndarray]:
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

    def velocity(self, kind: str, i: int) -> np.ndarray:
        """World-frame linear then angular velocity."""
        res = np.zeros(6)
        # mj_objectVelocity reads the body velocities cached by the last physics step, which lag the
        # joint state by a substep (0.7 mm/s on a 3.8 mm/s push); comVel refreshes them from qpos/qvel.
        mujoco.mj_comVel(self.model, self.data)
        mujoco.mj_objectVelocity(self.model, self.data, _KINDS[kind], i, res, 0)
        return np.concatenate([res[3:], res[:3]])

    def label(self, body: int) -> str:
        return self._label[body]

    def geom_name(self, g: int) -> str | None:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)

    def body_of(self, kind: str, i: int) -> int:
        if kind == "body":
            return i
        return int(self.model.site_bodyid[i] if kind == "site" else self.model.geom_bodyid[i])

    # --- ground truth for free bodies (what the object locators read) -------------------
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.pose("body", self._body_id(name))

    def object_velocity(self, name: str) -> np.ndarray:
        return self.velocity("body", self._body_id(name))

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
    def contacts(self) -> list[Contact]:
        out = []
        for a, b, force, _, _ in self.contact_pairs():
            out.append(Contact(self._label[a], self._label[b], force))
        return out

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

    # --- rendering, state ------------------------------------------------------------------
    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        r = self._renderers.get((width, height))
        if r is None:
            r = self._renderers[(width, height)] = mujoco.Renderer(self.model, height, width)
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

    def close(self) -> None:
        # The environment owns its renderer's GL context lifetime too: nothing here outlives a test.
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


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
