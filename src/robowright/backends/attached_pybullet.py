"""A PyBullet simulation that robowright did not build (panda-gym and other PyBullet environments).

The environment owns a PyBullet physics client and steps it; this adapter reads that client through
its integer id. Bodies are named the way the environment names them when it keeps a registry
(panda-gym's ``sim._bodies_idx``), and otherwise by their URDF name. PyBullet has no site concept
and no name for a link beyond the URDF's, so a link is named by its URDF link name.

Two things PyBullet does not give, and this adapter says so rather than inventing them:

* The clock. PyBullet has no simulation time to read, so time is the number of control steps the
  environment took times its control period (``dt``). The environment's own ``step`` is what counts.
* Controls. PyBullet keeps no actuator state that can be read back, so ``ctrl()`` is NaN.
"""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np

from ..errors import CapabilityError
from .attached_engine import AttachedEngine
from .base import CONTACTS, FORCES, GROUND_TRUTH


def _text(b) -> str:
    return b.decode() if isinstance(b, bytes) else str(b)


def _wxyz(q) -> np.ndarray:
    """PyBullet's quaternions are (x, y, z, w); robowright's are (w, x, y, z), as MuJoCo's."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


class PyBulletEngine(AttachedEngine):
    """Handles are ``("body", id)`` or ``("link", id, link index)``, with the PyBullet body id."""

    engine = "pybullet"
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, FORCES})

    def __init__(self, client: int | None = None, *, seed: int = 0, dt: float | None = None, bodies: dict | None = None, env=None):
        import pybullet as p

        self.p = p
        self.client: int | None = None  # None until the simulation is found (see ensure_bound)
        self._env = env  # searched for its client, and for its control period and names, once it is found
        self._dt_given = dt
        self._bodies_given = bodies
        super().__init__(seed, dt=dt)
        self._physics_dt = 0.0
        self._steps = 0
        self._name_of: dict[int, str] = {}  # body id -> name
        self._bodies: dict[str, int] = {}  # name -> body id
        self._links: dict[str, tuple[int, int]] = {}  # URDF link name -> (body id, link index)
        self._joints: dict[str, tuple[int, int]] = {}  # joint name -> (body id, joint index)
        self._joint_order: list[tuple[int, int, str]] = []  # the qpos order: every revolute or prismatic joint
        self._mass: dict[int, float] = {}
        if client is not None:
            self._bind(int(client))

    @property
    def pending(self) -> bool:
        return self.client is None

    def ensure_bound(self) -> bool:
        """Find the client if it was not known at attach time: an environment often makes its client on its first reset."""
        if self.client is not None:
            return False
        cid = _client_of(self._env) if self._env is not None else None
        if cid is None:
            cid = _only_connected(self.p)  # no handle anywhere: the one connection pybullet knows of
        if cid is None:
            return False
        self._bind(cid)
        return True

    def rebind(self, env) -> None:
        if self._env is None:
            self._env = env
        self.ensure_bound()

    def _bind(self, client: int) -> None:
        self.client = client
        self._physics_dt = float(self.p.getPhysicsEngineParameters(physicsClientId=client)["fixedTimeStep"])
        core = getattr(self._env, "unwrapped", self._env)
        if self._dt_given is not None:
            self._dt = float(self._dt_given)
        else:
            self._dt = _env_dt(core) or self._physics_dt  # the environment's control period, else one physics step
        registry = self._bodies_given if self._bodies_given is not None else _registry_of(core)
        self._scan(registry or {})

    def _scan(self, registry: dict) -> None:
        p, c = self.p, self.client
        registry_names = {bid: name for name, bid in registry.items()}
        # Names the environment gave win; the rest are named by URDF in body order, and a repeated name is suffixed (name#2)
        # so that every body is listed (a repeated name used to collapse them into one).
        order = sorted(range(p.getNumBodies(physicsClientId=c)), key=lambda bid: (bid not in registry_names, bid))
        for bid in order:
            info = p.getBodyInfo(bid, physicsClientId=c)
            base = registry_names.get(bid) or _text(info[1]) or _text(info[0]) or f"body{bid}"
            name = _unique(base, self._bodies)
            self._name_of[bid] = name
            self._bodies[name] = bid
            self._mass[bid] = float(p.getDynamicsInfo(bid, -1, physicsClientId=c)[0])
        for bid in range(p.getNumBodies(physicsClientId=c)):
            for j in range(p.getNumJoints(bid, physicsClientId=c)):
                ji = p.getJointInfo(bid, j, physicsClientId=c)
                self._links.setdefault(_text(ji[12]), (bid, j))
                if ji[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    jname = _unique(_text(ji[1]) or f"joint{j}", self._joints)
                    self._joints[jname] = (bid, j)
                    self._joint_order.append((bid, j, jname))
        self.joint_names = [n for _, _, n in self._joint_order]
        # A free body is a dynamic one with no joints: the things a test can place and push (not the table or the robot).
        self.free_bodies = [
            name for name, bid in self._bodies.items() if self._mass[bid] > 0 and self.p.getNumJoints(bid, physicsClientId=c) == 0
        ]

    # --- naming -----------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        if kind == "body":
            return list(self._bodies)
        if kind == "geom" or kind == "site":
            return []  # PyBullet has neither: a link is its own thing, so nothing is named that way
        raise ValueError(f"unknown kind {kind!r}")

    def resolve(self, name: str):
        if name in self._bodies:
            return ("body", self._bodies[name])
        if name in self._links:
            bid, j = self._links[name]
            return ("link", bid, j)
        return None

    def label(self, handle) -> str:
        if handle[0] == "body":
            return self._name_of[handle[1]]
        return _link_name(self.p, self.client, handle[1], handle[2])

    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        p, c = self.p, self.client
        if handle[0] == "body":
            pos, q = p.getBasePositionAndOrientation(handle[1], physicsClientId=c)
        else:
            st = p.getLinkState(handle[1], handle[2], computeForwardKinematics=1, physicsClientId=c)
            pos, q = st[4], st[5]
        return np.asarray(pos, dtype=float), _wxyz(q)

    def velocity(self, handle) -> np.ndarray:
        p, c = self.p, self.client
        if handle[0] == "body":
            lin, ang = p.getBaseVelocity(handle[1], physicsClientId=c)
        else:
            st = p.getLinkState(handle[1], handle[2], computeLinkVelocity=1, physicsClientId=c)
            lin, ang = st[6], st[7]
        return np.concatenate([np.asarray(lin, float), np.asarray(ang, float)])

    # --- contacts ----------------------------------------------------------------------
    def _raw_contacts(self):
        """``(body a, link a, body b, link b, normal force)`` for every contact between two different bodies."""
        out = []
        for c in self.p.getContactPoints(physicsClientId=self.client):  # [1]/[2] bodies, [3]/[4] links, [9] normal force
            if c[1] == c[2]:
                continue
            out.append((c[1], c[3], c[2], c[4], abs(float(c[9]))))
        return out

    def _side(self, bid: int, link: int) -> str:
        return self._name_of[bid] if link < 0 else _link_name(self.p, self.client, bid, link)

    def contact_pairs(self) -> list[tuple]:
        return [(self._side(a, la), self._side(b, lb), f) for a, la, b, lb, f in self._raw_contacts()]

    def contacts_of(self, handle) -> list[str]:
        out = set()
        for a, la, b, lb, _ in self._raw_contacts():
            if handle[0] == "body":
                if handle[1] == a:
                    other = (b, lb)
                elif handle[1] == b:
                    other = (a, la)
                else:
                    continue
            else:
                if (a, la) == (handle[1], handle[2]):
                    other = (b, lb)
                elif (b, lb) == (handle[1], handle[2]):
                    other = (a, la)
                else:
                    continue
            out.add(self._name_of[other[0]])
            if other[1] >= 0:
                out.add(_link_name(self.p, self.client, other[0], other[1]))
        return sorted(out)

    # --- joints and the free bodies ---------------------------------------------------
    def joint_qadr(self, name: str) -> int:
        """The index of a joint's value in ``qpos()``: a PyBullet joint has one scalar value, so no address."""
        for i, (_, _, n) in enumerate(self._joint_order):
            if n == name:
                return i
        if name in self._joints:  # a fixed or spherical joint: not one scalar
            raise CapabilityError(
                f"joint {name!r} is not a revolute or prismatic joint: it has no single value; use world.body() for its pose"
            )
        raise KeyError(f"no joint named {name!r}; joints: {', '.join(self.joint_names) or 'none'}")

    def joint_kinds(self) -> dict[str, str]:
        """Each hinge or slide joint's kind, as PyBullet reports its joint type."""
        p = self.p
        return {
            name: "hinge" if p.getJointInfo(bid, j, physicsClientId=self.client)[2] == p.JOINT_REVOLUTE else "slide"
            for name, (bid, j) in self._joints.items()
        }

    def joint_value(self, name: str) -> float:
        self.joint_qadr(name)
        bid, j = self._joints[name]
        return float(self.p.getJointState(bid, j, physicsClientId=self.client)[0])

    def set_joint(self, name: str, value: float) -> None:
        if name not in self._joints or name not in self.joint_names:
            raise ValueError(f"{name!r} is not a revolute or prismatic joint in this simulation")
        bid, j = self._joints[name]
        self.p.resetJointState(bid, j, float(value), targetVelocity=0.0, physicsClientId=self.client)

    def qpos(self) -> np.ndarray:
        return np.array([self.p.getJointState(b, j, physicsClientId=self.client)[0] for b, j, _ in self._joint_order], dtype=float)

    def qvel(self) -> np.ndarray:
        return np.array([self.p.getJointState(b, j, physicsClientId=self.client)[1] for b, j, _ in self._joint_order], dtype=float)

    def ctrl(self) -> np.ndarray:
        return np.full(len(self._joint_order), np.nan)  # PyBullet keeps no readable actuator state; see the module docstring

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        if name not in self.free_bodies:
            raise ValueError(f"{name!r} is not a free body, so its pose cannot be set (free bodies: {', '.join(self.free_bodies)})")
        bid = self._bodies[name]
        p, c = self.p, self.client
        if quat is None:
            _, q = p.getBasePositionAndOrientation(bid, physicsClientId=c)
        else:
            q = [quat[1], quat[2], quat[3], quat[0]]  # back to PyBullet's (x, y, z, w)
        p.resetBasePositionAndOrientation(bid, list(map(float, pos)), q, physicsClientId=c)
        p.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=c)

    def apply_force(self, name: str, force) -> None:
        """A force for one control step, delivered as the impulse that step would give.

        PyBullet clears an external force after the physics step it was applied to (measured: the velocity of a
        free sphere rose by one step's worth and no more). A control step is ``control_dt / physics_dt`` physics
        steps, so the force is applied once, scaled by that count: the same impulse over the control step. The
        exact force-over-time within the step is not reproduced, and contacts within the step see the impulse at once.
        """
        bid = self._bodies.get(name)
        if bid is None:
            raise KeyError(f"no body named {name!r} in this simulation")
        n = max(1, int(round(self._dt / self._physics_dt)))
        pos, _ = self.p.getBasePositionAndOrientation(bid, physicsClientId=self.client)
        force_n = [float(f) * n for f in force]
        self.p.applyExternalForce(bid, -1, force_n, list(pos), self.p.WORLD_FRAME, physicsClientId=self.client)

    # --- clock -------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return self._steps * self._dt

    def note_step(self) -> None:
        self._steps += 1

    def trace_meta(self) -> dict:
        return {"engine": self.engine}


def _link_name(p, client: int, bid: int, link: int) -> str:
    return _text(p.getJointInfo(bid, link, physicsClientId=client)[12])


_CLIENT_NAMES = (
    "physicsClientId",
    "_physicsClientId",
    "bullet_client_id",
    "_bullet_client_id",
    "CLIENT",
    "PYB_CLIENT",
    "physics_client_id",
)
_NOT_WALKED = (int, float, str, bytes, bool, type, list, tuple, dict, set)


def _unique(name: str, taken) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name}#{k}" in taken:
        k += 1
    return f"{name}#{k}"


def _connected(p, cid: int) -> bool:
    return bool(p.isConnected(physicsClientId=int(cid)))


def _only_connected(p) -> int | None:
    """The one connected physics client, or None when none is. Several is an error: the caller must say which."""
    ids = [i for i in range(256) if _connected(p, i)]
    if len(ids) > 1:
        raise ValueError(f"several PyBullet clients are connected (ids {ids}): pass rw.attach(engine='pybullet', client=<id>)")
    return ids[0] if ids else None


def _client_of(env) -> int | None:
    """The physics client an environment holds, found by walking its attributes breadth first (wrappers included).

    panda-gym keeps a ``BulletClient`` on ``env.unwrapped.sim.physics_client`` (its ``_client`` is the id); others keep the
    bare id on the environment (``CLIENT``, ``bullet_client_id``, ``physicsClientId``). A wrapper's inner environment is
    reached through its own attributes. An id that is no longer connected is ignored.
    """
    import pybullet as p

    seen: set[int] = set()
    queue: list[tuple[object, int]] = [(env, 0)]
    while queue and len(seen) < 500:  # wrappers stack (TimeLimit, OrderEnforcing, ...): depth and count are both bounded
        obj, depth = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        for name, value in _attributes(obj):
            cid = None
            if isinstance(value, int) and not isinstance(value, bool) and name in _CLIENT_NAMES:
                cid = value
            elif type(value).__name__ == "BulletClient" and isinstance(getattr(value, "_client", None), int):
                cid = value._client
            if cid is not None and _connected(p, cid):
                return cid
            if depth < 6 and _walkable(value):
                queue.append((value, depth + 1))
    return None


def _attributes(obj):
    try:
        items = list(vars(obj).items())
    except TypeError:
        return []
    return [(k, v) for k, v in items if not k.startswith("__")]


def _walkable(value) -> bool:
    if isinstance(value, _NOT_WALKED) or inspect.ismodule(value) or inspect.isroutine(value):
        return False
    return hasattr(value, "__dict__")


def _env_dt(core) -> float | None:
    """The control period an environment reports: panda-gym's ``sim.dt``, or the ``CTRL_TIMESTEP``/``dt`` of the env itself."""
    sim = getattr(core, "sim", None)
    for cand in (getattr(sim, "dt", None), getattr(core, "CTRL_TIMESTEP", None), getattr(core, "dt", None)):
        if isinstance(cand, (int, float)) and not isinstance(cand, bool) and cand > 0:
            return float(cand)
    return None


def _registry_of(core) -> dict | None:
    """panda-gym's own body names (``sim._bodies_idx``), or None."""
    registry = getattr(getattr(core, "sim", None), "_bodies_idx", None)
    return dict(registry) if isinstance(registry, dict) else None


def pybullet_engine_of(env, *, seed: int = 0, dt: float | None = None) -> PyBulletEngine | None:
    """The PyBullet adapter for an environment that holds a PyBullet client, or None.

    The client is found by :func:`_client_of`, else it is the one connection pybullet knows of (racecar_gym keeps no handle).
    When there is none yet (an environment that makes its client on its first
    reset), the environment is taken as a PyBullet one if it has a client-shaped slot that is still unset, such as
    bullet-safety-gym's ``bullet_client_id``; the adapter then binds at the first reset or step.
    """
    if importlib.util.find_spec("pybullet") is None:
        return None
    core = getattr(env, "unwrapped", env)
    cid = _client_of(env)
    if cid is None:
        cid = _only_connected(importlib.import_module("pybullet"))  # no handle (racecar_gym): the one connection there is
    if cid is None and not any(n in _CLIENT_NAMES or "bullet" in n for n in vars(core)):
        return None
    return PyBulletEngine(cid, seed=seed, dt=dt, env=env)
