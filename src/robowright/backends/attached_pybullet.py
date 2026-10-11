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

    def __init__(self, client: int, *, seed: int = 0, dt: float | None = None, bodies: dict | None = None):
        import pybullet as p

        self.p = p
        self.client = int(client)
        if dt is None:
            dt = float(p.getPhysicsEngineParameters(physicsClientId=self.client)["fixedTimeStep"])
        super().__init__(seed, dt=dt)
        self._steps = 0
        self._name_of: dict[int, str] = {}  # body id -> name
        self._bodies: dict[str, int] = {}  # name -> body id
        self._links: dict[str, tuple[int, int]] = {}  # URDF link name -> (body id, link index)
        self._joints: dict[str, tuple[int, int]] = {}  # joint name -> (body id, joint index)
        self._joint_order: list[tuple[int, int, str]] = []  # the qpos order: every revolute or prismatic joint
        self._mass: dict[int, float] = {}
        self._scan(bodies or {})

    def _scan(self, registry: dict) -> None:
        p, c = self.p, self.client
        registry_names = {bid: name for name, bid in registry.items()}
        for bid in range(p.getNumBodies(physicsClientId=c)):
            info = p.getBodyInfo(bid, physicsClientId=c)
            name = registry_names.get(bid) or _text(info[1]) or _text(info[0]) or f"body{bid}"
            self._name_of[bid] = name
            self._bodies.setdefault(name, bid)
            self._mass[bid] = float(p.getDynamicsInfo(bid, -1, physicsClientId=c)[0])
            for j in range(p.getNumJoints(bid, physicsClientId=c)):
                ji = p.getJointInfo(bid, j, physicsClientId=c)
                jname = _text(ji[1])
                self._links.setdefault(_text(ji[12]), (bid, j))
                if ji[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    self._joints.setdefault(jname, (bid, j))
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
        for c in self.p.getContactPoints(physicsClientId=self.client):
            if c[2] == c[3]:
                continue
            out.append((c[2], c[4], c[3], c[5], abs(float(c[9]))))
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
        """Applied for the next physics step only: PyBullet clears external forces after each step, so a push
        that lasts several steps is re-applied on each one (the core does that)."""
        bid = self._bodies.get(name)
        if bid is None:
            raise KeyError(f"no body named {name!r} in this simulation")
        pos, _ = self.p.getBasePositionAndOrientation(bid, physicsClientId=self.client)
        self.p.applyExternalForce(bid, -1, list(map(float, force)), list(pos), self.p.WORLD_FRAME, physicsClientId=self.client)

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


def pybullet_engine_of(env, *, seed: int = 0, dt: float | None = None) -> PyBulletEngine | None:
    """The PyBullet adapter for an environment that holds a PyBullet client, or None.

    panda-gym keeps a ``BulletClient`` on ``env.unwrapped.sim`` and names its bodies in ``sim._bodies_idx``;
    its control period is ``sim.dt``.
    """
    core = getattr(env, "unwrapped", env)
    sim = getattr(core, "sim", None)
    pc = getattr(sim, "physics_client", None)
    client = getattr(pc, "_client", None)
    if not isinstance(client, int):
        return None
    registry = getattr(sim, "_bodies_idx", None)
    if dt is None:
        sim_dt = getattr(sim, "dt", None)
        if isinstance(sim_dt, (int, float)):
            dt = float(sim_dt)
    return PyBulletEngine(client, seed=seed, dt=dt, bodies=dict(registry) if isinstance(registry, dict) else None)
