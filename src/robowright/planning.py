"""Collision-free arm motion: joint-space paths that keep the arm off the table and the objects.

robowright's moves interpolate joints, which is safe while the hand comes from above. A hand
turned to the side, or an arm folding down low, can sweep through the table or knock things
over on the way. :class:`Planner` finds a path that does not:

* collisions are checked on a MuJoCo copy of the scene, whatever engine runs the simulation,
  with the objects where the simulation has them and a held object carried with the hand;
* the search is RRT-Connect in joint space, from a seeded generator (the same scene plans the
  same path), followed by shortcutting;
* a straight joint-space move that is already clear is used as it is.

The robot (and what it holds) against everything else counts, and its links against each other
(the hand's own parts aside); contacts present where the arm starts or ends (a fingertip resting
on the table, a shoulder turning on the table's surface) do not.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .errors import UnreachableError

CLEARANCE = 0.004  # metres the arm keeps from the table and objects
_STEP = 0.02  # rad: how finely an edge is checked
_EXTEND = 0.25  # rad: how far a tree grows towards a sample at once


class Planner:
    """Collision checking and path search for the arm of ``world``'s robot."""

    def __init__(self, world):
        from .backends.mujoco_backend import build_spec
        from .robots import PREFIX
        from .robots.model import joint_followers

        self.world = world
        rm = world.robot.model
        spec = build_spec(world.spec)
        self.m = spec.compile()
        self.m.geom_margin[:] = CLEARANCE
        self.d = mujoco.MjData(self.m)
        m = self.m
        arm = [m.joint(PREFIX + j).id for j in rm.arm_joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in arm])
        self.followers = joint_followers(m, arm)
        self.fingers = []
        if rm.has_gripper:
            for name, (closed, opened) in rm.derived.gripper_joints.items():
                self.fingers.append((m.joint(PREFIX + name).qposadr[0], closed, opened))
        # The robot's moving parts: its base, fixed where it stands on the table, is not in the way.
        robot = {b for b in range(m.nbody) if m.body(b).name.startswith(PREFIX) and m.body_weldid[b] != 0}
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in robot}
        hand = m.body(PREFIX + rm.hand).id
        self.hand = hand
        below = {hand}
        for b in range(hand + 1, m.nbody):
            if m.body_parentid[b] in below:
                below.add(b)
        self.hand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in below}
        self.objects = {o.name: m.body(o.name).id for o in world.spec.objects}
        self.free = {o.name: m.joint(f"{o.name}/free").qposadr[0] for o in world.spec.objects if not o.static and o.kind != "bin"}
        self.kin = world.robot.kin
        self.held = None
        self.allowed: set = set()
        self.base_contacts: set = set()

    # --- the scene ----------------------------------------------------------------------------
    def sync(self, held: str | None = None) -> None:
        """Objects where the simulation has them, the gripper as open as it is, and ``held``
        carried with the hand from where it is now."""
        b = self.world.backend
        for name, adr in self.free.items():
            p, q = b.object_pose(name)
            self.d.qpos[adr : adr + 3], self.d.qpos[adr + 3 : adr + 7] = p, q
        self.opening = float(np.clip(self.world.robot.qpos()[-1], 0, 1)) if self.fingers else 0.0
        self.held = None
        self.allowed = set()
        if held is not None and held in self.free:
            self._pose(self.world.robot._target[: self.world.robot.n_arm])
            R, p = self.d.xmat[self.hand].reshape(3, 3), self.d.xpos[self.hand]
            o = self.objects[held]
            self.held = (held, R.T @ (self.d.xpos[o] - p), R.T @ self.d.xmat[o].reshape(3, 3))
            self.held_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == o}
        # What the arm touches where it is (a shoulder turning on the table's surface) is how it
        # is mounted, not in its way.
        self.base_contacts = self.contacts(self.world.robot._target[: self.world.robot.n_arm])

    def _pose(self, q) -> None:
        d = self.d
        d.qpos[self.qadr] = q
        for qa, _, i, offset, ratio in self.followers:
            d.qpos[qa] = offset + ratio * q[i]
        for qa, closed, opened in self.fingers:
            d.qpos[qa] = closed + self.opening * (opened - closed)
        mujoco.mj_kinematics(self.m, d)
        if self.held is not None:
            name, p_rel, R_rel = self.held
            R, p = d.xmat[self.hand].reshape(3, 3), d.xpos[self.hand]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (R @ R_rel).ravel())
            adr = self.free[name]
            d.qpos[adr : adr + 3], d.qpos[adr + 3 : adr + 7] = p + R @ p_rel, quat
            mujoco.mj_kinematics(self.m, d)

    def contacts(self, q) -> set[tuple[int, int]]:
        """Pairs of geoms (moving, still) closer than the clearance with the arm at ``q``."""
        self._pose(q)
        mujoco.mj_collision(self.m, self.d)
        moving = self.robot_geoms | (self.held_geoms if self.held is not None else set())
        out = set()
        for c in self.d.contact[: self.d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist > CLEARANCE:
                continue
            if g1 in self.robot_geoms and g2 in self.robot_geoms:
                # The arm folding into itself counts too; the hand's own parts touching do not.
                if not (g1 in self.hand_geoms and g2 in self.hand_geoms) and c.dist < 0:
                    out.add((min(g1, g2), max(g1, g2)))
                continue
            if (g1 in moving) == (g2 in moving):
                continue
            g1, g2 = (g1, g2) if g1 in moving else (g2, g1)
            if self.held is not None and g1 in self.hand_geoms and g2 in self.held_geoms:
                continue
            out.add((g1, g2))
        return out

    def clear(self, q) -> bool:
        return not (self.contacts(q) - self.allowed - self.base_contacts)

    def edge_clear(self, a, b) -> bool:
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / _STEP)))
        return all(self.clear(a + (b - a) * i / n) for i in range(1, n + 1))

    # --- the search ---------------------------------------------------------------------------
    def plan(self, start, goal, iterations: int = 3000) -> list[np.ndarray]:
        """A collision-free joint path from ``start`` to ``goal`` (both included)."""
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        # What touches at both ends is part of the task (a fingertip on the table), not in the way.
        self.allowed = self.contacts(start) | self.contacts(goal)
        try:
            if self.edge_clear(start, goal):
                return [start, goal]
            rng = np.random.default_rng(self.world.seed)
            lo, hi = np.maximum(self.kin.lower, -np.pi), np.minimum(self.kin.upper, np.pi)
            a, b = [start], [goal]
            pa, pb = [-1], [-1]
            for _ in range(iterations):
                sample = goal if rng.random() < 0.1 else rng.uniform(lo, hi)
                i = self._extend(a, pa, sample)
                if i is not None:
                    j = self._connect(b, pb, a[i])
                    if j is not None and np.allclose(b[j], a[i]):
                        path = self._path(a, pa, i)[::-1] + self._path(b, pb, j)[1:]
                        if a[0] is not start:
                            path = path[::-1]
                        return self._shortcut(path, rng)
                a, b, pa, pb = b, a, pb, pa
            raise UnreachableError(f"no collision-free path found in {iterations} tries")
        finally:
            self.allowed = set()

    def _extend(self, tree, parent, target):
        near = int(np.argmin([np.max(np.abs(n - target)) for n in tree]))
        d = target - tree[near]
        step = np.max(np.abs(d))
        new = target if step <= _EXTEND else tree[near] + d * (_EXTEND / step)
        if not self.edge_clear(tree[near], new):
            return None
        tree.append(new)
        parent.append(near)
        return len(tree) - 1

    def _connect(self, tree, parent, target):
        last = None
        while True:
            i = self._extend(tree, parent, target)
            if i is None:
                return last
            last = i
            if np.allclose(tree[i], target):
                return i

    @staticmethod
    def _path(tree, parent, i) -> list[np.ndarray]:
        out = []
        while i != -1:
            out.append(tree[i])
            i = parent[i]
        return out  # from i back to the root

    def _shortcut(self, path, rng, tries: int = 80) -> list[np.ndarray]:
        path = list(path)
        for _ in range(tries):
            if len(path) < 3:
                break
            i, j = sorted(rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self.edge_clear(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path
