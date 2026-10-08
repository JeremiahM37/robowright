"""Policies: anything that maps an observation to joint targets.

A policy is a callable ``policy(obs) -> action`` where ``action`` is the
arm's joint position targets followed by a gripper opening (0 closed, 1 open),
or an ``(n, dof + 1)`` action chunk executed one control step per row. An optional
``reset()`` is called before each rollout, and an optional ``to_config()``
lets codegen recreate the policy in a generated test.

``ScriptedPickPlace`` is a closed-loop, chunked reference policy. It stands
in for a learned policy in examples and benchmarks: it reads joint
encoders (so injected sensor noise hurts it) and ground-truth object poses.
"""

from __future__ import annotations

import numpy as np

from . import robots
from .errors import TooWideError
from .robot import GRIPPER_CLOSED, GRIPPER_OPEN, _kinematics, home_q, release_reach, solve_ik, top_grasp, unfold, widest_gap


class ScriptedPickPlace:
    privileged = True  # reads object poses: run_policy observes them for it

    def __init__(
        self,
        object: str = "cube",
        target: str = "bin",
        chunk: int = 10,
        speed: float = 1.5,
        tolerance: float = 0.012,
        tool_speed: float = 0.5,
        size: tuple = (0.025, 0.025),
    ):
        self.object, self.target = object, target
        self.chunk, self.speed, self.tolerance, self.tool_speed = chunk, speed, tolerance, tool_speed
        self.size = tuple(size)  # the object's width and height: how far to open, how low to go
        self.robot = None
        self.reset()

    def to_config(self) -> dict:
        return {
            "object": self.object,
            "target": self.target,
            "chunk": self.chunk,
            "speed": self.speed,
            "tolerance": self.tolerance,
            "tool_speed": self.tool_speed,
            "size": list(self.size),
        }

    def reset(self):
        self.phase = 0
        self.wait = 0
        self.cmd = None
        self.goal = None
        self.readings = []
        self.arrived = None

    def _bind(self, robot: str):
        if self.robot != robot:
            model = robots.get(robot)
            if model.has_gripper and widest_gap(model) < self.size[0]:
                opens = widest_gap(model)
                raise TooWideError(f"{self.object} is {self.size[0] * 1000:.0f} mm across; this gripper opens {opens * 1000:.0f} mm")
            self.robot = robot
            self.kin = _kinematics(robot)
            self.home = home_q(robot)
            self.der = model.derived

    def _ik(self, p, seed, yaw):
        q, err = solve_ik(self.kin, p, seed, self.home, yaw=yaw)
        return unfold(self.robot, self.kin, q, p, seed, self.home, yaw=yaw) if err < 1e-3 else q

    def _plan(self, objs):
        op, oq = objs[self.object]
        tp, tq = objs[self.target]
        oyaw = 2 * np.arctan2(oq[3], oq[0])
        tyaw = 2 * np.arctan2(tq[3], tq[0])
        width, h = self.size
        opening, z = top_grasp(self.der, op[2], op[2] + h / 2, op[2] - h / 2, width)
        grasp = np.array([op[0], op[1], z])
        open_ = GRIPPER_CLOSED + opening * (GRIPPER_OPEN - GRIPPER_CLOSED)
        rim = tp[2] + 0.04
        release = rim + max(0.015, release_reach(self.der, opening) + 0.006)
        return [  # (tcp goal, yaw, gripper, settle steps)
            (grasp + [0, 0, 0.05], oyaw, open_, 0),
            (grasp, oyaw, open_, 0),
            (grasp, oyaw, GRIPPER_CLOSED, 20),
            (grasp + [0, 0, 0.05], oyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], release + 0.03]), tyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], release]), tyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], release]), tyaw, open_, 15),
            (np.array([tp[0], tp[1], release + 0.03]), tyaw, open_, 0),
        ]

    def __call__(self, obs) -> np.ndarray:
        self._bind(obs.get("robot", "so101"))
        q = np.asarray(obs["qpos"], float)
        n = len(q) - 1
        if self.cmd is None:
            self.cmd = q.copy()
        plan = self._plan(obs["objects"])
        if self.phase >= len(plan):
            return np.tile(self.cmd, (self.chunk, 1))
        if self.goal is None:
            # Re-plan from fresh observations at the start of every phase, then
            # hold that goal: the object moves once it is grasped.
            goal_p, yaw, grip, settle = plan[self.phase]
            self.goal = (np.concatenate([self._ik(goal_p, self.cmd[:n], yaw), [grip]]), goal_p, settle)
        goal, goal_p, settle = self.goal
        # Arrived when the encoders put the tool at the goal, or the mean of the readings taken
        # while holding it (the last 20) does: noisy encoders (0.02 rad on a 1.3 m UR10e is
        # 2.6 cm at the tool) rarely give one reading within tolerance, and it waited there
        # until it timed out.
        held = np.allclose(self.cmd, goal)
        self.readings = (self.readings + [q[:n]])[-20:] if held else []
        off = np.linalg.norm(self.kin.tcp(q[:n]) - goal_p)
        if held and len(self.readings) > 1:
            off = min(off, np.linalg.norm(self.kin.tcp(np.mean(self.readings, axis=0)) - goal_p))
        # A step that only works the gripper leaves the arm where the last one saw it arrive.
        # Checking again under noisy encoders kept the jaws squeezing, and a cube turns in the
        # Panda's fingers in PyBullet the longer they squeeze (0.07 rad after 1.2 s, 0.5 after 1.8).
        stays = self.arrived is not None and np.linalg.norm(goal_p - self.arrived) < 0.002
        if held and (off < self.tolerance or stays):
            self.wait += self.chunk
            if self.wait >= settle:
                self.phase, self.wait, self.goal, self.arrived = self.phase + 1, 0, None, goal_p
            return np.tile(self.cmd, (self.chunk, 1))
        # Stream a chunk that moves the command toward the goal at bounded joint and tool speeds.
        step = self.speed * 0.02
        out = []
        for _ in range(self.chunk):
            delta = np.clip(goal - self.cmd, -step, step)
            moved = np.linalg.norm(self.kin.tcp(self.cmd[:n] + delta[:n]) - self.kin.tcp(self.cmd[:n]))
            if moved > self.tool_speed * 0.02:
                delta[:n] *= self.tool_speed * 0.02 / moved
            self.cmd = self.cmd + delta
            out.append(self.cmd.copy())
        return np.array(out)


class Replay:
    """Plays back a fixed joint trajectory (e.g. from a recorded demo), then holds."""

    def __init__(self, actions):
        self.actions = np.asarray(actions, float)
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self, obs):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a
