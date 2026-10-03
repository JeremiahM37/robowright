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
from .robot import GRIPPER_CLOSED, GRIPPER_OPEN, TABLE_CLEARANCE, _kinematics, home_q, solve_ik


class ScriptedPickPlace:
    def __init__(
        self,
        object: str = "cube",
        target: str = "bin",
        chunk: int = 10,
        speed: float = 1.5,
        tolerance: float = 0.012,
        tool_speed: float = 0.5,
    ):
        self.object, self.target = object, target
        self.chunk, self.speed, self.tolerance, self.tool_speed = chunk, speed, tolerance, tool_speed
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
        }

    def reset(self):
        self.phase = 0
        self.wait = 0
        self.cmd = None
        self.goal = None

    def _bind(self, robot: str):
        if self.robot != robot:
            self.robot = robot
            self.kin = _kinematics(robot)
            self.home = home_q(robot)
            self.reach = robots.get(robot).derived.finger_reach
            self.min_z = self.reach + TABLE_CLEARANCE

    def _ik(self, p, seed, yaw):
        return solve_ik(self.kin, p, seed, self.home, yaw=yaw)[0]

    def _plan(self, objs):
        op, oq = objs[self.object]
        tp, tq = objs[self.target]
        oyaw = 2 * np.arctan2(oq[3], oq[0])
        tyaw = 2 * np.arctan2(tq[3], tq[0])
        grasp = np.array([op[0], op[1], max(op[2], self.min_z)])
        rim = tp[2] + 0.04
        release = rim + max(0.015, self.reach + 0.006)
        return [  # (tcp goal, yaw, gripper, settle steps)
            (grasp + [0, 0, 0.05], oyaw, GRIPPER_OPEN, 0),
            (grasp, oyaw, GRIPPER_OPEN, 0),
            (grasp, oyaw, GRIPPER_CLOSED, 20),
            (grasp + [0, 0, 0.05], oyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], release + 0.03]), tyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], release]), tyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], release]), tyaw, GRIPPER_OPEN, 15),
            (np.array([tp[0], tp[1], release + 0.03]), tyaw, GRIPPER_OPEN, 0),
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
        tcp = self.kin.tcp(q[:n])
        if np.allclose(self.cmd, goal) and np.linalg.norm(tcp - goal_p) < self.tolerance:
            self.wait += self.chunk
            if self.wait >= settle:
                self.phase, self.wait, self.goal = self.phase + 1, 0, None
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
