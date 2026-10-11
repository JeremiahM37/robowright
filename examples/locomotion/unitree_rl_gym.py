"""Unitree's published locomotion policies, run as they are published.

unitree_rl_gym ships, for each robot, a model (MJCF), a TorchScript checkpoint and a YAML config
its ``deploy/deploy_mujoco/deploy_mujoco.py`` reads: the PD gains, the default pose, the
observation scales and the control decimation. This module reads those files as data (the
repository's Python is never imported), registers the robot with the maker's gains, and wraps the
checkpoint as a robowright policy that builds the observation its deploy script builds.

Nothing here is a copy of their controller: the torques are the maker's PD loop (robowright runs
it inside the physics step, from ``pd_gains``), and the network is their ``.pt`` file.

Point ``ROBOWRIGHT_UNITREE_RL_GYM`` at a checkout to use it instead of the pinned fetch.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO = "https://github.com/unitreerobotics/unitree_rl_gym.git"
COMMIT = "276801e46c5d433564f24658bac64f254b7d2d4b"
DIRECTORIES = ("deploy", "resources/robots")  # what the deploy scripts read; the rest of the repo is not needed
GAIT_PERIOD = 0.8  # seconds: the period the deploy script's gait clock uses


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "robowright" / "unitree_rl_gym"


def checkout() -> Path:
    """The repository at the pinned commit: ``ROBOWRIGHT_UNITREE_RL_GYM`` if set, else fetched once into the cache."""
    env = os.environ.get("ROBOWRIGHT_UNITREE_RL_GYM")
    if env:
        return Path(env)
    d = cache_dir()
    d.parent.mkdir(parents=True, exist_ok=True)
    with open(d.parent / "unitree_rl_gym.lock", "w") as lock:  # serialised across pytest workers
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (d / ".git").exists():
            _git("clone", "--filter=blob:none", "--no-checkout", "--sparse", REPO, str(d))
            _git("sparse-checkout", "set", "--no-cone", *(f"/{x}/" for x in DIRECTORIES), cwd=d)
        _git("checkout", "-q", COMMIT, cwd=d)
    return d


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


@dataclass
class Published:
    """One robot as its makers deploy it in MuJoCo.

    ``xml`` is the scene the deploy script loads; ``robot_xml`` is the robot model it includes,
    which is what robowright takes (the floor and lights come from robowright's own scene).
    """

    name: str
    xml: Path
    policy: Path
    config: dict

    @property
    def robot_xml(self) -> Path:
        for el in ET.parse(self.xml).getroot().iter("include"):
            return (self.xml.parent / el.get("file")).resolve()
        return self.xml

    @property
    def joints(self) -> list[str]:
        m = mujoco.MjModel.from_xml_path(str(self.robot_xml))
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)][: self.config["num_actions"]]


def load(config_name: str = "g1.yaml", repo: Path | None = None) -> Published:
    """The robot a deploy config names: its model, checkpoint and settings, as the config gives them."""
    repo = Path(repo) if repo is not None else checkout()
    config = yaml.safe_load((repo / "deploy/deploy_mujoco/configs" / config_name).read_text())
    sub = lambda s: Path(str(s).replace("{LEGGED_GYM_ROOT_DIR}", str(repo)))
    return Published(
        name=config_name.rsplit(".", 1)[0],
        xml=sub(config["xml_path"]),
        policy=sub(config["policy_path"]),
        config=config,
    )


def register(pub: Published, name: str | None = None):
    """Register the robot with the maker's gains (``kps``/``kds`` in the order of its joints)."""
    from robowright import robots

    m = mujoco.MjModel.from_xml_path(str(pub.robot_xml))
    base = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[0]))
    gains = tuple((float(kp), float(kd)) for kp, kd in zip(pub.config["kps"], pub.config["kds"]))
    return robots.load(
        pub.robot_xml,
        name=name or f"unitree_{pub.name}",
        arm_joints=tuple(pub.joints),
        pd_gains=gains,
        base_body=base,
        base_pos=(0.0, 0.0, float(m.qpos0[2])),
        stand=tuple(float(q) for q in pub.config["default_angles"]),
        family="legged",
    )


def start_height(pub: Published) -> float:
    """The base height the deploy script starts at: the model's own start pose, with the joints at zero."""
    return float(mujoco.MjModel.from_xml_path(str(pub.robot_xml)).qpos0[2])


def _gravity_orientation(q) -> np.ndarray:
    """The gravity vector in the base's axes, as the deploy script computes it (z is +1 when upright)."""
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


class LocomotionPolicy:
    """The checkpoint, behind robowright's ``policy(obs) -> joint targets``.

    Its observation is the deploy script's: base angular velocity, gravity, the command, joint
    positions relative to the default pose, joint velocities, the previous action and a gait
    phase. A policy call every ``control_decimation`` physics steps; robowright's control step is
    that many steps (50 Hz at the model's 0.002 s), so one call is one control step.
    """

    privileged = False

    def __init__(self, pub: Published, cmd=(0.5, 0.0, 0.0)):
        import torch

        self.pub = pub
        c = pub.config
        self.net = torch.jit.load(str(pub.policy), map_location="cpu").eval()
        self.torch = torch
        self.default = np.asarray(c["default_angles"], np.float32)
        self.n = c["num_actions"]
        self.kwargs = dict(ang=c["ang_vel_scale"], pos=c["dof_pos_scale"], vel=c["dof_vel_scale"], act=c["action_scale"])
        self.cmd = np.asarray(cmd, np.float32) * np.asarray(c["cmd_scale"], np.float32)
        self.dt = float(c["simulation_dt"]) * int(c["control_decimation"])
        self.reset()

    def reset(self):
        self.calls = 0
        self.action = np.zeros(self.n, np.float32)

    def __call__(self, obs) -> np.ndarray:
        k = self.calls
        self.calls += 1
        if k == 0:
            # The deploy loop holds the default pose until its first policy call a control step in.
            return self.default.copy()
        k_ = self.kwargs
        ph = (k * self.dt % GAIT_PERIOD) / GAIT_PERIOD
        o = np.zeros(self.pub.config["num_obs"], np.float32)
        n = self.n
        o[:3] = np.asarray(obs["base_ang_vel_body"], np.float32) * k_["ang"]
        o[3:6] = _gravity_orientation(obs["imu_quat"])
        o[6:9] = self.cmd
        o[9 : 9 + n] = (np.asarray(obs["qpos"], np.float32) - self.default) * k_["pos"]
        o[9 + n : 9 + 2 * n] = np.asarray(obs["qvel"], np.float32) * k_["vel"]
        o[9 + 2 * n : 9 + 3 * n] = self.action
        o[9 + 3 * n : 9 + 3 * n + 2] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
        with self.torch.inference_mode():
            out = self.net(self.torch.from_numpy(o).unsqueeze(0)).numpy().squeeze()
        self.action = out.astype(np.float32)
        return self.action * k_["act"] + self.default
