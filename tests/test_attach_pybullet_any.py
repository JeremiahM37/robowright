"""rw.attach on a PyBullet simulation the way other projects hold one, with plain pybullet and no third-party package.

Each test builds a small scene the way the published environments do: a client held in a private attribute (panda-gym's
BulletClient), a bare module-level connection with no handle (racecar_gym), a client made on reset (bullet-safety-gym,
bullet3's envs), a repeated body name (bullet-safety-gym's puddles, the reacher's links), and an environment whose own
loop calls the engine's step rather than env.step. The published environments themselves are in examples/environments.
"""

import os

import numpy as np
import pybullet as p
import pybullet_data
import pytest

import robowright as rw
from robowright import expect
from robowright.attached import _make_library
from robowright.codegen import generate
from robowright.errors import CapabilityError
from robowright.replay import _make_from_spec

CUBE = os.path.join(pybullet_data.getDataPath(), "cube.urdf")
KUKA = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")


class BulletClient:
    """Stands in for pybullet_utils' BulletClient: an object holding the client id as ``_client``."""

    def __init__(self, client: int):
        self._client = client


@pytest.fixture(autouse=True)
def _disconnect_everything():
    yield
    for cid in range(256):
        if p.isConnected(physicsClientId=cid):
            p.disconnect(physicsClientId=cid)


def _cubes(client: int, n: int = 2):
    return [p.loadURDF(CUBE, [float(i), 0.0, 0.5], physicsClientId=client) for i in range(n)]


class HeldClientEnv:
    """panda-gym's shape: the client is held on a private attribute, and the env steps it itself."""

    def __init__(self, dt: float = 1 / 120):
        self._pybullet_client = BulletClient(p.connect(p.DIRECT))
        self._dt = dt
        p.setTimeStep(dt, physicsClientId=self._pybullet_client._client)
        _cubes(self._pybullet_client._client)

    def reset(self, *, seed=None, options=None):
        return None, {}

    def step(self, action=None):
        p.stepSimulation(physicsClientId=self._pybullet_client._client)
        return None, 0.0, False, False, {}


class ClientOnReset:
    """bullet-safety-gym's shape: no client until the first reset, and only then is the id known."""

    def __init__(self):
        self.bullet_client_id = None
        self.dt = 1 / 100

    def reset(self, *, seed=None, options=None):
        if self.bullet_client_id is None:
            self.bullet_client_id = p.connect(p.DIRECT)
            p.setTimeStep(1 / 240, physicsClientId=self.bullet_client_id)
            _cubes(self.bullet_client_id)
        return None, {}

    def step(self, action=None):
        p.stepSimulation(physicsClientId=self.bullet_client_id)
        return None, 0.0, False, False, {}


class Wrapper:
    """A gym-style wrapper: its ``step`` calls the inner environment's, which is the unwrapped one."""

    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action=None):
        return self.env.step(action)


class Core:
    """The unwrapped environment: its own step is what a loop that bypasses the wrapper calls."""

    def __init__(self):
        self._pybullet_client = BulletClient(p.connect(p.DIRECT))
        self.unwrapped = self
        _cubes(self._pybullet_client._client)

    @property
    def cid(self):
        return self._pybullet_client._client

    def reset(self, *, seed=None, options=None):
        return None, {}

    def step(self, action=None):
        p.stepSimulation(physicsClientId=self.cid)
        return None, 0.0, False, False, {}


def test_a_client_held_in_a_private_attribute_is_found_from_the_environment():
    env = HeldClientEnv()
    w = rw.attach(env)
    assert w.bodies() == ["cube", "cube#2"]
    assert w.backend.client == env._pybullet_client._client
    assert w.backend.control_dt == pytest.approx(1 / 120)  # the environment's own step period is the control period
    w.close(failed=False)


def test_a_bare_connection_with_no_handle_is_found_by_attach_with_no_argument():
    cid = p.connect(p.DIRECT)
    _cubes(cid)
    w = rw.attach()  # no environment, no client: the one connection pybullet knows of
    assert w.backend.client == cid
    assert w.bodies() == ["cube", "cube#2"]
    w.close(failed=False)


def test_a_bare_connection_is_found_by_engine_name_alone():
    cid = p.connect(p.DIRECT)
    _cubes(cid, 1)
    w = rw.attach(engine="pybullet")
    assert w.backend.client == cid
    w.close(failed=False)


def test_two_connections_with_no_handle_are_an_error_that_names_the_fix():
    p.connect(p.DIRECT)
    p.connect(p.DIRECT)
    with pytest.raises(ValueError, match="client=<id>"):
        rw.attach()


def test_a_client_made_on_reset_binds_at_the_first_reset_when_attach_came_first():
    env = ClientOnReset()
    w = rw.attach(env)  # before any reset: no client exists yet
    assert w.bodies() == []  # nothing to list until the simulation is found
    with pytest.raises(CapabilityError, match="reset the environment first"):
        env.step()  # ...and a step before it is found says so, rather than stepping nothing
    env.reset()
    assert w.bodies() == ["cube", "cube#2"]
    assert w.backend.client == env.bullet_client_id
    assert w.backend.control_dt == pytest.approx(1 / 100)  # the environment's dt, not the physics step
    w.close(failed=False)


def test_a_repeated_body_name_is_listed_once_per_body_and_each_is_its_own_subject():
    env = HeldClientEnv()
    w = rw.attach(env)
    cid = env._pybullet_client._client
    engine_ids = [p.getBodyUniqueId(i, physicsClientId=cid) for i in range(p.getNumBodies(physicsClientId=cid))]
    assert len(w.bodies()) == len(engine_ids) == 2
    env.step()
    expect(w.body("cube#2")).to_be_above(0.0)
    pos = np.array(p.getBasePositionAndOrientation(engine_ids[1], physicsClientId=cid)[0])
    assert np.allclose(w.body("cube#2").position, pos)
    w.close(failed=False)


def test_joints_are_listed_as_the_engine_reports_them():
    cid = p.connect(p.DIRECT)
    kuka = p.loadURDF(KUKA, useFixedBase=True, physicsClientId=cid)
    engine = [
        p.getJointInfo(kuka, j, physicsClientId=cid)[1].decode()
        for j in range(p.getNumJoints(kuka, physicsClientId=cid))
        if p.getJointInfo(kuka, j, physicsClientId=cid)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
    ]
    w = rw.attach(engine="pybullet", client=cid)
    assert list(w.joints()) == engine and len(engine) == 7
    assert set(w.joints().values()) == {"hinge"}
    assert w.joint(engine[0]).value == pytest.approx(0.0)
    w.close(failed=False)


def test_control_period_falls_back_to_the_physics_step_when_the_environment_has_none():
    cid = p.connect(p.DIRECT)
    p.setTimeStep(1 / 200, physicsClientId=cid)
    w = rw.attach(engine="pybullet", client=cid)
    assert w.backend.control_dt == pytest.approx(1 / 200)
    w.close(failed=False)


def test_a_step_taken_through_the_wrapper_or_the_unwrapped_env_is_recorded_once():
    core = Core()
    env = Wrapper(core)
    w = rw.attach(env, settings=rw.Settings(trace="off"))
    env.step()  # the wrapper: its inner step is the unwrapped one, recorded once
    core.step()  # the unwrapped environment, called by a loop that bypasses the wrapper
    assert w.step_count == 2
    w.observe()  # the step above is already recorded: observe adds nothing
    assert w.step_count == 2
    w.close(failed=False)


def test_a_loop_that_steps_the_engine_directly_is_recorded_by_observe_once_per_step():
    cid = p.connect(p.DIRECT)
    _cubes(cid, 1)
    w = rw.attach(engine="pybullet", client=cid, dt=1 / 240, settings=rw.Settings(trace="off"))
    for _ in range(5):
        p.stepSimulation(physicsClientId=cid)
        w.observe()
    assert w.step_count == 5
    assert w.backend.time == pytest.approx(5 / 240)
    w.close(failed=False)


def test_an_environment_made_by_its_registered_id_is_rebuilt_with_the_library_that_made_it():
    gym_like = type("EnvSpec", (), {"__module__": "gym.envs.registration"})()
    gymnasium_like = type("EnvSpec", (), {"__module__": "gymnasium.envs.registration"})()
    assert _make_library(gym_like) == "gym"
    assert _make_library(gymnasium_like) == "gymnasium"


def test_replay_builds_a_gym_0x_environment_with_gym_make(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.ModuleType("gym")
    fake.make = lambda env_id, **kw: calls.append((env_id, kw)) or "built"
    monkeypatch.setitem(sys.modules, "gym", fake)
    tr = type("T", (), {})()
    tr.meta = {
        "env": {
            "actions_recorded": True,
            "spec": {
                "id": "ToyBullet-v0",
                "kwargs": {"render": False},
                "entry_point": "toy_bullet:ToyEnv",
                "library": "gym",
                "reproducible": True,
                "max_episode_steps": None,
                "registered_max_episode_steps": None,
            },
        }
    }
    assert _make_from_spec(tr)() == "built"
    assert calls == [("toy_bullet:ToyBullet-v0", {"render": False})]


def test_a_pybullet_run_with_no_way_to_rebuild_it_is_refused_with_the_factory_flag(tmp_path):
    env = HeldClientEnv()
    w = rw.attach(env, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name="pybullet_any_hand_built")
    env.reset()
    for _ in range(3):
        env.step()
    w.close(failed=False)
    with pytest.raises(ValueError, match="--factory module:func"):
        generate(w.trace_path)
