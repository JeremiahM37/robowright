"""Attaching to any MuJoCo simulation, with no third-party package: inline MJCF and toy environments.

Each test is one case a cold-start trial met in a simulation it had never seen: a free joint, an unnamed body, a model
built only at the first reset (and rebuilt by a later one), a reset that takes no seed, a reset that draws from a global
or environment-owned generator, a bare model/data loop, an environment that has no gymnasium id.
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import mujoco
import numpy as np
import pytest

import robowright as rw
from robowright.codegen import companion, generate
from robowright.errors import CapabilityError
from robowright.replay import replay

SRC = Path(rw.__file__).resolve().parent.parent
TRACE = rw.Settings(trace="on")  # a trace is kept whether or not the run failed, so the test can replay it

XML_FREE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="box" pos="0 0 1">
      <joint name="box_free" type="free"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" mass="0.1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" name="m"/></actuator>
</mujoco>
"""

XML_UNNAMED = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.3">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="tip" pos="0.1 0 0">
        <geom type="sphere" size="0.02" mass="0.1"/>
        <body pos="0 0.1 0">
          <geom type="sphere" size="0.02" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slide"/></actuator>
</mujoco>
"""

XML_SLIDE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="cart" pos="0 0 0.1">
      <joint name="slide" type="slide" axis="1 0 0" damping="0.1"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="slide" gear="1" ctrlrange="-1 1" ctrllimited="true"/></actuator>
</mujoco>
"""


def _model(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)  # a fresh MjData has zero qpos, not the model's start pose
    mujoco.mj_forward(model, data)
    return model, data


class BuildsOnReset:
    """Owns no model until reset builds one; a later reset rebuilds it with another body (safety-gymnasium does this)."""

    def __init__(self):
        self.model = None
        self.data = None
        self.resets = 0

    def reset(self, seed=None, options=None):
        extra = '<body name="extra" pos="0 0 0.2"><geom type="sphere" size="0.01" mass="0.1"/></body>' if self.resets else ""
        z = 1.0 if self.resets == 0 else 0.5
        self.model = mujoco.MjModel.from_xml_string(
            f'<mujoco><option timestep="0.002"/><worldbody><body name="ball" pos="0 0 {z}">'
            '<joint name="ball_slide" type="slide" axis="0 0 1"/><geom type="sphere" size="0.05" mass="1"/></body>'
            f"{extra}</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.resets += 1
        mujoco.mj_forward(self.model, self.data)
        return None, {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return None, 0.0, False, False, {}


class DrawsOnReset:
    """A reset that takes no seed (dm_control style) and draws its start from a generator; ``owned`` makes it an
    environment-owned generator (robohive's self.np_random), otherwise the global numpy one (ogbench's maze.py)."""

    def __init__(self, owned=False):
        self.model, self.data = _model(XML_SLIDE)
        self._np_random = np.random.default_rng() if owned else None  # unseeded: from the OS, as make() leaves it
        self.owned = owned

    def reset(self):
        gen = self._np_random if self.owned else np.random
        self.data.qpos[0] = gen.uniform(-0.2, 0.2)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return None, 0.0, False, False, {}


def _record(env, path, n=30):
    world = rw.attach(env, settings=TRACE)
    env.reset()
    for k in range(n):
        env.step(np.array([np.sin(0.2 * k)]))
    return Path(world.close(trace_path=path))


def test_free_joint_lists_as_free_and_refuses_with_its_body():
    model, data = _model(XML_FREE)
    world = rw.attach(model=model, data=data, settings=rw.Settings(trace="off"))
    assert world.joints() == {"box_free": "free", "hinge": "hinge"}
    with pytest.raises(CapabilityError, match="free joint of body 'box'") as err:
        world.joint("box_free")
    assert "world.body('box')" in str(err.value)
    assert world.joint("hinge").value == 0.0
    assert world.body("box").position[2] == pytest.approx(1.0)


def test_unnamed_bodies_are_counted_listed_and_resolved():
    model, data = _model(XML_UNNAMED)
    world = rw.attach(model=model, data=data, settings=rw.Settings(trace="off"))
    names = world.bodies()
    assert len(names) == model.nbody == 4
    assert {"world", "tip", "body#1", "body#3"} <= set(names)
    mujoco.mj_forward(model, data)
    for i in (1, 3):
        assert np.allclose(world.body(f"body#{i}").position, data.xpos[i])
    with pytest.raises(KeyError):
        world.body("body#99")


def test_model_built_by_reset_binds_at_reset_and_rebinds_on_rebuild():
    env = BuildsOnReset()
    world = rw.attach(env, settings=rw.Settings(trace="off"))  # before any reset: must not raise
    assert world.bodies() == []
    env.reset()
    assert world.body("ball").position[2] == pytest.approx(1.0)
    env.reset()  # the second reset rebuilds the model, with an extra body
    assert world.body("ball").position[2] == pytest.approx(0.5)
    assert world.body("extra").position[2] == pytest.approx(0.2)
    env.model, env.data = None, None  # the simulation is gone: a read must say so, never answer from the old one
    with pytest.raises(CapabilityError, match="gone"):
        _ = world.body("ball").position


def test_environment_that_never_builds_a_model_says_so_at_reset():
    class NoModel:
        def reset(self, seed=None):
            return None, {}

        def step(self, action):
            return None, 0.0, False, False, {}

    env = NoModel()
    rw.attach(env, settings=rw.Settings(trace="off"))
    with pytest.raises(ValueError, match="cannot find a MuJoCo model"):
        env.reset()


def test_reading_after_unobserved_steps_warns_once():
    model, data = _model(XML_SLIDE)
    world = rw.attach(model=model, data=data, settings=rw.Settings(trace="off"))
    for _ in range(3):
        mujoco.mj_step(model, data)  # the caller's loop, without world.observe()
    with pytest.warns(UserWarning, match="without world.observe"):
        _ = world.body("cart").position
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = world.body("cart").position  # the second read says nothing more


@pytest.mark.parametrize("owned", [False, True], ids=["global-numpy", "env-generator"])
def test_replay_of_a_reset_without_seed_restores_what_the_reset_drew_from(tmp_path, owned):
    np.random.seed(7)
    path = _record(DrawsOnReset(owned=owned), tmp_path / "t.zip")
    np.random.seed(999)  # the replaying process starts from another global state
    res = replay(path, make_env=lambda: DrawsOnReset(owned=owned), tol=1e-12)
    assert res.matches, res.summary()


def test_divergence_says_the_step_and_suggests_seeding_a_factory(tmp_path):
    np.random.seed(7)
    path = _record(DrawsOnReset(), tmp_path / "t.zip")

    class Shifted(DrawsOnReset):
        def reset(self):
            super().reset()
            self.data.qpos[0] += 0.01  # a different start: must diverge at step 0

    res = replay(path, make_env=Shifted, tol=1e-9)
    assert not res.matches
    text = res.summary()
    assert "diverged at step 0" in text and "--factory module:func" in text


def test_bare_model_replays_and_its_generated_test_passes(tmp_path):
    model, data = _model(XML_SLIDE)
    world = rw.attach(model=model, data=data, settings=TRACE)
    for k in range(40):
        data.ctrl[:] = [np.sin(0.2 * k)]
        mujoco.mj_step(model, data)
        world.observe()
    path = Path(world.close(trace_path=tmp_path / "bare.zip"))
    res = replay(path, tol=1e-12)
    assert res.matches, res.summary()
    src = generate(path, test_name="test_bare_model")
    f = tmp_path / "test_bare_model.py"
    f.write_text(src)
    name, blob = companion(path, test_name="test_bare_model")  # the model, control and start state, beside the test
    (tmp_path / name).write_bytes(blob)
    assert "NotImplementedError" not in src
    p = subprocess.run(
        [sys.executable, "-m", "pytest", str(f), "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        timeout=600,
    )
    assert p.returncode == 0, p.stdout[-1500:]


def test_environment_without_a_gym_id_is_refused_naming_the_factory(tmp_path):
    path = _record(DrawsOnReset(), tmp_path / "t.zip")
    with pytest.raises(ValueError, match=r"--factory module:func"):
        replay(path)
    with pytest.raises(ValueError, match=r"--factory module:func"):
        generate(path)
