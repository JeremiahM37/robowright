"""robowright attached to a simulation it did not build: an environment with its own model, data and loop.

The environment here is a stand-in written for the test (a model and data with reset and step),
because gymnasium is not in the project's venv. The real published environment is tested in
examples/published_policy/test_act_aloha.py.
"""

import mujoco
import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.attached import AttachedRenderer
from robowright.errors import CapabilityError, InvariantViolation
from robowright.render import jpeg_frames
from robowright.trace import Trace

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="table" pos="0 0 0.2">
      <geom name="table_top" type="box" size="0.3 0.3 0.02"/>
    </body>
    <body name="box" pos="0 0 0.3">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.02 0.02 0.02" mass="0.05"/>
      <site name="box_site" pos="0 0 0.02"/>
    </body>
    <body name="slider" pos="0.5 0 0.1">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="pad" type="sphere" size="0.03"/>
    </body>
    <camera name="cam" pos="0 -1 0.5" xyaxes="1 0 0 0 0.5 1"/>
  </worldbody>
  <actuator>
    <position joint="slide" kp="100"/>
  </actuator>
</mujoco>
"""
CONTROL_STEPS = 10  # physics steps per control step, as a dm_control environment does


class ToyEnv:
    """Owns its model and data and has a gymnasium-shaped reset and step, which is all attach needs."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.resets = 0

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.resets += 1
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(CONTROL_STEPS):
            mujoco.mj_step(self.model, self.data)
        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        return {"qpos": self.data.qpos.copy()}


@pytest.fixture
def toy(tmp_path):
    worlds = []

    def make(advance=None, **settings):
        env = ToyEnv()
        settings.setdefault("trace_dir", str(tmp_path))
        w = rw.attach(env, advance=advance, settings=rw.Settings(**settings))
        worlds.append(w)
        return env, w

    yield make
    for w in worlds:
        if w.status == "running":
            w.close(failed=False)


def loop(env, n, action=None):
    """Their step loop: the environment is stepped by the test, as an RL loop does."""
    a = np.zeros(1) if action is None else action
    for _ in range(n):
        env.step(a)


def test_bodies_sites_and_geoms_are_subjects_with_ground_truth(toy):
    env, w = toy()
    env.reset(seed=0)
    box = w.body("box")
    assert "box" in w.bodies() and "table" in w.bodies()
    assert w.object_names == ["box"]  # the free body is an object; the table is not
    assert np.allclose(box.position, env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "box")])
    assert np.allclose(w.body("box_site").position, env.data.site_xpos[0])
    geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
    assert np.allclose(w.body("box_geom").position, env.data.geom_xpos[geom])
    assert np.isclose(np.linalg.norm(box.quaternion), 1.0)
    with pytest.raises(KeyError, match="no body, site or geom named 'nope'"):
        w.body("nope")
    assert w.model_changes == []


def test_invariant_is_checked_on_every_step_of_their_loop(toy):
    env, w = toy()
    env.reset(seed=0)
    expect(w.body("box")).always.to_be_above(0.25)  # the box falls from 0.30 to rest on the table near 0.24
    with pytest.raises(InvariantViolation, match="invariant violated"):
        loop(env, 200)
    assert w.step_count < 200  # it stopped at the first step that broke the invariant


def test_auto_wait_advances_their_loop_through_advance(toy):
    env, w = toy(advance=lambda: env.step(np.zeros(1)))
    env.reset(seed=0)
    t0 = w.time
    # the box starts at 0.30 and settles near 0.24: this is false now and true only after time passes
    expect(w.body("box")).to_be_near((0.0, 0.0, 0.24), tol=0.01, timeout=3.0)
    assert w.time > t0
    t1 = w.time
    with pytest.raises(Exception, match="failed after"):
        expect(w.body("box")).to_be_above(0.9, timeout=0.2)
    assert w.time - t1 == pytest.approx(0.2, abs=w.dt)


def test_without_advance_nothing_waits(toy):
    env, w = toy()
    env.reset(seed=0)
    t0, n0 = w.time, w.step_count
    with pytest.raises(Exception, match="failed"):
        expect(w.body("box")).to_be_above(0.9, timeout=5.0)  # checked once: no time passes
    assert (w.time, w.step_count) == (t0, n0)
    with pytest.raises(CapabilityError, match="time advances when your loop steps"):
        w.step(1)


def test_contacts_are_ground_truth_by_body_and_geom_name(toy):
    env, w = toy()
    env.reset(seed=0)
    loop(env, 300)  # the box lands on the table and settles
    box = w.body("box")
    assert set(box.contacts()) == {"table", "table_top"}  # by body and by geom name
    expect(box).to_be_touching("table_top")
    assert np.abs(box.velocity).max() < 1e-3
    expect(w.body("pad")).not_.to_be_touching("box")


def test_move_object_and_push_are_faults_recorded_in_the_trace(toy):
    env, w = toy(advance=lambda: env.step(np.zeros(1)))
    env.reset(seed=0)
    w.move_object("box", [0.3, 0.0, 0.5])
    assert np.allclose(w.body("box").position, [0.3, 0.0, 0.5], atol=1e-9)
    assert np.allclose(w.body("box").velocity, 0.0)
    w.push("box", [0.0, 0.0, 0.5], duration=0.02)  # applied for the one control step it covers
    w.step(1)
    m = env.model.body_mass[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "box")]
    expected = (0.5 / m - 9.81) * w.dt  # the push, less gravity, over one control period
    assert w.body("box").velocity[2] == pytest.approx(expected, abs=1e-9)  # exact, as MuJoCo's own qvel is
    assert np.all(env.data.xfrc_applied == 0)  # restored after the step: nothing of ours is left in their data
    names = [(e.type, e.name) for e in w.trace.events]
    assert ("edit", "move_object") in names and ("fault", "Push") in names
    with pytest.raises(ValueError, match="push moves free bodies"):
        w.push("table", [1, 0, 0])


def test_robowright_changes_nothing_in_their_model_or_data(toy):
    env, w = toy()
    env.reset(seed=0)
    before = (env.model.body_mass.copy(), env.model.opt.timestep, env.model.geom_friction.copy())
    loop(env, 50, action=np.array([0.2]))
    after = (env.model.body_mass, env.model.opt.timestep, env.model.geom_friction)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert w.model_changes == []
    assert env.resets == 1  # reset was observed, not repeated


def test_observe_records_the_steps_their_own_code_takes(toy):
    env, w = toy()
    env.reset(seed=0)
    for _ in range(5):
        mujoco.mj_step(env.model, env.data)  # below env.step: not seen by itself
        w.observe()
    assert w.step_count == 5
    assert len(w.trace.t) == 6  # the start and one row per observed step


def test_bare_model_with_a_stepper(tmp_path):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    w = rw.attach(model=m, data=d, step=lambda: mujoco.mj_step(m, d), settings=rw.Settings(trace_dir=str(tmp_path)))
    try:
        w.step(20)
        assert w.step_count == 20
        assert w.time == pytest.approx(20 * m.opt.timestep)
        assert w.dt == pytest.approx(m.opt.timestep)
    finally:
        w.close(failed=False)


def test_trace_saved_with_model_and_redrawn_from_state(toy):
    env, w = toy(advance=lambda: env.step(np.zeros(1)))
    env.reset(seed=0)
    w.step(40)
    path = w.close(failed=False)
    tr = Trace(path)
    assert tr.meta["attached"]["model"] == "model.mjb"
    assert tr.meta["steps"] == 40 and len(tr) == 41
    assert tr.arrays["qpos"].shape == (41, env.model.nq)
    frames = jpeg_frames(tr, every=10, size=(64, 48), cameras=["cam"])
    assert sorted(frames["cam"]) == [0, 10, 20, 30, 40]
    r = AttachedRenderer(tr)
    try:
        img = r.frame(40, "cam", (64, 48))
        assert img.shape == (48, 64, 3) and img.any()
    finally:
        r.close()
    assert "box" in tr.summary()
