"""robowright attached to a simulation it did not build: an environment with its own model, data and loop.

The environment here is a stand-in written for the test (a model and data with reset and step),
because gymnasium is not in the project's venv. The real published environment is tested in
examples/published_policy/test_act_aloha.py.
"""

import zipfile

import mujoco
import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.attached import AttachedRenderer, _physics_of
from robowright.backends.attached_engine import AttachedEngine
from robowright.errors import CapabilityError, ExpectationError, InvariantViolation
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


def test_run_drives_their_step_until_a_condition_then_gives_back_the_loop(toy):
    env, w = toy()
    env.reset(seed=0)
    ok = w.run(lambda: env.step(np.zeros(1)), until=lambda: w.body("box").position[2] < 0.25, timeout=3.0)
    assert ok and 0 < w.time < 3.0
    assert w.step_count == round(w.time / w.dt)
    with pytest.raises(CapabilityError):  # the run was for that call only: nothing waits afterwards
        w.step(1)


def test_joint_by_name_is_a_scalar_subject_for_to_be_between(toy):
    env, w = toy()
    env.reset(seed=0)
    slide = env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "slide")]
    env.data.qpos[slide] = 0.4  # set directly: the test reads the state, it does not step the model
    mujoco.mj_forward(env.model, env.data)
    expect(w.joint("slide")).to_be_between(0.3, 0.5)
    with pytest.raises(ExpectationError, match="slide = 0.400"):
        expect(w.joint("slide")).to_be_between(-1.0, 0.0)
    assert w.joint("slide").value == pytest.approx(0.4)
    with pytest.raises(KeyError, match="joints: box_free, slide"):
        w.joint("nope")
    with pytest.raises(CapabilityError, match="free joint of body 'box'"):
        w.joint("box_free")  # a free joint has seven values, not one


def test_redraw_of_a_model_with_no_camera_uses_the_free_camera():
    """Shadow-hand style models have no cameras; the redraw then falls back to MuJoCo's free camera."""
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1" pos="0 0 0.2"/></worldbody></mujoco>')
    r = AttachedRenderer.__new__(AttachedRenderer)  # a renderer over a model and states, without a saved trace file
    r.model, r.data = model, mujoco.MjData(model)
    r.trace = type("Trace", (), {"arrays": {"qpos": np.zeros((1, model.nq))}})()
    r.steps, r._renderer, r._renderer_size = 1, None, None
    try:
        assert r.cameras == []
        img = r.frame(0, size=(64, 48))
        assert img.shape == (48, 64, 3) and img.any()
    finally:
        r.close()


def test_set_joint_is_a_recorded_fault_on_a_slide_joint(toy):
    env, w = toy()
    env.reset(seed=0)
    w.set_joint("slide", 0.2)
    assert w.joint("slide").value == pytest.approx(0.2)
    with pytest.raises(ValueError, match="not a hinge or slide"):
        w.set_joint("box_free", 0.0)
    path = w.close(failed=False)
    edits = [e for e in Trace(path).events if e.get("type") == "edit"]
    assert [(e["name"], e["args"]) for e in edits] == [("set_joint", {"joint": "slide", "value": 0.2})]


def test_model_and_data_found_behind_a_binding_wrapper_sim_attribute():
    """robosuite keeps the MuJoCo structs behind env.sim.model._model and env.sim.data._data."""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    model_wrapper = type("ModelWrapper", (), {"_model": model})()  # the binding's own attribute name for its struct
    data_wrapper = type("DataWrapper", (), {"_data": data})()
    sim = type("FakeSim", (), {"model": model_wrapper, "data": data_wrapper})()
    env = type("FakeEnv", (), {"sim": sim})()
    found_model, found_data = _physics_of(env)
    assert found_model is model and found_data is data
    with pytest.raises(ValueError, match="cannot find a MuJoCo model"):
        _physics_of(type("Bare", (), {})())


def test_a_reset_that_rebuilds_the_simulation_is_followed(tmp_path):
    """robosuite's reset builds a new MjModel and MjData. Subjects must read the new ones, not the ones at attach time."""

    class RebuildingToy(ToyEnv):
        def reset(self, seed=None, options=None):
            self.model = mujoco.MjModel.from_xml_string(XML)  # a new simulation object each reset, as robosuite does
            self.data = mujoco.MjData(self.model)
            return super().reset(seed=seed, options=options)

    env = RebuildingToy()
    w = rw.attach(env, settings=rw.Settings(trace_dir=str(tmp_path)))
    env.reset(seed=0)
    first = w.model
    env.reset(seed=1)
    assert w.model is env.model and w.model is not first
    assert w.body("box").position[2] == pytest.approx(env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "box"), 2])
    w.close(failed=False)


class FakeEngine(AttachedEngine):
    """The smallest engine that meets the contract: one free puck, one hinge, a clock; no contacts, no render.

    It is written for this test, so it checks the core against the contract, not against any real simulator.
    """

    engine = "fake"

    def __init__(self):
        super().__init__(seed=0, dt=0.01)
        self.free_bodies = ["puck"]
        self.joint_names = ["hinge"]
        self.puck = np.zeros(3)
        self.angle = 0.0
        self.steps = 0

    def names(self, kind="body"):
        return ["puck", "frame"] if kind == "body" else []

    def resolve(self, name):
        return {"puck": ("body", 0), "frame": ("body", 1)}.get(name)

    def label(self, handle):
        return ["puck", "frame"][handle[1]]

    def pose(self, handle):
        pos = self.puck if handle[1] == 0 else np.array([1.0, 0.0, 0.0])
        return pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def velocity(self, handle):
        return np.zeros(6)

    def contacts_of(self, handle):
        return []

    def contact_pairs(self):
        return []

    def joint_qadr(self, name):
        if name != "hinge":
            raise KeyError(f"no joint named {name!r}; joints: hinge")
        return 0

    def joint_value(self, name):
        self.joint_qadr(name)
        return self.angle

    def qpos(self):
        return np.array([self.angle])

    def set_object_pose(self, name, pos, quat=None):
        self.puck = np.asarray(pos, float)

    def apply_force(self, name, force):
        pass

    @property
    def time(self):
        return self.steps * self._dt

    def note_step(self):
        self.steps += 1
        self.angle += 0.1  # the environment's own motion, which the test drives


def test_an_engine_meeting_the_contract_is_attached_without_mujoco(tmp_path):
    from robowright.attached import AttachedWorld

    eng = FakeEngine()
    env_steps = []

    def advance():
        env_steps.append(1)
        eng.note_step()

    w = AttachedWorld(eng, advance, name="fake", seed=0, settings=rw.Settings(trace_dir=str(tmp_path)))
    assert w.bodies() == ["puck", "frame"]
    assert w.body("puck").position.tolist() == [0.0, 0.0, 0.0]
    assert w.joint("hinge").value == pytest.approx(0.0)
    w.step(3)
    assert len(env_steps) == 3 and w.joint("hinge").value == pytest.approx(0.3)
    assert w.backend.time == pytest.approx(0.03)
    with pytest.raises(KeyError, match="no joint named 'nope'"):
        w.joint("nope")
    with pytest.raises(CapabilityError, match="the fake engine has no MuJoCo model"):
        _ = w.model  # a non-MuJoCo engine has no MjModel: the core says so rather than inventing one
    path = w.close(failed=False)
    tr = Trace(path)
    assert tr.meta["attached"] == {"engine": "fake"}
    assert "model.mjb" not in set(zipfile.ZipFile(path).namelist())


def test_redraw_of_a_non_mujoco_trace_says_why(tmp_path):
    from robowright.attached import AttachedWorld
    from robowright.render import renderer_for

    eng = FakeEngine()
    w = AttachedWorld(eng, lambda: eng.note_step(), name="fake", seed=0, settings=rw.Settings(trace_dir=str(tmp_path)))
    w.step(2)
    tr = Trace(w.close(failed=False))
    with pytest.raises(CapabilityError, match="frame redraw is not available for the fake engine"):
        renderer_for(tr)


def test_an_engine_without_render_or_state_says_so_in_one_line():
    eng = FakeEngine()
    with pytest.raises(CapabilityError, match="frame redraw is not available for the fake engine"):
        eng.render("cam", 64, 48)
    with pytest.raises(CapabilityError, match="saving and restoring state is not available for the fake engine"):
        eng.get_state()
