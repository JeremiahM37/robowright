"""The MCP server on a simulation robowright did not build: sim_attach, the tools that work on it, and the
reasons the rest say they cannot.

gymnasium is not in the project's venv, so the environment is a gymnasium-shaped MuJoCo stand-in (reset,
step, action_space, close) that these tests load through ``entry_point``, the same way test_attach.py builds
its toy. The ``env_id`` path is tested against a fake ``gymnasium`` module: that checks robowright's own
lookup, error and hint code, not the library.
"""

import asyncio
import sys
import types

import mujoco
import numpy as np
import pytest

from robowright.session import Session

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from robowright.mcp_server import build_server  # noqa: E402

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
HORIZON = 20  # control steps before the episode ends


class ToyBox:
    """The shape, dtype and sample() that sim_env_step reads from a gymnasium Box."""

    shape = (1,)
    dtype = np.float64

    def sample(self):
        return np.random.default_rng(0).uniform(-1, 1, size=self.shape)


class ToyEnv:
    action_space = ToyBox()

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.steps = 0
        self.closed = False

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return {"qpos": self.data.qpos.copy()}, {}

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(CONTROL_STEPS):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        return {"qpos": self.data.qpos.copy()}, 1.0, self.steps >= HORIZON, False, {}

    def close(self):
        self.closed = True


class NoPhysicsEnv:
    """Has a reset and a step, but no MuJoCo model, SAPIEN scene or PyBullet client to attach to."""

    def reset(self, seed=None, options=None):
        return None, {}

    def step(self, action):
        return None, 0.0, False, False, {}


ENV_REF = f"{__name__}:make_toy"  # the test module is importable under its own name


def make_toy(**kwargs):
    return ToyEnv()


def make_no_physics(**kwargs):
    return NoPhysicsEnv()


def drive(session, scenario):
    """Run ``scenario(client)`` against a server on ``session``, as an MCP client would."""

    async def go():
        async with Client(build_server(session)) as c:
            await scenario(c)

    try:
        asyncio.run(go())
    finally:
        if session.world is not None:
            session.close()


async def call(c, tool, args=None):
    """(text, is_error) of one tool call."""
    r = await c.call_tool(tool, args or {})
    return r.content[0].text, bool(r.is_error)


@pytest.fixture
def session(tmp_path):
    return Session(trace_dir=tmp_path)


def test_the_attach_tools_are_listed_beside_the_robot_tools(session):
    async def scenario(c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"sim_attach", "sim_env_step", "sim_subjects", "robot_snapshot", "robot_expect", "robot_pick"} <= names

    drive(session, scenario)


def test_attach_by_entry_point_reports_the_free_bodies(session):
    async def scenario(c):
        text, err = await call(c, "sim_attach", {"entry_point": ENV_REF, "kwargs": {}, "seed": 0})
        assert not err, text
        assert "attached ToyEnv (mujoco)" in text and "t=0.000s" in text
        assert "- box: at [0.000, 0.000, 0.300]" in text  # the free body, by name, where the environment's reset put it
        assert "table" not in text.split("objects")[1]  # a static body is not an object

    drive(session, scenario)
    assert session.env is None  # close() gave the environment back


def test_sim_subjects_names_what_the_tools_take(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        text, err = await call(c, "sim_subjects")
        assert not err
        bodies = text.splitlines()[0]
        assert "box*" in bodies and "table" in bodies and "slider" in bodies  # * marks the free body
        assert "box_site" in text.splitlines()[1]  # sites on their own line
        assert "slide" in text.splitlines()[2]  # joints on their own line
        assert "cam" in text.splitlines()[3]  # cameras on their own line

    drive(session, scenario)


def test_sim_env_step_steps_their_loop_and_reports_what_moved(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        text, err = await call(c, "sim_env_step", {"steps": 30})
        assert not err, text
        assert "stepped 20 of 30: reward 20, episode terminated" in text  # the horizon ends it
        assert "moved: box" in text  # the box fell under gravity; the table did not move
        assert "table" not in text.split("moved:")[1].splitlines()[0]
        assert "the episode has ended" in text
        assert "t=0.400s" in text  # 20 control steps of 10 physics steps at 0.002 s
        assert "steps taken: 20" in text

    drive(session, scenario)


def test_sim_env_step_takes_an_action_of_the_right_shape(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        text, err = await call(c, "sim_env_step", {"action": [0.5, 0.5]})
        assert err and "action has shape (2,); this environment's action space is (1,)" in text
        text, err = await call(c, "sim_env_step", {"action": [0.5], "sample": True})
        assert err and "pass action or sample=true, not both" in text
        text, err = await call(c, "sim_env_step", {"action": [0.5], "steps": 5})
        assert not err and "stepped 5 of 5: reward 5, episode continues" in text

    drive(session, scenario)


def test_expect_checks_attached_bodies_and_joints(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        await call(c, "sim_env_step", {"action": [0.5], "steps": 20})  # the box has fallen onto the table by then
        text, _ = await call(c, "robot_expect", {"subject": "box", "matcher": "to_be_above", "args": {"target": "table"}})
        assert text.startswith("PASS"), text  # the box rests on the table, which is below it
        text, _ = await call(c, "robot_expect", {"subject": "slide", "matcher": "to_be_between", "args": {"lo": -1, "hi": 1}})
        assert text.startswith("PASS"), text
        text, _ = await call(c, "robot_expect", {"subject": "slide", "matcher": "to_be_between", "args": {"lo": 2, "hi": 3}})
        assert text.startswith("FAIL") and "want 2.000..3.000" in text, text
        text, err = await call(c, "robot_expect", {"subject": "nothing_here", "matcher": "to_be_above", "args": {"target": "table"}})
        assert err and "no body, site or geom named 'nothing_here'" in text

    drive(session, scenario)


def test_the_robot_tools_say_why_they_cannot_run_on_an_attached_simulation(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        for tool, args, why in [
            ("robot_pick", {"object": "box"}, "pick is not available on an attached simulation because"),
            ("robot_run_policy", {"policy": "x:y"}, "run_policy is not available on an attached simulation"),
            ("robot_wait", {"seconds": 1.0}, "wait is not available on an attached simulation"),
            ("robot_watch", {}, "watch is not available on an attached simulation"),
            ("robot_generate_test", {}, "generate_test is not available on an attached simulation"),
        ]:
            text, err = await call(c, tool, args)
            assert err and why in text, (tool, text)
            assert "Traceback" not in text and "AttributeError" not in text, (tool, text)
        text, err = await call(c, "robot_snapshot")  # but the snapshot works
        assert not err and "steps taken: 0" in text

    drive(session, scenario)


def test_push_and_move_object_act_on_the_attached_bodies(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        text, err = await call(c, "robot_move_object", {"object": "box", "position": [0.1, 0.0, 0.5]})
        assert not err and "- box: at [0.100, 0.000, 0.500]" in text, text
        text, err = await call(c, "robot_push", {"target": "box", "force": [0.0, 0.0, 1.0], "duration": 0.1})
        assert not err and "queued a push of [0.0, 0.0, 1.0] N on box" in text, text
        text, err = await call(c, "robot_push", {"target": "table", "force": [0.0, 0.0, 1.0]})
        assert err and "push moves free bodies; 'table' is not one" in text, text
        text, err = await call(c, "robot_move_object", {"object": "table", "position": [0.0, 0.0, 0.0]})
        assert "FAILED" in text and "is not a free body" in text, text  # a static body is not moved: the engine says so

    drive(session, scenario)


def test_screenshot_draws_from_the_engine_cameras(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        r = await c.call_tool("robot_screenshot", {"camera": "front"})
        assert r.is_error and "its cameras: cam" in r.content[0].text, r.content[0].text
        try:
            r = await c.call_tool("robot_screenshot", {"camera": "cam", "width": 160, "height": 120})
        except Exception as e:  # no GL on this machine
            pytest.skip(f"rendering unavailable: {e}")
        if r.is_error:
            pytest.skip(f"rendering unavailable: {r.content[0].text}")
        assert r.content[0].type == "image"

    drive(session, scenario)


def test_robot_launch_and_close_give_the_environment_back(session):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        env = session.env
        text, err = await call(c, "robot_launch", {"robot": "so101"})
        assert not err and "SO-101" in text
        assert env.closed and session.env is None
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        text, err = await call(c, "robot_close", {})
        assert not err and "trace saved to" in text
        assert session.env is None

    drive(session, scenario)


def test_the_attached_session_saves_a_trace(session, tmp_path):
    async def scenario(c):
        await call(c, "sim_attach", {"entry_point": ENV_REF})
        await call(c, "sim_env_step", {"steps": 3})
        path = tmp_path / "attached.zip"
        text, err = await call(c, "robot_save_trace", {"path": str(path)})
        assert not err and text == str(path)
        assert path.exists() and path.read_bytes()[:2] == b"PK"  # a zip, as any trace is

    drive(session, scenario)


def test_an_environment_robowright_cannot_read_says_which_engines_it_reads(session):
    async def scenario(c):
        text, err = await call(c, "sim_attach", {"entry_point": f"{__name__}:make_no_physics"})
        assert err and "cannot attach to NoPhysicsEnv" in text and "reads MuJoCo" in text, text
        text, err = await call(c, "sim_attach", {"entry_point": "no_such_module_xyz:make"})
        assert err and "cannot import entry_point" in text, text
        text, err = await call(c, "sim_attach", {"entry_point": "not-a-callable"})
        assert err and "entry_point is 'module:callable'" in text, text
        text, err = await call(c, "sim_attach", {})
        assert err and "give env_id" in text, text

    drive(session, scenario)
    assert session.env is None


def test_env_id_without_gymnasium_says_so(session, monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", None)  # makes `import gymnasium` fail, as on a machine without it

    async def scenario(c):
        text, err = await call(c, "sim_attach", {"env_id": "gym_aloha/AlohaTransferCube-v0"})
        assert err and "gymnasium is not installed here" in text and "entry_point" in text, text

    drive(session, scenario)


class _NameNotFound(Exception):
    pass


def _fake_gymnasium(monkeypatch):
    """A stand-in gymnasium: a registry of ids and a make() that knows one of them."""

    def make(env_id, **kwargs):
        if env_id != "Toy-v0":
            raise _NameNotFound(f"Environment {env_id} doesn't exist.")
        return ToyEnv()

    fake = types.ModuleType("gymnasium")
    fake.make = make
    fake.registry = {"Toy-v0": None, "gym_aloha/AlohaTransferCube-v0": None, "CartPole-v1": None}
    monkeypatch.setitem(sys.modules, "gymnasium", fake)


def test_env_id_makes_the_registered_environment(session, monkeypatch):
    _fake_gymnasium(monkeypatch)

    async def scenario(c):
        text, err = await call(c, "sim_attach", {"env_id": "Toy-v0", "seed": 1})
        assert not err and "attached ToyEnv (mujoco)" in text, text

    drive(session, scenario)


def test_an_unknown_env_id_lists_the_registered_ids_that_look_like_it(session, monkeypatch):
    _fake_gymnasium(monkeypatch)

    async def scenario(c):
        text, err = await call(c, "sim_attach", {"env_id": "gym_aloha/AlohaTransferCubeX-v0"})
        assert err and "cannot make 'gym_aloha/AlohaTransferCubeX-v0'" in text, text
        assert "_NameNotFound" in text
        assert "registered ids that look like it: gym_aloha/AlohaTransferCube-v0" in text, text

    drive(session, scenario)


def test_suite_note_names_what_registered_and_what_failed():
    from robowright.session import _suite_note

    assert _suite_note(["gymnasium_robotics"], {"panda_gym": "ImportError: no module named pybullet"}) == (
        "packages registered: gymnasium_robotics; failed to import: panda_gym (ImportError: no module named pybullet)"
    )
    assert "none (no known environment package is installed)" in _suite_note([], {})
