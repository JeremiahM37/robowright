"""The interactive session behind the MCP server, and the server itself."""

import asyncio

import numpy as np
import pytest

from robowright.backends.base import RENDER
from robowright.session import Session


@pytest.fixture
def session(tmp_path):
    s = Session(trace_dir=tmp_path)
    yield s
    if s.world is not None:
        s.close()


def _final(world):
    return np.concatenate([world.backend.qpos(), *(world.backend.object_pose(n)[0] for n in world.object_names)])


def test_snapshot_names_every_object_with_its_state(session):
    snap = session.launch(robot="so101")
    assert "SO-101" in snap and "status=running" in snap
    assert "- cube: red box 25x25x25 mm" in snap and "touching floor" in snap
    assert "- bin: blue bin" in snap
    assert "cameras: front, top, side" in snap


def test_snapshot_of_a_legged_robot(session):
    snap = session.launch(robot="go2")
    assert "robot (legged):" in snap and "height" in snap and "tilt" in snap


def test_actions_assertions_and_a_reproducing_test(session, tmp_path):
    session.launch(
        robot="so101",
        objects=[
            {"name": "can", "kind": "cylinder", "size": [0.015, 0.02], "pos": [0.22, -0.06], "color": "green"},
            {"name": "bin", "kind": "bin", "size": [0.05, 0.05, 0.02], "pos": [0.2, 0.12, 0.0], "color": "blue", "mass": 0.0},
        ],
    )
    # A planning failure moves nothing and is left out of the test.
    assert session.move_to([2.0, 0.0, 0.3]).startswith("FAILED: UnreachableError")
    # A failed check lets time pass, so the test repeats it inside pytest.raises.
    assert session.expect("can", "to_be_inside", {"container": "bin"}, timeout=0.3).startswith("FAIL")
    assert "holding can" in session.pick("can")
    session.place("bin")
    assert session.expect("can", "to_be_inside", {"container": "bin"}).startswith("PASS")
    assert "inside bin" in session.snapshot()

    code = session.generate_test("test_regen")
    assert "move_to" not in code
    assert "with pytest.raises(ExpectationError):" in code and "timeout=0.3" in code
    assert "robot.reset_to()" in code

    expected = _final(session.world)
    namespace = {}
    import robowright as rw

    captured = {}
    real_launch = rw.launch

    def spy(*a, **k):
        w = real_launch(*a, **k)
        captured["world"] = w
        close = w.close

        def close_keep(*ca, **ck):
            captured["final"] = _final(w)
            return close(*ca, **ck)

        w.close = close_keep
        return w

    rw.launch = spy
    try:
        exec(compile(code, "regen", "exec"), namespace)
        namespace["test_regen"]()
    finally:
        rw.launch = real_launch
    assert np.array_equal(captured["final"], expected)


def test_a_tilted_pick_is_saved_with_its_direction(session):
    session.launch(robot="ur5e")
    assert "holding cube" in session.pick("cube", approach=[1, 0, -1])
    session.place("bin")
    assert session.expect("cube", "to_be_inside", {"container": "bin"}).startswith("PASS")
    code = session.generate_test("test_tilted")
    assert "approach=[1.0, 0.0, -1.0]" in code
    namespace = {}
    exec(compile(code, "tilted", "exec"), namespace)
    namespace["test_tilted"]()


def test_failed_actions_explain_and_keep_the_session(session):
    session.launch(robot="panda")
    with pytest.raises(KeyError, match="scene has"):
        session.pick("banana")
    with pytest.raises(Exception, match="is an arm"):
        session.stand()
    assert session.world.status == "running"


def test_move_joints_by_name(session):
    session.launch(robot="so101")
    name = session.world.backend.joint_names[0]
    session.move_joints({name: 0.3})
    assert abs(session.world.robot.qpos()[0] - 0.3) < 0.02
    with pytest.raises(Exception, match="unknown joint"):
        session.move_joints({"elbow_of_doom": 1.0})


def test_screenshot_is_a_png(session):
    session.launch(robot="so101")
    if RENDER not in session.world.backend.capabilities:
        pytest.skip("no rendering")
    try:
        png = session.screenshot("top", 160, 120)
    except Exception as e:  # no GL on this machine
        pytest.skip(f"rendering unavailable: {e}")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_mcp_server_exposes_the_session():
    pytest.importorskip("mcp")
    try:
        from mcp import Client
    except ImportError:
        pytest.skip("in-process client needs mcp >= 2")
    from robowright.mcp_server import build_server

    async def go():
        async with Client(build_server()) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"robot_launch", "robot_snapshot", "robot_pick", "robot_expect", "robot_generate_test"} <= names
            r = await c.call_tool("robot_launch", {"robot": "so101"})
            assert "cube" in r.content[0].text
            r = await c.call_tool("robot_pick", {"object": "banana"})
            assert r.is_error and "scene has" in r.content[0].text  # the reason reaches the agent
            await c.call_tool("robot_close", {})

    asyncio.run(go())


def test_legged_push_and_checks_on_the_robot(session):
    session.launch(robot="go2")
    snap = session.push("robot", [0.0, 60.0, 0.0], duration=0.1)
    assert "t=0.100s" in snap
    session.wait(1.0)
    assert session.expect("robot", "to_be_upright", {"tol_deg": 15}).startswith("PASS")  # the robot's pose is its base's
    code = session.generate_test("test_push")
    assert "world.wait(0.1)" in code and "expect(robot.base).to_be_upright(tol_deg=15)" in code
