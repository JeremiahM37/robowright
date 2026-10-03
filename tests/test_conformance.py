"""The contract every backend meets for every robot.

Run it for one backend across the whole robot roster with::

    pytest tests/test_conformance.py --rw-backend genesis --rw-robot all

A backend is done when this file passes for every robot it claims to support.
Each test is small and names one property, so a failure says what is wrong
(the kinematics, the servo, the gripper, contacts) rather than "pick failed".
"""

import numpy as np
import pytest

from robowright import expect
from robowright.backends.base import CONTACTS, DETERMINISTIC, GROUND_TRUTH, RENDER, STATE
from robowright.robot import DOWN


@pytest.fixture(autouse=True)
def _arms_only(rw_robot):
    from robowright import robots

    if robots.get(rw_robot).family != "arm":
        pytest.skip("arm contract (legged robots: tests/test_legged.py)")


def test_reports_the_robot_joints(world):
    b = world.backend
    m = b.robot_model
    assert b.joint_names == [*m.arm_joints, "gripper"]
    assert b.qpos().shape == (m.n_arm + 1,) and b.qvel().shape == (m.n_arm + 1,) and b.ctrl().shape == (m.n_arm + 1,)


def test_kinematics_match_the_simulated_hand(world):
    """The simulator's hand pose agrees with robowright's kinematics for random joint angles."""
    r, b = world.robot, world.backend
    rng = np.random.default_rng(0)
    lo, hi = np.maximum(r.kin.lower, -2.5), np.minimum(r.kin.upper, 2.5)
    for _ in range(10):
        q = rng.uniform(lo, hi)
        r.reset_to(q)
        pos, quat = b.hand_pose()
        w, x, y, z = quat
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        assert np.linalg.norm(pos + R @ r.kin.tcp_offset - r.kin.tcp(q)) < 1e-4
        assert np.allclose(b.qpos()[: r.n_arm], q, atol=1e-6)


def test_holds_the_home_pose(world):
    """Servos hold the arm still against gravity: no sag, no drift, no oscillation."""
    world.wait(1.0)
    q, home = world.robot.true_qpos()[: world.robot.n_arm], world.robot.home_q
    assert np.max(np.abs(q - home)) < 0.02
    assert np.max(np.abs(world.backend.qvel()[: world.robot.n_arm])) < 0.05
    assert np.linalg.norm(world.robot.tcp.position - world.robot.model.home) < 0.005


def test_tracks_a_joint_move(world):
    r = world.robot
    target = r.home_q + np.where(np.arange(r.n_arm) % 2 == 0, 0.25, -0.15)
    target = np.clip(target, r.kin.lower + 0.05, r.kin.upper - 0.05)
    r.arm.move_joints(target)
    assert np.max(np.abs(r.true_qpos()[: r.n_arm] - target)) < 0.03


def test_moves_the_tool_to_a_point(world):
    world.robot.arm.move_to((0.22, 0.04, 0.06))
    expect(world.robot.tcp).to_be_near((0.22, 0.04, 0.06), tol=0.01)
    T = world.robot.kin.fk(world.robot.true_qpos()[: world.robot.n_arm])
    assert T[:3, :3] @ world.robot.kin.tool_axis @ DOWN > 0.99


def test_gripper_opens_and_closes(world):
    g = world.robot.gripper
    g.close()
    expect(g).to_be_closed(max_opening=0.1)
    g.open()
    expect(g).to_be_open(min_opening=0.9)
    g.open(0.5)
    assert abs(g.opening - 0.5) < 0.08


def test_objects_rest_on_the_floor(world):
    if GROUND_TRUTH not in world.backend.capabilities:
        pytest.skip("no ground truth")
    world.wait(0.5)
    cube = world.scene["cube"]
    assert cube.position[2] == pytest.approx(0.0125, abs=0.002)
    expect(cube).to_be_at_rest()
    if CONTACTS in world.backend.capabilities:
        pairs = {frozenset((c.a, c.b)) for c in world.backend.contacts()}
        assert frozenset(("cube", "floor")) in pairs


def test_grasp_is_seen_by_both_fingers(world):
    cube = world.scene["cube"]
    world.robot.pick(cube)
    expect(world.robot.gripper).to_be_holding(cube)
    assert cube.position[2] > 0.04  # lifted


def test_pick_and_place(world):
    cube, bin_ = world.scene["cube"], world.scene["bin"]
    world.robot.pick(cube)
    world.robot.place(on=bin_)
    expect(cube).to_be_inside(bin_)
    expect(cube).to_be_at_rest()


def test_no_arm_collisions_during_a_pick(world):
    if CONTACTS not in world.backend.capabilities:
        pytest.skip("no contacts")
    expect(world.robot).always.to_have_no_collisions()
    world.robot.pick(world.scene["cube"])


def test_renders_the_scene(world):
    if RENDER not in world.backend.capabilities:
        pytest.skip("no rendering")
    img = world.backend.render("front", 160, 120)
    assert img.shape == (120, 160, 3) and img.dtype == np.uint8 and img.std() > 5


def test_state_round_trip(world):
    b = world.backend
    if STATE not in b.capabilities:
        pytest.skip("no state save/restore")
    world.robot.arm.move_to((0.22, 0.0, 0.06))
    s = b.get_state()
    q0 = b.qpos()
    world.wait(0.3)
    b.set_state(s)
    assert np.allclose(b.qpos(), q0, atol=1e-9)


def test_deterministic(world, rw_backend, rw_robot):
    if DETERMINISTIC not in world.backend.capabilities:
        pytest.skip("not deterministic")
    import robowright as rw

    def run():
        w = rw.launch(robot=rw_robot, backend=rw_backend, settings=rw.Settings(trace="off"))
        w.robot.reset_to()
        w.robot.pick(w.scene["cube"])
        out = np.concatenate([w.backend.qpos(), w.scene["cube"].position])
        w.close()
        return out

    assert np.array_equal(run(), run())
