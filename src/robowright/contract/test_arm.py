"""The contract every backend meets for every robot arm.

Run it for one backend across the whole robot roster with::

    robowright check --backend genesis --robot all

(in robowright's own repository, ``pytest tests/test_conformance.py`` runs the same tests).
A backend is done when this file passes for every robot it claims to support, and a robot
when it passes on every engine.
Each test is small and names one property, so a failure says what is wrong
(the kinematics, the servo, the gripper, contacts) rather than "pick failed".
"""

import numpy as np
import pytest

from robowright import expect
from robowright.backends.base import CONTACTS, DETERMINISTIC, GROUND_TRUTH, RENDER, STATE, Backend
from robowright.robot import DOWN


def _sees(world, *objects):
    """Skip unless the world knows where ``objects`` are: an engine's ground truth, or on
    hardware a perception source for each (a ROS 2 robot's TF frames)."""
    if GROUND_TRUTH not in world.backend.capabilities:
        blind = [o for o in objects if o not in world.perception]
        if blind:
            pytest.skip(f"no ground truth, and no perception of {', '.join(blind)}")


@pytest.fixture(autouse=True)
def _arms_only(rw_robot):
    from robowright import robots

    if robots.get(rw_robot).family != "arm":
        pytest.skip("arm contract (legged robots: robowright.contract.test_legged)")


def test_reports_the_robot_joints(world):
    b = world.backend
    m = b.robot_model
    assert b.joint_names == [*m.arm_joints, "gripper"]
    assert b.qpos().shape == (m.n_arm + 1,) and b.qvel().shape == (m.n_arm + 1,) and b.ctrl().shape == (m.n_arm + 1,)


def test_kinematics_match_the_simulated_hand(world):
    """The simulator's hand pose agrees with robowright's kinematics for random joint angles."""
    r, b = world.robot, world.backend
    if type(b).hand_pose is Backend.hand_pose:
        pytest.skip("the backend does not report link poses")
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
    _sees(world, "cube")
    world.wait(0.5)
    cube = world.scene["cube"]
    assert cube.position[2] == pytest.approx(0.0125, abs=0.002)
    expect(cube).to_be_at_rest()
    if CONTACTS in world.backend.capabilities:
        pairs = {frozenset((c.a, c.b)) for c in world.backend.contacts()}
        assert frozenset(("cube", "floor")) in pairs


def test_grasp_is_seen_by_both_fingers(world):
    _sees(world, "cube")
    cube = world.scene["cube"]
    world.robot.pick(cube)
    expect(world.robot.gripper).to_be_holding(cube)
    assert cube.position[2] > 0.04  # lifted


def test_pick_and_place(world):
    _sees(world, "cube", "bin")
    cube, bin_ = world.scene["cube"], world.scene["bin"]
    world.robot.pick(cube)
    world.robot.place(on=bin_)
    expect(cube).to_be_inside(bin_)
    expect(cube).to_be_at_rest()


def test_places_a_second_object_beside_the_first(rw_backend, rw_robot):
    """place(on=bin) into an occupied bin finds free floor instead of stacking on what is there."""
    import robowright as rw
    from robowright.scene import ObjectSpec, tabletop

    scene = tabletop(
        ObjectSpec("cube", "box", (0.0125, 0.0125, 0.0125), (0.22, -0.06, None), color="red"),
        ObjectSpec("cube2", "box", (0.0125, 0.0125, 0.0125), (0.16, -0.10, None), color="green"),
        ObjectSpec("bin", "bin", (0.05, 0.05, 0.02), (0.2, 0.12, 0.0), color="blue", mass=0.0),
        robot=rw_robot,
    )
    with rw.launch(scene, backend=rw_backend, settings=rw.Settings(trace="off")) as w:
        _sees(w, "cube", "cube2", "bin")
        w.robot.reset_to()
        for name in ("cube", "cube2"):
            w.robot.pick(w.scene[name])
            w.robot.place(on=w.scene["bin"])
        for name in ("cube", "cube2"):
            expect(w.scene[name]).to_be_inside(w.scene["bin"])
            expect(w.scene[name]).to_be_at_rest()
            # not stacked on the other cube (that puts its centre 37.5 mm up; leaning on a wall, ~27 mm)
            assert w.scene[name].position[2] < w.scene["bin"].bounds()[0][2] + 0.031


def test_grip_force_matches_the_datasheet(world, robot):
    """Where the maker publishes a grip force, a held cube is pressed with it by each jaw (within 15%)."""
    spec = robot.model.grip_force
    if spec is None:
        pytest.skip("no published grip force: the gripper squeezes as modelled")
    world.require("contacts", "contact forces")
    robot.pick(world.scene["cube"])
    world.wait(0.3)
    jaw = {"left_finger": [], "right_finger": []}
    for _ in range(10):
        world.step()
        force = dict.fromkeys(jaw, 0.0)
        for c in world.backend.contacts():
            for me, other in ((c.a, c.b), (c.b, c.a)):
                if other == "cube" and me.split(":")[-1] in force:
                    force[me.split(":")[-1]] += c.force
        for k in jaw:
            jaw[k].append(force[k])
    measured = float(np.mean([np.mean(v) for v in jaw.values()]))
    assert abs(measured - spec) < 0.15 * spec, f"each jaw presses {measured:.1f} N; the datasheet says {spec} N"


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


def test_state_restore_mid_grasp_is_exact(world):
    """A state captured with the cube squeezed in the jaws replays the same future, even after the run moved on."""
    b = world.backend
    if STATE not in b.capabilities or DETERMINISTIC not in b.capabilities:
        pytest.skip("no deterministic state save/restore")
    cube = world.scene["cube"]
    world.robot.pick(cube)
    ctrl = b.qpos().copy()
    ctrl[0] += 0.3  # carry it: the grasp is loaded while the arm swings

    def run():
        b.set_ctrl(ctrl)
        out = []
        for _ in range(30):
            b.step()
            out.append(np.concatenate([b.qpos(), *b.object_pose("cube")]))
        return np.array(out)

    s = b.get_state()
    first = run()
    world.wait(0.5)
    b.set_state(s)
    assert np.array_equal(run(), first)


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
