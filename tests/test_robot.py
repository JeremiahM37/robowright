import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.errors import GraspError


def test_every_public_name_is_importable():
    """``from robowright import *`` and each name in ``robowright.__all__`` resolve (GraspError was listed, not imported)."""
    assert [n for n in rw.__all__ if not hasattr(rw, n)] == []


def test_move_to_reaches_target(quiet_world):
    w = quiet_world()
    target = np.array([0.24, 0.05, 0.05])
    w.robot.arm.move_to(target)
    assert np.linalg.norm(w.robot.tcp.position - target) < 0.005


def test_linear_move_stays_on_line(quiet_world):
    w = quiet_world()
    w.robot.arm.move_to((0.22, 0.0, 0.07))
    start = w.robot.tcp.position.copy()
    end = np.array([0.22, 0.0, 0.025])
    worst = []
    w._step_hooks.append(lambda world: worst.append(_dist_to_segment(world.robot.tcp.position, start, end)))
    w.robot.arm.move_to(end, linear=True)
    assert max(worst) < 0.006


def _dist_to_segment(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def test_unreachable_target_raises(quiet_world):
    w = quiet_world()
    with pytest.raises(rw.UnreachableError, match="mm"):
        w.robot.arm.move_to((0.5, 0.0, 0.3))


def test_weak_servo_times_out_with_diagnostic(quiet_world):
    w = quiet_world()
    w.faults.weak_joint("shoulder_lift", 0.001)
    with pytest.raises(rw.ActionTimeoutError, match="shoulder_lift|elbow_flex|wrist_flex"):
        w.robot.arm.move_to((0.28, 0.0, 0.03), timeout=1.0)


def test_gripper_opening(quiet_world):
    w = quiet_world()
    w.robot.gripper.close()
    assert w.robot.gripper.opening < 0.1
    w.robot.gripper.open()
    assert w.robot.gripper.opening > 0.9
    w.robot.gripper.open(0.5)
    assert 0.4 < w.robot.gripper.opening < 0.6


def test_pick_and_place_skill(quiet_world):
    w = quiet_world()
    w.robot.pick(w.scene["cube"])
    assert w.robot.gripper.holding() == "cube"
    w.robot.place(on=w.scene["bin"])
    expect(w.scene["cube"]).to_be_inside(w.scene["bin"])
    assert w.robot.gripper.holding() is None


def test_joint_readings_and_observation(quiet_world):
    w = quiet_world()
    assert list(w.robot.joints) == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    obs = w.robot.observe(cameras=("front",), privileged=True, task="t")
    assert obs["images"]["front"].shape == (240, 320, 3)
    assert set(obs["objects"]) == {"cube", "bin"} and obs["task"] == "t"


def test_run_policy_executes_action_chunks(quiet_world):
    w = quiet_world()
    calls = []
    hold = np.concatenate([w.robot.home_q, [1.2]])

    def policy(obs):
        calls.append(obs["t"])
        return np.tile(hold, (5, 1))

    r = w.robot.run_policy(policy, timeout=1.0)
    assert r.success and r.steps == 50 and len(calls) == 10


def test_run_policy_stops_when_condition_met(quiet_world):
    w = quiet_world()
    r = w.robot.run_policy(lambda obs: obs["qpos"], until=lambda: w.time > 0.3, timeout=5)
    assert r.success and 0.3 < r.sim_seconds < 0.4


@pytest.mark.parametrize("name", ["so101", "ur5e"])
def test_a_plain_script_starts_at_home_and_can_pick(name):
    """``rw.launch`` outside pytest, with no fixture: the robot is at its home pose, not the model's zero pose.
    A UR5e's zero pose lies flat, and a pick planned from there dragged its forearm across the floor."""
    with rw.launch(robot=name, settings=rw.Settings(trace="off")) as w:
        r = w.robot
        assert np.allclose(np.asarray(w.backend.qpos())[: r.n_arm], r.home_q, atol=1e-6)
        r.pick(w.scene["cube"])
        expect(r).to_be_holding(w.scene["cube"])


def test_launch_names_a_mujoco_too_old_to_build_with_and_warns_about_an_untested_one():
    from robowright.world import check_mujoco

    with pytest.raises(rw.errors.RobowrightError, match="rw.launch builds its simulations with MuJoCo 3.7 or newer, and this is 3.3.0"):
        check_mujoco("3.3.0")
    with pytest.warns(UserWarning, match="tested on MuJoCo 3.13 and newer; this is 3.10.0"):
        check_mujoco("3.10.0")
    check_mujoco("3.13.0")
    check_mujoco()  # the installed one


def test_pick_of_an_object_already_in_the_jaws_keeps_the_hold(quiet_world):
    """After a shove that leaves the cube in the jaws, the pre-grasp pose (5 cm above it) is out of reach
    with the hand level, so pick() used to raise UnreachableError. The cube is already held: pick returns."""
    w = quiet_world()
    cube = w.scene["cube"]
    w.robot.pick(cube)
    w.faults.push("cube", (0, 3.0, 0), duration=0.2)
    w.wait(0.3)
    expect(cube).to_be_at_rest(hold=0.3, timeout=3)
    assert w.robot.gripper.holding() == "cube"
    w.robot.pick(cube)
    assert w.robot.gripper.holding() == "cube"


def test_pick_of_an_object_already_held_is_recorded_in_the_trace(tmp_path):
    w = rw.launch(robot="so101", backend="mujoco", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    try:
        w.robot.reset_to()
        cube = w.scene["cube"]
        w.robot.pick(cube)
        w.robot.pick(cube)
        notes = [e.name for e in w.trace.events if e.type == "log"]
        assert any("already held in the jaws" in str(n) for n in notes), notes
    finally:
        w.backend.close()


def test_object_touching_one_finger_is_not_already_held(tmp_path):
    """Only an object between both jaws counts as held. One touching a single finger takes the normal
    path, which lifts and checks the grasp again: here the jaws are not really closed on it, so that
    check fails instead of pick() returning early."""
    w = rw.launch(robot="so101", backend="mujoco", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    try:
        w.robot.reset_to()
        cube = w.scene["cube"]
        w.robot.gripper.touching = lambda: ({"cube"}, set())
        with pytest.raises(GraspError, match="lifted without it"):
            w.robot.pick(cube)
        assert not any("already held" in e.name for e in w.trace.events if e.type == "log")
    finally:
        del w.robot.gripper.touching
        w.backend.close()
