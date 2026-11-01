"""[ros2] limits: a command that would move a real arm too fast or too far is refused before it is sent."""

import pytest

from robowright import limits
from robowright.errors import ActionError, SafetyLimitError

NAMES = ["shoulder", "elbow"]
DT = 0.02


def test_no_limits_lets_anything_through():
    limits.check(None, NAMES, [0, 0], [[3.0, -3.0]], DT)
    limits.check({}, NAMES, [0, 0], [[3.0, -3.0]], DT)


def test_a_move_inside_the_limits_passes():
    lim = {"joint_speed": 1.0, "positions": {"elbow": [-1.0, 1.0]}}
    path = [[0.01 * i, -0.01 * i] for i in range(1, 50)]  # 0.5 rad/s
    limits.check(lim, NAMES, [0, 0], path, DT)


def test_too_fast_is_refused_naming_the_joint_and_speed():
    with pytest.raises(SafetyLimitError, match=r"elbow would move at 5\.00/s, faster than 1\.0/s") as e:
        limits.check({"joint_speed": 1.0}, NAMES, [0, 0], [[0.0, 0.1]], DT)
    assert "nothing was sent" in str(e.value)
    assert isinstance(e.value, ActionError)  # a failed action, as the test reads it


def test_the_jump_from_where_the_arm_is_counts():
    # The first point is near the start of the path, but far from where the arm actually is.
    with pytest.raises(SafetyLimitError, match="shoulder"):
        limits.check({"joint_speed": 1.0}, NAMES, [1.0, 0.0], [[0.0, 0.0], [0.01, 0.0]], DT)


def test_outside_a_position_bound_is_refused():
    with pytest.raises(SafetyLimitError, match=r"shoulder would go to -2\.100, outside its limits \[-2\.0, -0\.5\]"):
        limits.check({"positions": {"shoulder": [-2.0, -0.5]}}, NAMES, [-1, 0], [[-1.5, 0], [-2.1, 0]], DT)


def test_a_bound_on_a_joint_the_arm_does_not_have_is_said():
    with pytest.raises(SafetyLimitError, match="'wrist', which is not an arm joint"):
        limits.check({"positions": {"wrist": [0, 1]}}, NAMES, [0, 0], [[0, 0]], DT)


def test_limits_is_a_ros2_setting():
    from robowright.backends import ros2_backend

    cfg = ros2_backend.settings({"limits": {"joint_speed": 0.3}})
    assert cfg["limits"] == {"joint_speed": 0.3}


def _tcp(q):  # a stand-in arm whose tool is where its two joints say, at a height of 0.5
    return (q[0], q[1], 0.5)


BOX = {"workspace": {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]}}


def test_a_path_inside_the_workspace_passes():
    limits.check(BOX, NAMES, [0, 0], [[0.5, 0.5], [0.9, -0.9]], DT, tcp=_tcp)


def test_a_path_that_leaves_the_workspace_is_refused_with_where_the_tool_would_be():
    with pytest.raises(SafetyLimitError, match=r"tool would reach \(1\.200, 0\.000, 0\.500\), outside the workspace .* in x") as e:
        limits.check(BOX, NAMES, [0, 0], [[0.5, 0.0], [1.2, 0.0], [0.5, 0.0]], DT, tcp=_tcp)  # out and back in
    assert "nothing was sent" in str(e.value)


def test_a_workspace_without_kinematics_refuses_rather_than_passing():
    with pytest.raises(SafetyLimitError, match="needs the arm's kinematics"):
        limits.check(BOX, NAMES, [0, 0], [[0.1, 0.1]], DT)


@pytest.mark.parametrize(
    "box", [{"min": [0, 0, 0]}, {"min": [0, 0], "max": [1, 1]}, {"min": [0, 0, 1], "max": [1, 1, 1]}, [0, 0, 0, 1, 1, 1]]
)
def test_a_malformed_workspace_says_what_it_should_be(box):
    with pytest.raises(SafetyLimitError, match="limits.workspace"):
        limits.check({"workspace": box}, NAMES, [0, 0], [[0.1, 0.1]], DT, tcp=_tcp)


def test_a_misspelt_limit_is_refused_not_ignored():
    with pytest.raises(SafetyLimitError, match="has no joint_sped: it takes joint_speed, positions and workspace"):
        limits.check({"joint_sped": 0.5}, NAMES, [0, 0], [[0.1, 0.1]], DT)


def test_the_workspace_uses_the_arms_own_kinematics():
    """robot.tcp.position's kinematics: the SO-101 at home is inside a box around it, and a target that
    folds the wrist down through the floor is refused."""
    import numpy as np

    from robowright import robots
    from robowright.robot import _kinematics, home_q

    model = robots.get("so101")
    kin = _kinematics("so101")
    home = np.asarray(home_q("so101"), float)[: model.n_arm]
    at = kin.tcp(home)
    box = {"workspace": {"min": list(at - 0.3), "max": list(at + 0.3)}}
    names = list(model.arm_joints)
    limits.check(box, names, home, [home], DT, tcp=kin.tcp)
    low = {"workspace": {"min": [-1, -1, at[2] + 0.05], "max": [1, 1, 1]}}  # its floor above where the tool is
    with pytest.raises(SafetyLimitError, match="outside the workspace .* in z"):
        limits.check(low, names, home, [home], DT, tcp=kin.tcp)
