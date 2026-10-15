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
