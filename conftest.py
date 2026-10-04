"""Known divergences: (test, backend, robot) combinations that fail for a measured reason.

These are findings, not bugs in robowright: the robot model or the engine
behaves differently from the others, and the test says so. They are marked
strict xfail, so CI stays green while they hold and fails loudly the day one
starts passing (a model fix upstream, an engine change), prompting removal.
Each reason is what the trace showed.
"""

import pytest

WEAK_GRIP = (
    "MuJoCo Menagerie's model of this gripper squeezes {force}; MuJoCo's soft elliptic friction "
    "still holds a 30 g cube, PyBullet's contact model lets it slip"
)
# The PiPER is no longer here: robowright drives it with its datasheet's 40 N (RobotModel.grip_force).
_SLIDE = {"arx_l5": "0.9 N", "yam": "1.9 N", "wx250s": "2.2 N"}

KNOWN: dict[tuple[str, str, str], str] = {}
for robot, force in _SLIDE.items():
    for test in ("test_policy_with_randomized_cube", "test_policy_with_sensor_noise_and_latency"):
        KNOWN[(test, "pybullet", robot)] = WEAK_GRIP.format(force=force)
for test in ("test_pick_and_place", "test_places_a_second_object_beside_the_first"):
    KNOWN[(test, "pybullet", "arx_l5")] = WEAK_GRIP.format(force=_SLIDE["arx_l5"])
KNOWN[("test_places_a_second_object_beside_the_first", "genesis", "piper")] = (
    "Genesis couples the PiPER's second finger through a soft mimic constraint: closing at the datasheet's 40 N "
    "it lags and shoves the second cube 2.3 cm aside, so pick raises GraspError (stiffening the constraint makes "
    "Genesis knock even the first cube away)"
)
KNOWN[("test_grip_force_matches_the_datasheet", "genesis", "xarm7")] = (
    "Genesis's soft mimic constraints lose some of the xArm Gripper's linkage force: each jaw presses 25.4 N "
    "of the datasheet's 30 N (MuJoCo, Drake, Isaac: 28.5-30.7 N)"
)
# A 1.5 N shove for 0.1 s against grippers modelled at a few newtons. MuJoCo lets the cube go;
# PyBullet's stiffer contacts hold it, except for the weakest grip.
for robot, force in _SLIDE.items():
    KNOWN[("test_cube_survives_a_shove_when_held", "mujoco", robot)] = (
        f"MuJoCo Menagerie's model of this gripper squeezes {force}; a 1.5 N shove knocks the cube out"
    )
# Drake holds the 1.9 N YAM grip through the shove; the rest go, as in MuJoCo.
for robot, force in {**{r: _SLIDE[r] for r in ("arx_l5", "wx250s")}, "vx300s": "5.4 N"}.items():
    KNOWN[("test_cube_survives_a_shove_when_held", "drake", robot)] = (
        f"MuJoCo Menagerie's model of this gripper squeezes {force}; a 1.5 N shove knocks the cube out"
    )
for robot in ("yam", "arx_l5", "vx300s", "wx250s"):
    KNOWN[("test_cube_survives_a_shove_when_held", "genesis", robot)] = (
        "this gripper is modelled at a few newtons of squeeze; in Genesis, as in MuJoCo, a 1.5 N shove knocks the cube out"
    )
# Squeeze per finger measured in Isaac Sim, holding the cube at rest. The ViperX (0.75 N) keeps it.
for robot, force in {"wx250s": "0.73 N", "yam": "0.63 N", "arx_l5": "0.25 N"}.items():
    KNOWN[("test_cube_survives_a_shove_when_held", "isaac", robot)] = (
        f"in Isaac Sim this gripper squeezes {force} per finger; a 1.5 N shove knocks the cube out, as in MuJoCo"
    )

KNOWN[("test_grip_force_matches_the_datasheet", "pybullet", "xarm7")] = (
    "PyBullet has no closed kinematic chains, so the xArm Gripper's six linkage joints are driven by "
    "separate motors and press the cube with 13.7 N of the datasheet's 30 N (MuJoCo, Drake, Isaac: 29-31 N)"
)
KNOWN[("test_stands_on_its_own", "genesis", "spot")] = (
    "standing still, Spot creeps backward ~2 cm/s on its sphere feet in Genesis (MuJoCo: settles to 0.2 mm/s)"
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        cs = getattr(item, "callspec", None)
        if cs is None:
            continue
        backend = cs.params.get("rw_backend", config.getoption("--rw-backend").split(",")[0])
        robot = cs.params.get("rw_robot", config.getoption("--rw-robot").split(",")[0])
        reason = KNOWN.get((item.originalname, backend, robot))
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
