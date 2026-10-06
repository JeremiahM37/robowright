"""Known divergences: (test, backend, robot) combinations that fail for a measured reason.

These are findings, not bugs in robowright: the robot model or the engine
behaves differently from the others, and the test says so. They are marked
strict xfail, so CI stays green while they hold and fails loudly the day one
starts passing (a model fix upstream, an engine change), prompting removal.
Each reason is what the trace showed.
"""

import pytest

# PyBullet once dropped cubes from every light gripper here (PiPER, ARX L5, YAM, WidowX), recorded
# as its contact model letting a weak grip slip. None of it was the contact model: the PiPER is now
# driven at its datasheet's 40 N; the rest were robowright driving PyBullet's arm in a staircase (a
# velocity spike each control step) and coupling sliding fingers by a second motor that let the
# pair drift sideways. Both are fixed in the PyBullet backend.
_SLIDE = {"arx_l5": "0.9 N", "yam": "1.9 N", "wx250s": "2.2 N"}

KNOWN: dict[tuple[str, str, str], str] = {}
# A 1.5 N shove for 0.1 s against grippers modelled at a few newtons. MuJoCo lets the cube go;
# PyBullet's stiffer contacts hold it, except for the weakest grip.
for robot, force in _SLIDE.items():
    KNOWN[("test_cube_survives_a_shove_when_held", "mujoco", robot)] = (
        f"MuJoCo Menagerie's model of this gripper squeezes {force}; a 1.5 N shove knocks the cube out"
    )
# The ViperX's modelled grip chatters on the cube (0 to 1.9 N, step to step), so whether a 1.5 N
# shove knocks it out depends on the chatter's phase when it lands: it lets go from 1.25-1.55 N.
KNOWN[("test_cube_survives_a_shove_when_held", "mujoco", "vx300s")] = (
    "MuJoCo Menagerie's ViperX grip chatters between 0 and 1.9 N on the cube; a 1.5 N shove knocks it out "
    "(it lets go from 1.25-1.55 N, depending on the chatter's phase)"
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
# A 10 cm can held 2.5 cm under its top from the side: in Genesis it creeps out of the fingers
# while carried (the gripper closing as it slips, opening 0.35 -> 0.06), even carried level in a
# straight line at 0.05 m/s; MuJoCo, PyBullet and Drake hold it. Panda and ViperX keep it.
for robot in ("ur5e", "ur10e", "gen3", "sawyer", "xarm7"):
    KNOWN[("test_picks_a_tall_can_from_the_side", "genesis", robot)] = (
        "in Genesis a can held from the side creeps out of the fingers while carried, whatever the speed "
        "(MuJoCo, PyBullet and Drake hold it)"
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
