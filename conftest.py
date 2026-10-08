"""Known divergences: (test, backend, robot) combinations that fail for a measured reason.

These are findings, not bugs in robowright: the robot model or the engine
behaves differently from the others, and the test says so. They are marked
strict xfail, so CI stays green while they hold and fails loudly the day one
starts passing (a model fix upstream, an engine change), prompting removal.
Each reason is what the trace showed.
"""

import functools

import pytest

# PyBullet once dropped cubes from every light gripper here (PiPER, ARX L5, YAM, WidowX), recorded
# as its contact model letting a weak grip slip. None of it was the contact model: the PiPER is now
# driven at its datasheet's 40 N; the rest were robowright driving PyBullet's arm in a staircase (a
# velocity spike each control step) and coupling sliding fingers by a second motor that let the
# pair drift sideways. Both are fixed in the PyBullet backend.
_SLIDE = {"arx_l5": "0.9 N", "yam": "1.9 N", "wx250s": "2.2 N"}

KNOWN: dict[tuple[str, str, str], str] = {}
# A 1.5 N shove for 0.1 s against grippers modelled at a few newtons. MuJoCo lets the cube go;
# PyBullet's stiffer contacts hold it, except for the weakest grip. The WidowX (2.2 N) and the
# ViperX hold it on MuJoCo since the gripper closes at 0.7 s a stroke, not 0.35.
for robot in ("arx_l5", "yam"):
    KNOWN[("test_cube_survives_a_shove_when_held", "mujoco", robot)] = (
        f"MuJoCo Menagerie's model of this gripper squeezes {_SLIDE[robot]}; a 1.5 N shove knocks the cube out"
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
# straight line at 0.05 m/s; MuJoCo, PyBullet and Drake hold it. Panda and ViperX keep it. The
# Robotiq 2F-85 arms (UR5e, UR10e, Gen3, Sawyer) hold it since the gripper linkages' mimic couplings
# are held rigid on Genesis, as on MuJoCo.
for robot in ("xarm7",):
    KNOWN[("test_picks_a_tall_can_from_the_side", "genesis", robot)] = (
        "in Genesis a can held from the side creeps out of the fingers while carried, whatever the speed "
        "(MuJoCo, PyBullet and Drake hold it)"
    )
# Hello Robot's Stretch 3 (from MuJoCo Menagerie's file): its rounded rubber fingertip pads pinch the
# cube at a point each. In Genesis the cube rides 1.7 mm up as the jaws close and slides 6 mm along
# the grip in the first centimetre of lift, then drops; MuJoCo holds it within 0.4 mm, as do
# PyBullet and Drake.
for test in (
    "test_pick_and_place",
    "test_grasp_is_seen_by_both_fingers",
    "test_places_a_second_object_beside_the_first",
    "test_no_arm_collisions_during_a_pick",
    "test_state_restore_mid_grasp_is_exact",
    "test_deterministic",
    "test_arm_never_hits_anything_while_picking",
    "test_cube_survives_a_shove_when_held",
    "test_policy_with_randomized_cube",
    "test_policy_with_sensor_noise_and_latency",
):
    KNOWN[(test, "genesis", "stretch")] = (
        "Stretch's rounded rubber pads pinch the cube at a point each; in Genesis it slides out of them as it is lifted "
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
        reason = KNOWN.get((item.originalname, backend, _name(robot)))
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


@functools.cache
def _name(robot: str) -> str:
    """A robot's name, for one given by its model file (the registry is keyed by name)."""
    if "/" not in robot and not robot.endswith((".xml", ".urdf", ".xacro")):
        return robot
    from robowright import robots

    try:
        return robots.get(robot).name
    except Exception:
        return robot
