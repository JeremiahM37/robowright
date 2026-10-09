"""Known divergences: (test, backend, robot) combinations that fail for a measured reason.

These are findings, not bugs in robowright: the robot model or the engine
behaves differently from the others, and the test says so. They are marked
strict xfail, so CI stays green while they hold and fails loudly the day one
starts passing (a model fix upstream, an engine change), prompting removal.
Each reason is what the trace showed.

``KNOWN`` holds for the default fidelity mode, ``published`` (the robots as their model files
have them, see robowright.fidelity); ``KNOWN_ADJUSTED`` for ``--rw-fidelity adjusted``, where
robowright's own tuning makes most of the published findings pass.
"""

import functools

import pytest

KNOWN: dict[tuple[str, str, str], str] = {}
KNOWN_ADJUSTED: dict[tuple[str, str, str], str] = {}
# --- published mode (the default) --------------------------------------------------------------
# Robots read from model files outside the catalogue (pytest --rw-robot path), as their makers
# modelled them. Each was measured against --rw-fidelity adjusted, where it passes: the one setting
# that differs is named.
#
# Menagerie's Franka FR3 v2 keeps MuJoCo's Euler integrator. Its joint 1 servo (kp 4500, kv 450,
# damping applied explicitly at the model's 2 ms step) oscillates +-15 mrad under a position command
# and never settles; measured in plain MuJoCo on the file as shipped, no robowright code involved.
# With implicitfast it settles exactly.
for test in (
    "test_arm_never_hits_anything_while_picking",
    "test_cube_survives_a_shove_when_held",
    "test_deterministic",
    "test_grasp_is_seen_by_both_fingers",
    "test_moves_the_tool_to_a_point",
    "test_no_arm_collisions_during_a_pick",
    "test_pick_and_place",
    "test_places_a_second_object_beside_the_first",
    "test_policy_with_randomized_cube",
    "test_state_restore_mid_grasp_is_exact",
    "test_state_round_trip",
    "test_tracks_a_joint_move",
):
    KNOWN[(test, "mujoco", "fr3v2")] = (
        "the FR3 v2 model's Euler integrator lets its joint 1 servo oscillate +-15 mrad at its own 2 ms step: the arm never settles"
    )
# Google Robot and TIAGo keep MuJoCo's default pyramidal friction cone: each picks the cube, and a held
# cube creeps out of its fingers on the way to the bin (with the elliptic cone and impratio 10 that
# robowright's adjusted mode gives them, it arrives). TIAGo's sliding fingers are also coupled softly.
for robot in ("google_robot", "tiago_position", "tiago_dual_position"):
    for test in (
        "test_pick_and_place",
        "test_places_a_second_object_beside_the_first",
        "test_policy_with_randomized_cube",
        "test_policy_with_sensor_noise_and_latency",
    ):
        KNOWN[(test, "mujoco", robot)] = f"{robot}'s model has the pyramidal friction cone; the held cube creeps out in transit"
KNOWN[("test_picks_a_tall_can_from_the_side", "mujoco", "google_robot")] = (
    "the Google Robot model's pyramidal friction cone lets a held can creep out of a side grasp"
)
for robot in ("tiago_position", "tiago_dual_position"):
    KNOWN[("test_cube_survives_a_shove_when_held", "mujoco", robot)] = (
        "TIAGo's sliding fingers, softly coupled in the model, part under the shove and the cube drops"
    )
KNOWN[("test_cube_survives_a_shove_when_held", "mujoco", "open_manipulator_x")] = (
    "the OpenManipulator-X model couples its two sliding fingers softly; shoved, they part and let the cube go"
)
# Menagerie's Unitree Z1 drives its jaw with the arm joints' 30 N m motor. With nothing between the jaws
# it never settles shut, and the reaction shakes the arm 0.16 rad off its home pose. (On a cube it holds:
# every pick test passes as published, which the 60 N cap of the adjusted mode broke.)
for test in ("test_gripper_opens_and_closes", "test_holds_the_home_pose"):
    KNOWN[(test, "mujoco", "z1_gripper")] = "the Z1 model's 30 N m jaw motor never settles shut on nothing; the arm shakes with it"

# Engines other than MuJoCo, published. Measured; each passes with --rw-fidelity adjusted.
#
# PyBullet's contacts are rigid. The SO-arm family's modelled gripper presses a 25 mm block with
# ~65 N in MuJoCo (the STS3215's full torque at its jaw); carried over, it pushes the cube 9 mm into
# the fixed jaw's 2 mm pad plates and, in some grasps, squeezes it out (MuJoCo's soft contacts hold
# it). The URDF SO-100 and SO-101 lose it while closing: the jaws shut on nothing, the cube 2-4 cm
# away. Genesis does the same to the URDF ones.
_SQUEEZED = "the SO-arm's modelled ~65 N squeeze, on rigid contacts, pushes the cube out of the jaws"
for test in (
    "test_learned_policy_puts_the_cube_in_the_bin",
    "test_policy_with_randomized_cube",
    "test_policy_with_sensor_noise_and_latency",
    "test_places_a_second_object_beside_the_first",
):
    KNOWN[(test, "pybullet", "so101")] = _SQUEEZED
for robot in ("so100", "so101_new_calib"):
    for test in (
        "test_deterministic",
        "test_grasp_is_seen_by_both_fingers",
        "test_no_arm_collisions_during_a_pick",
        "test_pick_and_place",
        "test_places_a_second_object_beside_the_first",
    ):
        KNOWN[(test, "pybullet", robot)] = _SQUEEZED
for test in (
    "test_deterministic",
    "test_grasp_is_seen_by_both_fingers",
    "test_no_arm_collisions_during_a_pick",
    "test_pick_and_place",
    "test_places_a_second_object_beside_the_first",
    "test_state_restore_mid_grasp_is_exact",
):
    KNOWN[(test, "genesis", "so101_new_calib")] = _SQUEEZED
KNOWN[("test_places_a_second_object_beside_the_first", "genesis", "so100")] = _SQUEEZED
# PyBullet has no closed kinematic chains: the xArm Gripper's four-bar linkage runs as a motor on each
# passive joint. Driven at the datasheet's 30 N the jaws close unevenly and push the cube 4 cm aside.
for test in (
    "test_arm_never_hits_anything_while_picking",
    "test_cube_survives_a_shove_when_held",
    "test_deterministic",
    "test_grasp_is_seen_by_both_fingers",
    "test_grip_force_matches_the_datasheet",
    "test_no_arm_collisions_during_a_pick",
    "test_pick_and_place",
    "test_picks_a_tall_can_from_the_side",
    "test_policy_with_randomized_cube",
    "test_policy_with_sensor_noise_and_latency",
):
    KNOWN[(test, "pybullet", "xarm7")] = "PyBullet runs the xArm Gripper's four-bar linkage as separate motors; its jaws close unevenly"
# The Sawyer's servos stop short under load without the integral help of the adjusted mode, and on the
# scripted policy's way over the bin a finger catches the rim (51.8 N on the bin): 16 of 20 arrive.
KNOWN[("test_policy_with_randomized_cube", "pybullet", "sawyer")] = "the Sawyer's finger catches the bin's rim in 4 of 20"
KNOWN[("test_picks_a_tall_can_from_the_side", "pybullet", "sawyer")] = "PyBullet: the Sawyer loses the tall can from a side grasp"
KNOWN[("test_places_a_second_object_beside_the_first", "pybullet", "panda_2")] = "PyBullet: the URDF Panda drops the second cube"
KNOWN[("test_picks_a_tall_can_from_the_side", "genesis", "xarm7")] = "Genesis (its own solver settings): the xArm 7 loses the tall can"
# Drake's default hydroelastic modulus (1e7 Pa): a 30 g cube set down on the bin floor chatters in place
# and never comes to rest (at robowright's adjusted 3e6 it settles).
_CHATTERS = "with Drake's default contact stiffness, the placed cube chatters on the bin floor and never rests"
for test in ("test_pick_and_place", "test_places_a_second_object_beside_the_first"):
    KNOWN[(test, "drake", "arx_l5")] = _CHATTERS
for robot in ("so101", "ur5e", "wx250s"):
    KNOWN[("test_policy_with_randomized_cube", "drake", robot)] = _CHATTERS
KNOWN[("test_learned_policy_puts_the_cube_in_the_bin", "drake", "so101")] = _CHATTERS
# Drake, the URDF SO-100: carried to the bin, the cube leaves the jaw fast as it opens and lands 7 cm past it.
for test in ("test_pick_and_place", "test_places_a_second_object_beside_the_first"):
    KNOWN[(test, "drake", "so100")] = "Drake: the SO-100's jaw flings the cube 7 cm past the bin as it opens"

# Outcomes that differ between builds of the same engine: the SO-101 on PyBullet keeps the cube
# through a 1.5 N shove on some builds and loses it on others (CI's Python 3.10, 3.12 and 3.13
# wheels have each gone either way; this machine's keeps it). It is the squeeze above, at its
# margin: whether the shove lands while the cube is pressed into the fixed jaw's pads decides it.
KNOWN[("test_cube_survives_a_shove_when_held", "pybullet", "so101")] = (
    _SQUEEZED + "; whether a 1.5 N shove dislodges it varies by PyBullet build"
)
UNSETTLED = {("test_cube_survives_a_shove_when_held", "pybullet", "so101")}

# --- adjusted mode -----------------------------------------------------------------------------
# Genesis's friction with the elliptic cone the backend uses (see its comment for every setting
# tried, and what each cost): the iiwa 14 picks its second cube and carries it, then drops it
# halfway through an ordinary swing to the bin (0.4 rad at the base and wrist), as no other engine
# does. The first cube, on a similar swing, arrives.
KNOWN_ADJUSTED[("test_places_a_second_object_beside_the_first", "genesis", "iiwa14")] = (
    "in Genesis the iiwa 14's Robotiq grip lets the second cube go mid-swing to the bin (MuJoCo, PyBullet, Drake and Isaac Sim carry it)"
)
# Robots read from model files outside the catalogue (pytest --rw-robot path), where their geometry
# rules a test out. Measured, not assumed:
# Menagerie's Unitree Z1 picks the cube, but cannot carry it. Its moving jaw swings from a pivot 9 cm
# above its pad, so at a 25 mm cube the jaws stand 13 degrees apart, closer at the top: the squeeze
# pushes the cube down with ~9 N that friction must hold for as long as it is carried (its weight is
# 0.3 N). MuJoCo's friction creeps under a sustained load, 0.5-1.5 mm/s whatever the contact
# stiffness, noslip iterations (to 100) or impratio (to 100), and the cube slides off the pads'
# lower edge on the way to the bin. The scripted policy regrasps a slipping object: it carries the
# cube 20/20 under sensor noise, but not from every randomized start.
for test in (
    "test_pick_and_place",
    "test_places_a_second_object_beside_the_first",
    "test_policy_with_randomized_cube",
):
    KNOWN_ADJUSTED[(test, "mujoco", "z1_gripper")] = (
        "the Z1's swinging jaw meets a 25 mm cube 13 degrees off its fixed jaw and wedges it downward; "
        "under that load the cube creeps out of the grip before it reaches the bin"
    )


def pytest_collection_modifyitems(config, items):
    from robowright import fidelity

    known = KNOWN_ADJUSTED if fidelity.adjusted() else KNOWN
    for item in items:
        cs = getattr(item, "callspec", None)
        if cs is None:
            continue
        backend = cs.params.get("rw_backend", config.getoption("--rw-backend").split(",")[0])
        robot = cs.params.get("rw_robot", config.getoption("--rw-robot").split(",")[0])
        reason = known.get((item.originalname, backend, _name(robot)))
        if reason:
            # Strict, so a divergence that stops holding fails loudly - except where the outcome
            # itself is known to differ between builds of an engine (UNSETTLED below).
            loose = (item.originalname, backend, _name(robot)) in UNSETTLED
            item.add_marker(pytest.mark.xfail(reason=reason, strict=not loose))


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
