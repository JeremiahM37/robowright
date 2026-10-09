"""Known divergences: (test, backend, robot) combinations that fail for a measured reason.

These are findings, not bugs in robowright: the robot model or the engine
behaves differently from the others, and the test says so. They are marked
strict xfail, so CI stays green while they hold and fails loudly the day one
starts passing (a model fix upstream, an engine change), prompting removal.
Each reason is what the trace showed.
"""

import functools

import pytest

KNOWN: dict[tuple[str, str, str], str] = {}
# Genesis's friction with the elliptic cone the backend uses (see its comment for every setting
# tried, and what each cost): the iiwa 14 picks its second cube and carries it, then drops it
# halfway through an ordinary swing to the bin (0.4 rad at the base and wrist), as no other engine
# does. The first cube, on a similar swing, arrives.
KNOWN[("test_places_a_second_object_beside_the_first", "genesis", "iiwa14")] = (
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
    KNOWN[(test, "mujoco", "z1_gripper")] = (
        "the Z1's swinging jaw meets a 25 mm cube 13 degrees off its fixed jaw and wedges it downward; "
        "under that load the cube creeps out of the grip before it reaches the bin"
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
