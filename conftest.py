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
_SLIDE = {"piper": "0.7 N", "arx_l5": "0.9 N", "yam": "1.9 N", "wx250s": "2.2 N"}

KNOWN: dict[tuple[str, str, str], str] = {}
for robot, force in _SLIDE.items():
    for test in ("test_policy_with_randomized_cube", "test_policy_with_sensor_noise_and_latency"):
        KNOWN[(test, "pybullet", robot)] = WEAK_GRIP.format(force=force)
for robot in ("piper", "arx_l5"):
    KNOWN[("test_pick_and_place", "pybullet", robot)] = WEAK_GRIP.format(force=_SLIDE[robot])
# A 1.5 N shove for 0.1 s against grippers modelled at a few newtons. MuJoCo lets the cube go;
# PyBullet's stiffer contacts hold it, except for the weakest grip.
for robot, force in {**_SLIDE, "panda": "1.3 N (the real Franka Hand: 70 N)", "vx300s": "5.4 N"}.items():
    KNOWN[("test_cube_survives_a_shove_when_held", "mujoco", robot)] = (
        f"MuJoCo Menagerie's model of this gripper squeezes {force}; a 1.5 N shove knocks the cube out"
    )
KNOWN[("test_cube_survives_a_shove_when_held", "pybullet", "piper")] = "the PiPER model squeezes 0.7 N; a 1.5 N shove knocks the cube out"


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
