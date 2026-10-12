"""Two arms from their own model files, which robowright could not drive: why the placement and the home
are what they are, and what an arm that cannot grasp top-down says.

The models are MuJoCo Menagerie's and are not in the repo. Point ROBOWRIGHT_ARM_FILES at a folder that
holds ``robotis_omy_3m/`` and ``seeed_rebot_devarm/`` to run these; otherwise they skip.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from robowright import robots
from robowright.robots.detect import DetectionError, arm_reach, build

ROOT = os.environ.get("ROBOWRIGHT_ARM_FILES")
OMY = Path(ROOT or ".") / "robotis_omy_3m" / "omy_3m.xml"
DEVARM = Path(ROOT or ".") / "seeed_rebot_devarm" / "seeed_rebot_devarm.xml"


def _need(path: Path):
    if not ROOT or not path.exists():
        pytest.skip("set ROBOWRIGHT_ARM_FILES to a folder holding robotis_omy_3m/ and seeed_rebot_devarm/")


def _load(path: Path, name: str, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MuJoCo's notes on attaching a gripper
        return robots.load(str(path), name=name, **overrides)


def test_omy_home_is_clear_of_the_floor_and_a_pose_pressed_into_it_is_not():
    """omy's shoulder hung on the floor at the home pose the first detection chose (3 mm into it by
    MuJoCo's own contacts, which the bounding boxes missed); the placed home is clear of it."""
    _need(OMY)
    from robowright.robot import _on_floor, home_q
    from robowright.robots.model import floor_test

    m = _load(OMY, "omy_floor")
    assert not _on_floor(m.name, home_q(m.name))
    pressed = np.array([0.694, 2.114, -1.715, -0.4, -0.694, -3.14])  # the home q before the floor check
    assert floor_test(m)(pressed)


def test_omy_reach_is_measured_from_its_joint_ranges():
    _need(OMY)
    m = _load(OMY, "omy_reach")
    assert 0.8 < arm_reach(m) < 1.1


def test_load_refuses_a_home_the_arm_cannot_reach_and_says_how_close_it_gets():
    _need(DEVARM)
    with pytest.raises(DetectionError) as info:
        _load(DEVARM, "devarm_home", base_pos=(0.0, 0.0, 0.0), home=(0.2, 0.0, 0.15))
    msg = str(info.value)
    assert "home pose (0.2, 0.0, 0.15) unreachable at base_pos=(0.0, 0.0, 0.0) (closest " in msg
    assert "mm)" in msg
    assert "the nearest home it reaches is home=" in msg or "no home on a 4 cm grid" in msg


def test_devarm_cannot_be_driven_and_the_error_says_it_cannot_and_why():
    """Its wrist cannot reach straight down to the table within its joint limits: no base_pos or home
    fixes that, so the message must not name them as the fix."""
    _need(DEVARM)
    with pytest.raises(DetectionError) as info:
        build(DEVARM)
    msg = str(info.value)
    assert "the arm reaches " in msg and " m from its base" in msg
    assert " mm away from any mounting tried" in msg
    assert "cannot do the tabletop tasks as modelled" in msg
    assert "base_pos=..." not in msg


def test_check_runs_pytest_without_a_cache(monkeypatch):
    """`robowright check` from a robot's own folder left a .pytest_cache in it."""
    import pytest as pytest_module

    from robowright import cli

    seen = {}

    def fake_main(args):
        seen["args"] = list(args)
        return 0

    monkeypatch.setattr(pytest_module, "main", fake_main)
    cli._check(["--robot", "so101"])
    assert "no:cacheprovider" in seen["args"]
