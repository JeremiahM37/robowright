"""Robot description files shipped with robowright.

The SO-101 MJCF and meshes come from MuJoCo Menagerie and the URDF from
TheRobotStudio/SO-ARM100, both Apache-2.0 (see ``so101/LICENSE-so101``).
"""

from pathlib import Path

ROOT = Path(__file__).parent


def so101_mjcf() -> Path:
    return ROOT / "so101" / "so101.xml"


def so101_urdf() -> Path:
    return ROOT / "so101" / "so101_new_calib.urdf"
