"""Robot description files shipped with robowright.

The SO-101 MJCF and meshes come from MuJoCo Menagerie, Apache-2.0 (see
``so101/LICENSE-so101``). Every other robot is fetched from Menagerie on
first use (``robowright.robots.menagerie``) and keeps its own licence.
"""

from pathlib import Path

ROOT = Path(__file__).parent


def so101_mjcf() -> Path:
    return ROOT / "so101" / "so101.xml"
