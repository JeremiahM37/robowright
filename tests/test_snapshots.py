"""expect(world).to_match_snapshot: a camera's view against an image kept next to the test, as Playwright's
toHaveScreenshot: the first run writes the baseline and fails, a matching view passes, a changed one fails
with the actual image and a diff written, and --rw-update-snapshots writes the baseline again."""

import pytest

TEST = """
from pathlib import Path
from robowright import expect

def test_view(world):
    if Path("move_cube").exists():
        world.move_object("cube", (0.25, 0.10, 0.012))
    world.step(5)
    expect(world).to_match_snapshot("tidy.png")
"""

BASELINE = "__snapshots__/test_view/tidy-mujoco-so101.png"


@pytest.fixture
def suite(pytester):
    pytester.makepyfile(test_view=TEST)
    return pytester


def _run(pytester, *args):
    return pytester.runpytest("-p", "no:cacheprovider", *args)


def test_the_first_run_writes_the_baseline_and_fails(suite):
    r = _run(suite)
    r.assert_outcomes(failed=1)
    assert "no snapshot" in r.stdout.str() and "wrote it from this run" in r.stdout.str()
    assert (suite.path / BASELINE).exists()


def test_the_same_view_passes(suite):
    _run(suite)
    r = _run(suite)
    r.assert_outcomes(passed=1)


def test_a_changed_view_fails_with_the_actual_image_and_a_diff(suite):
    _run(suite)
    (suite.path / "move_cube").touch()
    r = _run(suite)
    r.assert_outcomes(failed=1)
    out = r.stdout.str()
    assert "pixels differ" in out and "-actual.png" in out and "-diff.png" in out
    shots = sorted(p.name for p in (suite.path / "robowright-traces" / "snapshots").iterdir())
    assert any(s.endswith("-actual.png") for s in shots) and any(s.endswith("-diff.png") for s in shots)


def test_update_snapshots_writes_the_baseline_again(suite):
    _run(suite)
    (suite.path / "move_cube").touch()
    before = (suite.path / BASELINE).read_bytes()
    _run(suite, "--rw-update-snapshots").assert_outcomes(passed=1)
    assert (suite.path / BASELINE).read_bytes() != before
    _run(suite).assert_outcomes(passed=1)  # the moved cube is now what the view should show


def test_update_snapshots_changed_writes_only_what_no_longer_matches(suite):
    """The bare flag is Playwright's --update-snapshots=changed: a baseline that still matches is left as it is."""
    still_test = "from robowright import expect\n\ndef test_still(world):\n    expect(world).to_match_snapshot('still.png')\n"
    suite.makepyfile(test_still=still_test)
    _run(suite)
    still = suite.path / "__snapshots__/test_still/still-mujoco-so101.png"
    kept = still.stat().st_mtime_ns
    (suite.path / "move_cube").touch()
    before = (suite.path / BASELINE).read_bytes()
    r = _run(suite, "--rw-update-snapshots")
    r.assert_outcomes(passed=2)
    assert (suite.path / BASELINE).read_bytes() != before
    assert still.stat().st_mtime_ns == kept, "a matching baseline is not written again"
    _run(suite, "--rw-update-snapshots=all").assert_outcomes(passed=2)
    assert still.stat().st_mtime_ns != kept, "=all writes every one"


def test_update_snapshots_none_writes_nothing_even_a_missing_one(suite):
    r = _run(suite, "--rw-update-snapshots=none")
    r.assert_outcomes(failed=1)
    assert "--rw-update-snapshots none: not written" in r.stdout.str()
    assert not (suite.path / BASELINE).exists()


def test_the_bare_flag_before_a_test_path_leaves_the_path_a_path(suite):
    suite.makepyfile(test_other="def test_other():\n    pass\n")
    _run(suite, "--rw-update-snapshots", "test_other.py").assert_outcomes(passed=1)  # only test_other.py ran
    _run(suite, "--rw-update-snapshots", "all", "test_other.py").assert_outcomes(passed=1)


def test_a_snapshot_cannot_be_negated(suite):
    suite.makepyfile(
        test_neg="""
from robowright import expect
def test_neg(world):
    expect(world).not_.to_match_snapshot("x.png")
"""
    )
    r = _run(suite, "test_neg.py")
    r.assert_outcomes(failed=1)
    assert "cannot be negated" in r.stdout.str()


STATE = """
import os
from robowright import expect

def test_state(world, scene):
    if os.environ.get("MOVE_CUBE"):
        world.move_object("cube", (0.25, 0.05, 0.0125))
    world.wait(0.2)
    expect(world).to_match_state_snapshot("tidy")
"""


@pytest.fixture
def state_suite(pytester):
    pytester.makepyfile(test_state=STATE)
    return pytester


def _state_file(pytester):
    return pytester.path / "__snapshots__" / "test_state" / "tidy-mujoco-so101.state.txt"


def test_a_state_snapshot_is_written_first_then_matched(state_suite):
    r = _run(state_suite)
    r.assert_outcomes(failed=1)
    assert "no snapshot" in r.stdout.str()
    text = _state_file(state_suite).read_text()
    assert "objects:\n  cube: [" in text and "joints:\n" in text
    _run(state_suite).assert_outcomes(passed=1)


def test_a_moved_object_fails_naming_it(state_suite, monkeypatch):
    _run(state_suite)
    monkeypatch.setenv("MOVE_CUBE", "1")
    r = _run(state_suite)
    r.assert_outcomes(failed=1)
    out = r.stdout.str()
    assert "objects.cube: (0.250, 0.050, 0.01" in out and "0.01 allowed" in out
    assert "this run's state:" in out


def test_a_deleted_line_is_not_checked(state_suite, monkeypatch):
    _run(state_suite)
    path = _state_file(state_suite)
    path.write_text("".join(line for line in path.read_text().splitlines(keepends=True) if "cube:" not in line))
    monkeypatch.setenv("MOVE_CUBE", "1")
    _run(state_suite).assert_outcomes(passed=1)


def test_update_snapshots_writes_the_state_again(state_suite, monkeypatch):
    _run(state_suite)
    monkeypatch.setenv("MOVE_CUBE", "1")
    _run(state_suite, "--rw-update-snapshots").assert_outcomes(passed=1)
    _run(state_suite).assert_outcomes(passed=1)
    assert "cube: [0.2500, 0.0500" in _state_file(state_suite).read_text()


def test_update_snapshots_changed_keeps_a_deleted_line_deleted(state_suite, monkeypatch):
    _run(state_suite)
    path = _state_file(state_suite)
    path.write_text("".join(line for line in path.read_text().splitlines(keepends=True) if "wrist" not in line))
    monkeypatch.setenv("MOVE_CUBE", "1")
    _run(state_suite, "--rw-update-snapshots").assert_outcomes(passed=1)
    text = path.read_text()
    assert "cube: [0.2500, 0.0500" in text and "wrist" not in text
    _run(state_suite).assert_outcomes(passed=1)


def test_an_unreadable_state_snapshot_says_where():
    from robowright.snapshots import parse_state

    with pytest.raises(ValueError, match=r"x.state.txt:2: .*'cube: near'"):
        parse_state("objects:\n  cube: near\n", "x.state.txt")


MASKED = """
from pathlib import Path
from robowright import expect

def test_view(world):
    if Path("move_cube").exists():
        world.move_object("cube", (0.25, 0.10, 0.012))
    world.step(5)
    expect(world).to_match_snapshot("tidy.png", mask=["cube"])
"""


def test_a_masked_object_can_move_and_the_view_still_matches(pytester):
    pytester.makepyfile(test_view=MASKED)
    _run(pytester).assert_outcomes(failed=1)  # writes the baseline, and its mask
    assert (pytester.path / "__snapshots__/test_view/tidy-mujoco-so101.mask.png").exists()
    (pytester.path / "move_cube").write_text("")
    _run(pytester).assert_outcomes(passed=1)  # where it was and where it is are both left out


@pytest.mark.parametrize("engine", ["mujoco", "pybullet"])
def test_the_mask_covers_the_object_in_the_image(engine):
    """The mask is the object's box projected through the camera robowright placed: the red cube's pixels are inside."""

    import robowright as rw
    from robowright.snapshots import mask_of

    w = rw.launch(backend=engine)
    img = w.backend.render("front", 320, 240)[..., :3].astype(int)
    red = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
    m = mask_of(w, ["cube"], "front", 320, 240)
    assert red.sum() > 20  # the cube is in view
    assert (red & ~m).sum() <= 0.02 * red.sum()  # inside the mask
    assert m.sum() < 0.05 * m.size  # and the mask is not most of the image
    w.close()


REGION = """
import os
from robowright import expect

def test_cube_looks_right(world, robot, scene):
    if os.environ.get("GRIP"):
        robot.gripper.close()  # the robot changes, away from the cube
    if os.environ.get("MOVE"):
        x, y = os.environ["MOVE"].split(",")
        world.move_object("cube", (float(x), float(y), 0.0125))
    world.step(5)
    expect(scene["cube"]).to_match_snapshot("cube.png", max_diff_pixel_ratio=0.1)
    expect(world).to_match_snapshot("whole.png")
"""


def test_a_region_snapshot_checks_the_object_wherever_it_is(pytester, monkeypatch):
    """expect(cube).to_match_snapshot: the window around the cube, as Playwright's element screenshot."""
    import numpy as np
    from PIL import Image

    pytester.makepyfile(test_region=REGION)
    _run(pytester).assert_outcomes(failed=1)  # both baselines written
    _run(pytester, "--rw-update-snapshots").assert_outcomes(passed=1)
    crop = np.asarray(Image.open(pytester.path / "__snapshots__/test_region/cube-mujoco-so101.png"))
    assert crop.shape[0] < 80 and crop.shape[1] < 80  # a window around the cube, not the whole 320x240 view
    r, g, b = (crop[..., k].astype(int) for k in range(3))
    red = (r > 90) & (r > 2 * g) & (r > 2 * b)  # its lit and its shaded faces
    assert red.mean() > 0.4  # mostly the red cube
    monkeypatch.setenv("GRIP", "1")  # something else changes: the cube's region still matches
    r = _run(pytester)
    r.assert_outcomes(failed=1)
    # The cube's region is checked first; the whole view failing means the cube's check passed.
    assert "to_match_snapshot('whole.png') failed" in r.stdout.str()
    monkeypatch.delenv("GRIP")
    monkeypatch.setenv("MOVE", "0.23,-0.05")  # the cube itself moves a little: the window follows it
    r = _run(pytester)
    assert "to_match_snapshot('cube.png')" not in r.stdout.str()  # only the whole view differs


def test_a_snapshot_path_template_puts_the_baselines_where_it_says(pytester):
    """[snapshots] path in robowright.toml, Playwright's snapshotPathTemplate, relative to that file."""
    (pytester.path / "robowright.toml").write_text('[snapshots]\npath = "baselines/{robot}/{engine}/{test_file_path}/{name}{ext}"\n')
    pytester.mkdir("tests")
    (pytester.path / "tests" / "test_view.py").write_text(TEST)
    _run(pytester, "tests").assert_outcomes(failed=1)  # the first run writes it
    assert (pytester.path / "baselines/so101/mujoco/tests/test_view/tidy.png").exists()
    assert not (pytester.path / "tests/__snapshots__").exists()
    _run(pytester, "tests").assert_outcomes(passed=1)
