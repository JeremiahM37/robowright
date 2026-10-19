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


def test_an_unreadable_state_snapshot_says_where():
    from robowright.snapshots import parse_state

    with pytest.raises(ValueError, match=r"x.state.txt:2: .*'cube: near'"):
        parse_state("objects:\n  cube: near\n", "x.state.txt")
