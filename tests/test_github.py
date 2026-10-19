"""--rw-github: failures as GitHub Actions annotations at the line of the test that failed (Playwright's github
reporter), on by itself on a GitHub runner."""

import pytest

TEST = """
from robowright import expect


def lift(robot, cube):
    robot.pick(cube, lift=0.05)


def test_clears_the_lip(robot, scene):
    lift(robot, scene["cube"])
    expect(scene["cube"]).to_be_above(0.12, timeout=0.5)


def test_fine(world):
    pass
"""


def _annotations(r):
    return [line for line in r.stdout.lines if line.startswith("::")]


@pytest.mark.parametrize("workers", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_a_failure_is_an_error_at_its_line_on_a_github_runner(pytester, monkeypatch, workers):
    if workers:
        pytest.importorskip("xdist")
    pytester.makepyfile(test_lip=TEST)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(pytester.path))
    r = pytester.runpytest_subprocess("-p", "no:cacheprovider", *workers)
    r.assert_outcomes(passed=1, failed=1)
    (a,) = _annotations(r)
    assert a.startswith("::error file=test_lip.py,line=10,title=test_lip.py%3A%3Atest_clears_the_lip[mujoco]::")
    assert "to_be_above" in a and "%0A%0Arobowright show-trace robowright-traces/" in a and "\n" not in a


def test_no_annotations_off_github_and_none_when_turned_off(pytester, monkeypatch):
    pytester.makepyfile(test_lip=TEST)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert _annotations(pytester.runpytest("-p", "no:cacheprovider")) == []
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert _annotations(pytester.runpytest("-p", "no:cacheprovider", "--rw-github", "off")) == []


def test_paths_are_from_the_repository_root(pytester, monkeypatch):
    sub = pytester.mkdir("robot_tests")
    (sub / "test_lip.py").write_text(TEST)  # its leading blank line kept: the expect is on line 11
    monkeypatch.setenv("GITHUB_WORKSPACE", str(pytester.path))
    monkeypatch.chdir(sub)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-github", "on", str(sub / "test_lip.py"))
    (a,) = _annotations(r)
    assert a.startswith("::error file=robot_tests/test_lip.py,line=11,")


def test_a_flaky_test_is_a_warning(pytester, monkeypatch):
    monkeypatch.setenv("GITHUB_WORKSPACE", str(pytester.path))  # as on a runner: paths from the checkout
    pytester.makepyfile(
        test_once="""
from pathlib import Path

def test_once(world):
    f = Path("n"); n = int(f.read_text()) if f.exists() else 0; f.write_text(str(n + 1))
    assert n, "the first attempt fails"
"""
    )
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "1", "--rw-github", "on")
    (a,) = _annotations(r)  # the failed attempt is not an error: the test passed in the end
    assert a.startswith("::warning file=test_once.py,line=5,title=flaky%3A test_once.py%3A%3Atest_once[mujoco]::")
    assert "passed on a retry after failing: AssertionError: the first attempt fails" in a
