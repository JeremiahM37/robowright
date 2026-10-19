"""--rw-video: each test's run as an MP4 beside its trace, drawn from it (Playwright's video option)."""

import pytest

pytest.importorskip("imageio_ffmpeg", reason="--rw-video needs robowright[video]")

TEST = """
from robowright import expect

def test_passes(world, scene):
    expect(scene["cube"]).to_be_at_rest()

def test_fails(world, scene):
    expect(scene["cube"]).to_be_inside(scene["bin"], timeout=0.2)
"""


def _files(pytester, suffix):
    d = pytester.path / "robowright-traces"
    return sorted(p.name for p in d.glob(f"*{suffix}")) if d.exists() else []


def test_retain_on_failure_keeps_the_failed_tests_video(pytester):
    pytester.makepyfile(test_v=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-video", "retain-on-failure", "--rw-report", "report.html")
    r.assert_outcomes(passed=1, failed=1)
    assert _files(pytester, ".mp4") == ["test_v.py__test_fails[mujoco].mp4"]
    video = pytester.path / "robowright-traces" / "test_v.py__test_fails[mujoco].mp4"
    assert video.stat().st_size > 1000 and video.read_bytes()[4:8] == b"ftyp"  # an MP4
    assert '<video src="' in (pytester.path / "report.html").read_text()


def test_on_draws_every_test_and_keeps_only_the_traces_asked_for(pytester):
    pytester.makepyfile(test_v=TEST)
    pytester.runpytest("-p", "no:cacheprovider", "--rw-video", "on").assert_outcomes(passed=1, failed=1)
    assert _files(pytester, ".mp4") == ["test_v.py__test_fails[mujoco].mp4", "test_v.py__test_passes[mujoco].mp4"]
    assert _files(pytester, ".zip") == ["test_v.py__test_fails[mujoco].zip"]  # the default keeps a failure's trace only


def test_off_by_default(pytester):
    pytester.makepyfile(test_v=TEST)
    pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1, failed=1)
    assert _files(pytester, ".mp4") == []
