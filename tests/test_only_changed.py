"""--rw-only-changed: only the test files a change since a git commit can affect run (Playwright's --only-changed)."""

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

TEST = """
from {module} import VALUE

def test_{name}():
    assert VALUE == 1
"""


def _git(path, *args):
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


@pytest.fixture
def project(pytester):
    """Two tests, each importing its own module (one through a package's relative import), committed."""
    pytester.makepyfile(
        controller="VALUE = 1\n",
        test_controller=TEST.format(module="controller", name="controller"),
        test_planner=TEST.format(module="pkg.api", name="planner"),
        test_plain="def test_plain():\n    pass\n",
    )
    pkg = pytester.mkpydir("pkg")
    (pkg / "planner.py").write_text("VALUE = 1\n")
    (pkg / "api.py").write_text("from .planner import VALUE\n")
    _git(pytester.path, "init", "-q")
    _git(pytester.path, "add", "-A")
    _git(pytester.path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "start")
    return pytester


def _ran(r):
    return sorted(line.split("::")[0] for line in r.stdout.lines if "::" in line and "PASSED" in line)


def test_nothing_changed_runs_nothing_and_passes(project):
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed")
    assert r.ret == 0
    assert _ran(r) == []
    assert "0 test file(s) affected" in r.stdout.str()


def test_a_changed_test_file_runs_alone(project):
    (project.path / "test_plain.py").write_text("def test_plain():\n    assert True\n")
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed")
    assert _ran(r) == ["test_plain.py"]


def test_a_changed_module_runs_the_tests_that_import_it(project):
    (project.path / "controller.py").write_text("VALUE = 1  # tuned\n")
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed")
    assert _ran(r) == ["test_controller.py"]


def test_a_path_after_the_bare_flag_is_a_path_not_a_ref(project):
    (project.path / "controller.py").write_text("VALUE = 1  # tuned\n")
    (project.path / "test_plain.py").write_text("def test_plain():\n    assert True\n")
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed", "test_controller.py")
    assert _ran(r) == ["test_controller.py"]


def test_a_module_reached_through_relative_imports(project):
    (project.path / "pkg" / "planner.py").write_text("VALUE = 1  # faster\n")
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed")
    assert _ran(r) == ["test_planner.py"]


def test_a_new_untracked_test_file_runs(project):
    (project.path / "test_new.py").write_text("def test_new():\n    pass\n")
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed")
    assert _ran(r) == ["test_new.py"]


def test_a_changed_conftest_runs_everything(project):
    (project.path / "conftest.py").write_text("# a fixture would go here\n")
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed")
    assert _ran(r) == ["test_controller.py", "test_plain.py", "test_planner.py"]
    assert "so every test runs" in r.stdout.str()


def test_since_an_older_commit(project):
    (project.path / "controller.py").write_text("VALUE = 1  # tuned\n")
    _git(project.path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "tune")
    assert _ran(project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed")) == []
    r = project.runpytest("-p", "no:cacheprovider", "-v", "--rw-only-changed=HEAD~1")
    assert _ran(r) == ["test_controller.py"]


def test_a_bad_ref_is_a_usage_error(project):
    r = project.runpytest("-p", "no:cacheprovider", "--rw-only-changed=no-such-ref")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    assert "--rw-only-changed no-such-ref" in r.stderr.str()


def test_a_changed_plugin_module_runs_everything(project):
    (project.path / "myplugin.py").write_text("def pytest_configure(config):\n    pass\n")
    _git(project.path, "add", "-A")
    _git(project.path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "plugin")
    (project.path / "myplugin.py").write_text("def pytest_configure(config):\n    pass  # changed\n")
    project.syspathinsert()
    r = project.runpytest("-p", "no:cacheprovider", "-v", "-p", "myplugin", "--rw-only-changed")
    assert _ran(r) == ["test_controller.py", "test_plain.py", "test_planner.py"]
