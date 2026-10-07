"""Extending robowright from outside it: engine and robot plugins, a project's own robots, and
the contract they are held to (robowright.plugins, robowright.contract)."""

import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

import robowright as rw
from robowright import plugins, robots
from robowright.backends import base
from robowright.scene import default_scene


@pytest.fixture(autouse=True)
def _registry_as_found():
    before = dict(robots.REGISTRY)
    yield
    robots.REGISTRY.clear()
    robots.REGISTRY.update(before)


class _EntryPoint:
    """What importlib.metadata hands back for an installed package's entry point."""

    def __init__(self, name, obj):
        self.name, self.value, self._obj = name, f"fake_pkg:{name}", obj

    def load(self):
        if isinstance(self._obj, Exception):
            raise self._obj
        return self._obj


@pytest.fixture
def installed(monkeypatch):
    """Pretend packages are installed with these entry points: ``installed(group, name=obj, ...)``."""
    groups: dict = {}
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [_EntryPoint(n, o) for n, o in groups.get(group, {}).items()])
    plugins.plugin_robots.cache_clear()
    before = (dict(robots.REGISTRY), dict(base._REGISTRY))
    yield lambda group, **eps: groups.setdefault(group, {}).update(eps)
    plugins.plugin_robots.cache_clear()
    robots.REGISTRY.clear()
    robots.REGISTRY.update(before[0])
    base._REGISTRY.clear()
    base._REGISTRY.update(before[1])


def test_an_engine_from_another_package_runs_robot_tests(installed):
    """An installed engine is found by name, listed, and drives a pick like a built-in one."""
    from robowright.backends.mujoco_backend import MujocoBackend

    class Copycat(MujocoBackend):  # what a plugin's Backend subclass is
        reusable = False

    installed(plugins.BACKENDS, copycat=Copycat)
    assert "copycat" in base.available()
    with rw.launch(scene=default_scene("so101"), backend="copycat", settings=rw.Settings(trace="off")) as w:
        assert type(w.backend) is Copycat and w.backend.name == "copycat"
        w.robot.pick(w.scene["cube"])
        w.robot.place(on=w.scene["bin"])
        rw.expect(w.scene["cube"]).to_be_inside(w.scene["bin"])


def test_an_engine_plugin_that_fails_to_import_says_which(installed):
    installed(plugins.BACKENDS, broken=ModuleNotFoundError("No module named 'some_sdk'"))
    with pytest.raises(ImportError, match=r"backend plugin 'broken' \(fake_pkg:broken\) failed to load: No module named 'some_sdk'"):
        base.engine("broken")
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        base.engine("nope")


def test_robots_from_another_package_are_found_by_name(installed):
    """A robot package's entry point returns its models; they run under their own names."""
    so101 = robots.get("so101")
    installed(plugins.ROBOTS, lab=lambda: [replace(so101, name="lab_arm", title="Lab arm")])
    assert "lab_arm" in robots.names("arm")
    with rw.launch(scene=default_scene("lab_arm"), settings=rw.Settings(trace="off")) as w:
        assert w.robot.model.title == "Lab arm"
        w.robot.pick(w.scene["cube"])
        rw.expect(w.robot.gripper).to_be_holding(w.scene["cube"])


def test_a_broken_robot_plugin_is_reported_and_the_others_still_load(installed):
    so101 = robots.get("so101")
    installed(plugins.ROBOTS, bad=RuntimeError("mesh missing"), good=lambda: replace(so101, name="good_arm"), odd=lambda: ["not a robot"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert robots.get("good_arm").name == "good_arm"
    messages = [str(c.message) for c in caught]
    assert any("plugin 'bad'" in m and "mesh missing" in m for m in messages)
    assert any("plugin 'odd' returned str" in m for m in messages)


def _project(tmp_path, config: str, name="robowright.toml") -> Path:
    import shutil

    from robowright import assets

    shutil.copytree(assets.so101_mjcf().parent, tmp_path / "models" / "so101")
    (tmp_path / name).write_text(config)
    return tmp_path


def test_a_project_names_its_robots_in_robowright_toml(tmp_path, monkeypatch):
    """No code: a name, a model file relative to the settings, and fields robowright would
    otherwise work out (here its title, and its home height)."""
    _project(tmp_path, '[robots.bench_arm]\nfile = "models/so101/so101.xml"\ntitle = "Bench arm"\nhome = [0.2, 0.0, 0.1]\n')
    monkeypatch.chdir(tmp_path / "models")  # found from below, too
    assert plugins.project_robots() == {
        "bench_arm": (str(tmp_path / "models/so101/so101.xml"), {"title": "Bench arm", "home": (0.2, 0.0, 0.1)})
    }
    m = robots.get("bench_arm")
    assert (m.title, m.home) == ("Bench arm", (0.2, 0.0, 0.1))


def test_pyproject_can_hold_the_settings(tmp_path, monkeypatch):
    _project(tmp_path, '[project]\nname = "x"\n\n[tool.robowright.robots]\nbench_arm = "models/so101/so101.xml"\n', "pyproject.toml")
    monkeypatch.chdir(tmp_path)
    assert list(plugins.project_robots()) == ["bench_arm"]
    # Adding one goes there, not into a robowright.toml that would hide these.
    with pytest.raises(ValueError, match=r"add \[tool.robowright.robots.other\] there"):
        plugins.add_project_robot("models/so101/so101.xml", "other")


def test_robots_add_writes_the_entry(tmp_path, monkeypatch):
    _project(tmp_path, "# this project's robots\n")
    monkeypatch.chdir(tmp_path)
    target = plugins.add_project_robot(str(tmp_path / "models/so101/so101.xml"), "arm_a")
    plugins.add_project_robot("models/so101/so101.xml?x=1", "arm_b")
    assert target == tmp_path / "robowright.toml"
    assert target.read_text() == (
        "# this project's robots\n"
        '\n[robots.arm_a]\nfile = "models/so101/so101.xml"\n'
        '\n[robots.arm_b]\nfile = "models/so101/so101.xml?x=1"\n'
    )
    with pytest.raises(ValueError, match="already names a robot 'arm_a'"):
        plugins.add_project_robot("models/so101/so101.xml", "arm_a")
    with pytest.raises(ValueError, match="letters, digits"):
        plugins.add_project_robot("models/so101/so101.xml", "my arm")


def test_check_runs_the_contract_on_a_project_robot(tmp_path):
    """``robowright check`` runs the shipped contract from any directory, on any robot."""
    _project(tmp_path, '[robots.bench_arm]\nfile = "models/so101/so101.xml"\n')
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "robowright.cli",
            "check",
            "--robot",
            "bench_arm",
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "reports_the_robot_joints",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "1 passed" in out.stdout


def test_an_installed_package_is_discovered_through_its_entry_points(tmp_path):
    """No fakes: a package installed the way pip installs one (a dist-info with entry points)."""
    site = tmp_path / "site"
    (site / "acme_robots").mkdir(parents=True)
    (site / "acme_robots" / "__init__.py").write_text(
        "from dataclasses import replace\n"
        "from robowright import robots as _r\n"
        "def robots():\n"
        "    return [replace(_r.get('so101'), name='acme_r1', title='Acme R1')]\n"
    )
    info = site / "acme_robots-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: acme-robots\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text("[robowright.robots]\nacme = acme_robots:robots\n")
    env = {**__import__("os").environ, "PYTHONPATH": str(site)}
    out = subprocess.run(
        [sys.executable, "-m", "robowright.cli", "robots", "--family", "arm"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "acme_r1" in out.stdout and "Acme R1" in out.stdout
