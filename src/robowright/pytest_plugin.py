"""pytest integration.

Fixtures: ``world``, ``robot``, ``scene`` (fresh per test), ``rw_scene``
(override in a conftest to change the default scene).

Options::

    --rw-backend mujoco,pybullet     run every robot test on each backend
    --rw-robot so101,panda | all | legged   run every robot test on each robot ("all": every arm)
    --rw-robot path/to/robot.xml             ...or on any robot, from its model file (MJCF or URDF)
    --rw-trace on|off|retain-on-failure|on-first-retry
    --rw-retries N                   run a failed test again, up to N times; one that then passes is reported flaky
    --rw-fail-on-flaky               a flaky test fails the run
    --rw-repeat-each N               run every test N times (the same seed each time), to find the ones that only fail sometimes
    --rw-trace-dir DIR
    --rw-video on|off|retain-on-failure   each test's run as an MP4 beside its trace (needs robowright[video])
    --rw-seed N                      base seed
    --rw-all-trials                  run every trial of a trials test, not just until the verdict is settled
    --rw-trace-text                  print each failing trace as text (what ran, what failed, the state then)
    --rw-update-snapshots            write the images expect(world).to_match_snapshot compares against
    --rw-shard K/N                   run the K-th of N parts of the suite, one part per CI machine
    --rw-github auto|on|off          failures as GitHub Actions annotations on the failing line (auto: on a GitHub runner)
    --rw-only-changed [REF]          only the test files a change since REF (default HEAD) can affect, found from git
    --rw-timeout SECONDS             fail a test that runs longer than this, wall clock (0: no limit)
    --rw-report report.html          an HTML report of the run, each failure with its trace
    --rw-blob shard-2.zip            the run's results and traces in one zip, for robowright merge-reports
    --rw-fidelity published|adjusted robots as their model files have them (default), or with robowright's tuning
    --rw-live [PORT]                 watch each test live in a browser (http://127.0.0.1:8765)
    --rw-headed                      watch each test in MuJoCo's window as it runs, at real-time pace
    --rw-debug                       the live view as a debugger: pause before each test's first action, then step
    -n auto                          (pytest-xdist) as many workers as the cores and memory allow

Markers::

    @pytest.mark.scene(spec_or_factory)
    @pytest.mark.seed(7)
    @pytest.mark.backends("mujoco")              # restrict
    @pytest.mark.robots("panda", "ur5e")         # restrict
    @pytest.mark.trials(20, min_success=0.9)     # up to 20 seeds, each with the scene randomized; judge the rate
"""

from __future__ import annotations

import os
import re
import warnings
import zlib
from pathlib import Path

import pytest

import robowright as rw
from robowright import fidelity

from . import robots as _robots
from .errors import RobowrightError, TooWideError
from .scene import SceneSpec, default_camera, default_scene
from .stats import TrialReport
from .world import Settings, World, _safe

# The contract ships in the package (robowright check runs it): its asserts report their values too.
pytest.register_assert_rewrite("robowright.contract")

_TRACES = pytest.StashKey[list]()
_REPORTS = pytest.StashKey[dict]()
_LIVE = pytest.StashKey[object]()
_CHANGED = pytest.StashKey[object]()  # --rw-only-changed: the test files to run, or None for all
_STACK = pytest.StashKey[object]()  # this worker's ROS 2 stack ([ros2] stack), or None


def pytest_addoption(parser):
    g = parser.getgroup("robowright")
    g.addoption("--rw-backend", default="mujoco", help="comma-separated backends to run robot tests on")
    g.addoption(
        "--rw-robot", default="so101", help="robots to run robot tests on (comma-separated): names, 'all', 'legged', or MJCF/URDF files"
    )
    g.addoption(
        "--rw-trace",
        default="retain-on-failure",
        choices=["on", "off", "retain-on-failure", "on-first-retry"],
        help="on-first-retry: record only the first retry of a failed test (with --rw-retries), as Playwright's CI default",
    )
    g.addoption(
        "--rw-retries",
        type=int,
        default=0,
        metavar="N",
        help="run a failed test again, up to N times, in a fresh world; a test that fails and then passes is reported "
        "flaky (not passed), with the failed attempt's trace kept",
    )
    g.addoption(
        "--rw-fail-on-flaky",
        action="store_true",
        help="fail the run if any test was flaky (failed, then passed on a retry), as Playwright's --fail-on-flaky-tests",
    )
    g.addoption(
        "--rw-repeat-each",
        type=int,
        default=1,
        metavar="N",
        help="run every test N times, each with the same seed and a fresh world, as Playwright's --repeat-each: a "
        "test whose repeats disagree depends on something besides its seed (wall-clock timing, a simulator's threads)",
    )
    g.addoption(
        "--rw-video",
        choices=["off", "on", "retain-on-failure"],
        default="off",
        help="each test's run as an MP4 beside its trace, drawn from it after the test (Playwright's video option): "
        "every test, or only those that failed; it records the trace it is drawn from. Needs robowright[video]",
    )
    g.addoption("--rw-trace-dir", default="robowright-traces")
    g.addoption("--rw-seed", type=int, default=0, help="base seed added to each test's seed")
    g.addoption(
        "--rw-all-trials",
        action="store_true",
        help="run all n trials of a trials test; by default it stops once the rest cannot change the verdict",
    )
    g.addoption(
        "--rw-fidelity",
        choices=["published", "adjusted"],
        default=None,
        help="published (default): robots as their model files have them, plus cited specs; adjusted: also robowright's "
        "own tuning (stiffer contacts, capped grippers, ...), which passes more tests on physics nobody measured",
    )
    g.addoption(
        "--rw-live",
        nargs="?",
        type=int,
        const=8765,
        default=None,
        metavar="PORT",
        help="watch each test live in a browser at http://127.0.0.1:PORT (8765; xdist workers count up), at real-time pace",
    )
    g.addoption(
        "--rw-debug",
        action="store_true",
        help="debug in the live view (Playwright's --debug): each test pauses before its first action or expectation; "
        "Step runs one and pauses again, Resume runs on. Implies --rw-live and no --rw-timeout",
    )
    g.addoption("--rw-headed", action="store_true", help="watch each test in the engine's window, at real-time pace (MuJoCo)")
    g.addoption("--rw-report", metavar="PATH", help="write an HTML report of the run: every test, and each failure with its trace")
    g.addoption(
        "--rw-blob",
        metavar="PATH",
        help="write this run's results, traces included, to one zip, for robowright merge-reports (a sharded CI run)",
    )
    g.addoption(
        "--rw-timeout",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="fail a test that runs longer than this, wall clock, setup included (0: no limit); @pytest.mark.rw_timeout(s) sets one test's",
    )
    g.addoption(
        "--rw-shard",
        metavar="K/N",
        help="run the K-th of N parts of the suite (1-based), for N CI machines; every test is in exactly one part",
    )
    g.addoption(
        "--rw-github",
        choices=["auto", "on", "off"],
        default="auto",
        help="each failed test as a GitHub Actions error annotation at the line that failed, each flaky one a warning "
        "(Playwright's github reporter); auto: when GITHUB_ACTIONS=true",
    )
    g.addoption(
        "--rw-only-changed",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="REF",
        help="run only the test files that changes since the git commit REF (default HEAD: uncommitted changes) can "
        "affect: changed tests, and tests importing a changed module; a changed conftest.py or setting runs every test",
    )
    g.addoption(
        "--rw-update-snapshots",
        action="store_true",
        help="write every image expect(world).to_match_snapshot compares against, from this run",
    )
    g.addoption(
        "--rw-trace-text",
        action="store_true",
        help="print each failing trace as text at the end: what ran, what failed, the state then (for CI logs and AI agents)",
    )


def pytest_configure(config):
    if config.getoption("--rw-robot", "so101") != "so101":
        # Named here, not when a test first asks for the robot: a test that builds its own world never
        # does, and a misspelt name then ran nothing on the robot meant and passed.
        try:
            _robot_names(config)
        except (ValueError, KeyError) as e:
            raise pytest.UsageError(f"--rw-robot: {e}") from None
    _shard(config)  # a malformed --rw-shard is said before anything runs
    if config.getoption("--rw-debug", False) and config.getoption("--rw-live", None) is None:
        config.option.rw_live = 8765
    ref = config.getoption("--rw-only-changed", None)
    if ref:
        from . import changed

        try:
            files = changed.changed_files(ref, config.rootpath)
        except changed.GitError as e:
            raise pytest.UsageError(f"--rw-only-changed {ref}: {e}") from None
        plugins = {Path(f).resolve() for m in config.pluginmanager.get_plugins() if (f := getattr(m, "__file__", None))}
        config.stash[_CHANGED] = changed.affected(files, config.rootpath, plugins)
    mode = config.getoption("--rw-fidelity", None)
    if mode:
        _set_env(config, fidelity.ENV, mode)
    if config.getoption("--rw-update-snapshots", False):
        from . import snapshots

        _set_env(config, snapshots.UPDATE_ENV, "1")
    config.addinivalue_line("markers", "scene(spec): scene spec or zero-arg factory for this test")
    config.addinivalue_line("markers", "seed(n): seed for this test")
    config.addinivalue_line("markers", "backends(*names): only run on these backends")
    config.addinivalue_line("markers", "robots(*names): only run on these robots")
    config.addinivalue_line(
        "markers",
        "trials(n, min_success=1.0, lower_bound=False, randomize=True): run across n seeds, each with its free objects "
        "moved to a seeded random pose, and judge the success rate",
    )
    config.addinivalue_line("markers", "xdist_group(name): pytest-xdist's grouping, set per robot and engine")
    config.addinivalue_line("markers", "rw_timeout(seconds): this test's time limit, wall clock, setup included (0: none)")
    config.addinivalue_line("markers", "rw_retries(n): run this test again up to n times if it fails (over --rw-retries)")
    config.stash[_TRACES] = []
    config.stash[_REPORTS] = {}
    from . import github

    summary = _Summary()
    summary.config = config
    summary.github = github.enabled(config.getoption("--rw-github", "auto"))  # decided once: a test may set the variable
    config.pluginmanager.register(summary, "robowright-summary")
    if any(_robots.is_file(r.strip()) for r in config.getoption("--rw-robot").split(",")):
        _robot_names(config)  # a robot model file robowright cannot drive is a usage error, said once
    if getattr(config.option, "dist", None) == "loadgroup" and "--loadscope-reorder" not in config.invocation_params.args:
        # xdist hands out the largest groups first, which puts a lone trials test, the longest
        # kind, at the end of the run; robowright's own order (pytest_collection_modifyitems)
        # starts those first.
        config.option.loadscopereorder = False


def pytest_report_header(config):
    ref = config.getoption("--rw-only-changed", None)
    if not ref:
        return None
    wanted = config.stash.get(_CHANGED, None)
    if wanted is None:
        return f"robowright: --rw-only-changed {ref}: a conftest.py, plugin, setting or model file changed, so every test runs"
    names = sorted(os.path.relpath(p, config.rootpath) for p in wanted)
    return f"robowright: --rw-only-changed {ref}: {len(names)} test file(s) affected" + "".join(f"\n    {n}" for n in names)


def _shard(config) -> tuple[int, int] | None:
    """``--rw-shard K/N``, Playwright's ``--shard``: tests are dealt out by a hash of their id, so every
    machine that collects the same suite gets the same split without talking to the others."""
    s = config.getoption("--rw-shard", None)
    if not s:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", s)
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise pytest.UsageError(f"--rw-shard {s!r}: give K/N with 1 <= K <= N, e.g. 2/4")
    return int(m.group(1)), int(m.group(2))


def _set_env(config, name: str, value: str) -> None:
    """For this run, and the xdist workers it starts; put back after, as a run inside another
    (pytester, in-process) would otherwise leave every later test with it set."""
    previous = os.environ.get(name)
    os.environ[name] = value

    def restore():
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous

    config.add_cleanup(restore)


@pytest.hookimpl(optionalhook=True, tryfirst=True)
def pytest_xdist_auto_num_workers(config):
    """``-n auto``: workers by the cores and memory the selected engines need (see :mod:`robowright.workers`)."""
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None  # xdist's own hook reads it
    from .workers import auto_workers

    return auto_workers([b.strip() for b in config.getoption("--rw-backend").split(",") if b.strip()])


STACK_ROBOT = "stack"  # --rw-robot stack: the robot the ROS 2 stack's own description publishes


def _robot_names(config) -> list[str]:
    raw = [r.strip() for r in config.getoption("--rw-robot").split(",") if r.strip()]
    if raw == ["so101"] and config.getoption("--rw-backend", "") == "ros2" and _stack_configured():
        return [STACK_ROBOT]  # no robot named, against the project's own stack: its robot
    out = []
    for r in raw:
        if r == STACK_ROBOT:
            out.append(r)
        elif r in ("all", "arms", "legged"):
            out.extend(_robots.names({"all": "arm", "arms": "arm", "legged": "legged"}[r]))
        elif _robots.is_file(r):  # a model file: tests get its absolute path, so traces replay from anywhere
            from pathlib import Path

            file, _, query = r.partition("?")  # arm.urdf.xacro?ur_type=ur5e: xacro arguments
            out.append(str(Path(file).expanduser().resolve()) + (f"?{query}" if query else ""))
            from .robots.detect import DetectionError

            try:
                _robots.get(out[-1])
            except (DetectionError, FileNotFoundError) as e:
                raise pytest.UsageError(f"--rw-robot {r}: robowright cannot drive this robot: {e}") from None
        else:
            out.append(_robots.get(r).name)
    return list(dict.fromkeys(out))


def pytest_generate_tests(metafunc):
    if "rw_backend" in metafunc.fixturenames:
        names = [b.strip() for b in metafunc.config.getoption("--rw-backend").split(",") if b.strip()]
        m = metafunc.definition.get_closest_marker("backends")
        if m:
            names = [n for n in names if n in m.args] or list(m.args[:1])
        metafunc.parametrize("rw_backend", names, ids=names, scope="function")
    if "rw_robot" in metafunc.fixturenames:
        names = _robot_names(metafunc.config)
        m = metafunc.definition.get_closest_marker("robots")
        if m:
            names = [n for n in names if n in m.args] or list(m.args[:1])
        if len(names) > 1 or m:
            metafunc.parametrize("rw_robot", names, ids=[_robot_id(n) for n in names], scope="function")
    repeats = metafunc.config.getoption("--rw-repeat-each", 1)
    if repeats > 1:  # last, so the id reads test[mujoco-repeat2]
        metafunc.fixturenames.append("rw_repeat")
        metafunc.parametrize("rw_repeat", range(1, repeats + 1), ids=[f"repeat{i}" for i in range(1, repeats + 1)], scope="function")


def pytest_sessionfinish(session, exitstatus):
    """A shard that the hash dealt no tests (a small suite over many machines) has passed, as Playwright's does: pytest's
    "no tests collected" (5) would fail that CI job for having nothing to do; so has --rw-only-changed with no change.
    With ``--rw-fail-on-flaky`` a flaky test fails a run that otherwise passed."""
    nothing_to_do = _shard(session.config) is not None or session.config.stash.get(_CHANGED, None) is not None
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and nothing_to_do:
        session.exitstatus = pytest.ExitCode.OK
    summary = session.config.pluginmanager.get_plugin("robowright-summary")
    if session.config.getoption("--rw-fail-on-flaky", False) and summary is not None and summary.flaky:
        if session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(tryfirst=True)  # before xdist reads the groups into node ids
def pytest_collection_modifyitems(config, items):
    """Run each module's tests robot by robot, and group them for ``-n N --dist loadgroup``.

    A closed world's scene is kept and restored for the next world that needs it (see
    ``backends.create``), which runs bit-for-bit like a new build, so the order cannot change a
    result. Ordered and grouped, the tests that share a scene run one after another in one
    process and reuse it, instead of every robot's scene being built for every test.
    """
    wanted = config.stash.get(_CHANGED, None)
    if wanted is not None:
        dropped = [i for i in items if Path(i.path).resolve() not in wanted]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = [i for i in items if Path(i.path).resolve() in wanted]
    shard = _shard(config)
    if shard is not None:
        k, n = shard
        keep = [i for i in items if zlib.crc32(_plain(i.nodeid).encode()) % n == k - 1]
        dropped = [i for i in items if zlib.crc32(_plain(i.nodeid).encode()) % n != k - 1]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
        items[:] = keep
    module = {}

    def key(item):
        p = getattr(item, "callspec", None)
        params = p.params if p is not None else {}
        # Trials tests first: they are the longest, and handed out last they were the tail of
        # every parallel run.
        first = item.get_closest_marker("trials") is None
        return (first, module.setdefault(item.path, len(module)), params.get("rw_backend", ""), params.get("rw_robot", ""))

    items[:] = sorted(items, key=key)
    for item in items:
        p = getattr(item, "callspec", None)
        if p is not None and "rw_backend" in p.params:
            robot = p.params.get("rw_robot", "")
            group = f"{p.params['rw_backend']}-{_robot_id(robot) if robot else ''}-{item.path.stem}"
            if item.get_closest_marker("trials"):
                # A trials test builds a world per trial, so it shares little with its robot's
                # other tests, and it is the longest: on its own it can start while they run.
                group += f"-{item.originalname}"
            item.add_marker(pytest.mark.xdist_group(group))


@pytest.fixture
def rw_robot(request) -> str:  # parametrized by pytest_generate_tests when several robots are selected
    return _robot_names(request.config)[0]


@pytest.fixture
def rw_scene(rw_robot) -> SceneSpec:
    if rw_robot == STACK_ROBOT:  # a robot robowright knows only from the stack: no objects placed for its reach
        return SceneSpec(robot=None, objects=[], cameras=[])
    return default_scene(rw_robot)


def _scene_for(item, request) -> SceneSpec:
    m = item.get_closest_marker("scene")
    if m:
        s = m.args[0]
        spec = s() if callable(s) else s
    else:
        spec = request.getfixturevalue("rw_scene")
    robot = request.getfixturevalue("rw_robot")
    if robot == STACK_ROBOT:
        import dataclasses

        return dataclasses.replace(spec, robot=None, cameras=[c for c in spec.cameras if spec.robot is None])
    if spec.robot != robot:
        import dataclasses

        cams = [default_camera(robot)] if spec.cameras == [default_camera(spec.robot)] else spec.cameras
        spec = dataclasses.replace(spec, robot=robot, cameras=cams)
    return spec


def _settings(config, attempt: int = 0) -> Settings:
    headed = config.getoption("--rw-headed")
    trace = config.getoption("--rw-trace")
    if trace == "on-first-retry":  # the first run of a test is not recorded; its first retry is, whatever its outcome
        trace = "on" if attempt == 1 else "off"
    video = config.getoption("--rw-video", "off")
    if video == "on" or (video == "retain-on-failure" and trace == "off"):  # a video is drawn from the trace
        trace = "on" if video == "on" else "retain-on-failure"
    if config.getoption("--rw-live", None) is not None:  # watchable: at real-time pace
        return Settings(trace=trace, trace_dir=config.getoption("--rw-trace-dir"), headed=headed, realtime=True)
    return Settings(trace=trace, trace_dir=config.getoption("--rw-trace-dir"), headed=headed)


def _plain(nodeid: str) -> str:
    """``nodeid`` without the ``@group`` that ``--dist loadgroup`` appends to it."""
    return re.sub(r"@[\w.-]+$", "", nodeid)


def _trace_path(config, nodeid: str, attempt: int = 0) -> Path:
    """Where a test's trace goes; each retry has its own, so the failed attempt's trace survives the retry."""
    retry = f"-retry{attempt}" if attempt else ""
    return Path(config.getoption("--rw-trace-dir")) / f"{_safe(_plain(nodeid).replace('::', '__').replace('/', '_'))}{retry}.zip"


def _attempt(item) -> int:
    return getattr(item, "_rw_attempt", 0)


def _make_world(item, request, backend: str, seed: int, suffix: str = "") -> World:
    if backend != "ros2":
        rw.world.check_mujoco()
    stack = _ros2_stack(item.config) if backend == "ros2" else None
    reset = None
    if stack is not None:
        before = stack.restarts
        reset = {"seconds": round(stack.reset(), 3), "restarted": stack.restarts > before, "parts": dict(stack.last_reset)}
    settings = _settings(item.config, _attempt(item))
    scene = _scene_for(item, request)
    if backend == "ros2" and scene.robot is None:  # the robot the stack's own description publishes, as rw.launch takes it
        scene = rw.world.ros2_scene(scene, backend, None, {}, None)
    w = World(scene, backend=backend, seed=seed, name=_plain(item.nodeid) + suffix, settings=settings)
    if reset is not None and w.trace:
        w.trace.event("edit", "ros2_stack_reset", reset)
    rw._current.set(w)
    view = _live(item.config)
    if view is not None:
        view.watch(w)
        if item.config.getoption("--rw-debug", False):
            view.stepping = True  # held before the test's first action
    return w


def _robot_id(name: str) -> str:
    return name if name == STACK_ROBOT else _robots.get(name).name


def _stack_configured() -> bool:
    try:
        from .backends.ros2_backend import settings as ros2_settings

        return bool(ros2_settings().get("stack"))
    except Exception:
        return False


def _ros2_stack(config):
    """This worker's ROS 2 stack (``[ros2] stack``), started on first use and stopped when the run ends; None when
    the project names none (the stack is then expected to be running)."""
    if _STACK not in config.stash:
        from .backends.ros2_backend import settings as ros2_settings
        from .stack import Ros2Stack

        spec = ros2_settings().get("stack")
        stack = None
        if spec:
            spec = {"command": spec} if isinstance(spec, (str, list)) else dict(spec)
            stack = Ros2Stack(spec.pop("command"), **spec)
            config.add_cleanup(stack.stop)
            stack.start()
        config.stash[_STACK] = stack
    return config.stash[_STACK]


def _live(config):
    """This process's live view (--rw-live), started on first use: one port per xdist worker."""
    port = config.getoption("--rw-live", None)
    if port is None:
        return None
    if _LIVE not in config.stash:
        from .live import LiveView

        worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")
        view = config.stash[_LIVE] = LiveView(port=port + int(worker) if port else 0, quiet=True)
        view.announce = lambda msg: _say(config, msg)
        view.announce(f"robowright: watch live at {view.url}")
        config.add_cleanup(view.close)
    return config.stash[_LIVE]


def _seed(item) -> int:
    m = item.get_closest_marker("seed")
    return item.config.getoption("--rw-seed") + (m.args[0] if m else 0)


@pytest.fixture
def rw_backend():  # parametrized by pytest_generate_tests
    return "mujoco"


@pytest.fixture
def rw_repeat() -> int:  # which run of --rw-repeat-each this is (1-based); parametrized by pytest_generate_tests
    return 1


@pytest.fixture
def world(request, rw_backend, rw_robot):
    item = request.node
    if item.get_closest_marker("trials"):
        yield None  # pytest_pyfunc_call builds one world per trial
        return
    w = _make_world(item, request, rw_backend, _seed(item))
    if not (rw_backend == "ros2" and _ros2_stack(item.config) is not None):
        # With a stack of the project's own, a test starts where the stack's reset put it (where the stack came up):
        # driving on to robowright's home pose as well moved the arm twice, slowly, before every test.
        w.robot.reset_to()
    yield w
    rep = getattr(item, "_rw_call_report", None)
    failed = rep is None or rep.failed
    if failed and w.trace and not any(e.status == "failed" for e in w.trace.events):
        # The test's own code raised (a name that is not in the scene, an assert of its own): the trace says what.
        crash = getattr(getattr(rep, "longrepr", None), "reprcrash", None)
        message = crash.message if crash is not None else "the test failed before it ran (in a fixture)"
        w.trace.event("error", message.splitlines()[0][:300], status="failed", detail=message)
    path = w.close(failed=failed, trace_path=_trace_path(item.config, item.nodeid, _attempt(item)))
    if path and failed:
        item.config.stash[_TRACES].append((_plain(item.nodeid), str(path)))
    video = item.config.getoption("--rw-video", "off")
    if path and (video == "on" or (video == "retain-on-failure" and failed)):
        _video(item, path)
        mode = item.config.getoption("--rw-trace")
        if not failed and mode != "on" and not (mode == "on-first-retry" and _attempt(item) == 1):
            Path(path).unlink(missing_ok=True)  # recorded only to draw the video from


@pytest.fixture
def ros2(world):
    """This test's world served as a ROS 2 robot, for a ROS 2 stack (yours) to drive while the test
    watches: ``with ros2.run("ros2 launch ..."): expect(...)``. See robowright.ros2_bridge."""
    pytest.importorskip("rclpy", reason="the ros2 fixture needs a ROS 2 environment (rclpy)")
    from .ros2_bridge import Ros2Bridge

    bridge = Ros2Bridge(world)
    yield bridge
    bridge.close()


class _Attacher:
    """``rw_attach``, Playwright's ``testInfo.attach``: files this test adds to the HTML report
    (``--rw-report``), kept beside the traces. ``rw_attach("log.txt", text)``,
    ``rw_attach("bag", path="run.mcap")``, ``rw_attach("grasp.png", image_array)``, or
    ``rw_attach.camera(world)`` for what a camera sees now."""

    def __init__(self, item):
        self.item = item
        self.dir = (
            Path(item.config.getoption("--rw-trace-dir")) / "attachments" / _safe(_plain(item.nodeid).replace("::", "__").replace("/", "_"))
        )

    def __call__(self, name: str, body=None, *, path=None, content_type: str | None = None) -> Path:
        import json
        import mimetypes
        import shutil

        import numpy as np

        if (body is None) == (path is None):
            raise ValueError("rw_attach takes a body or a path=, not both and not neither")
        self.dir.mkdir(parents=True, exist_ok=True)
        dst = self.dir / _safe(name)
        if path is not None:
            shutil.copyfile(path, dst)
            kind = content_type or mimetypes.guess_type(str(path))[0]
        elif isinstance(body, np.ndarray):  # an image (H, W, 3)
            from PIL import Image

            dst = dst if dst.suffix else dst.with_suffix(".png")
            Image.fromarray(np.ascontiguousarray(body[..., :3]).astype(np.uint8)).save(dst)
            kind = "image/png"
        else:
            dst.write_bytes(body.encode() if isinstance(body, str) else bytes(body))
            kind = content_type or mimetypes.guess_type(str(dst))[0] or ("text/plain" if isinstance(body, str) else None)
        record = {"name": name, "path": str(dst), "type": kind or "application/octet-stream"}
        self.item.user_properties.append(("robowright_attachment", json.dumps(record)))
        return dst

    def camera(self, world, camera: str = "front", size: tuple[int, int] = (640, 480), name: str | None = None) -> Path:
        """Attach what ``camera`` sees now."""
        world.require("render", "rw_attach.camera")
        return self(name or f"{camera}-{world.step_count}.png", world.backend.render(camera, *size))


@pytest.fixture
def rw_attach(request):
    """Add files to this test's entry in the HTML report: see :class:`_Attacher`."""
    return _Attacher(request.node)


@pytest.fixture
def rw_live(request):
    """The live view (``--rw-live``), or None: ``rw_live.url`` is where to watch."""
    return _live(request.config)


@pytest.fixture
def seed(request) -> int:
    """This test's seed (each trial gets its own under ``@pytest.mark.trials``)."""
    return _seed(request.node)


@pytest.fixture
def robot(world):
    return world.robot if world is not None else None


@pytest.fixture
def scene(world):
    return world.scene if world is not None else None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item._rw_call_report = rep
        retry = _attempt(item) > 0  # a retry's trace is kept whatever it shows (on-first-retry records it)
        if (rep.failed or retry) and "world" in getattr(item, "fixturenames", ()) and not item.get_closest_marker("trials"):
            rep.user_properties.append(("robowright_trace", str(_trace_path(item.config, item.nodeid, _attempt(item)))))
        if hasattr(item, "_rw_trials"):
            rep.user_properties.append(("robowright_trials", item._rw_trials.summary()))


def _video(item, trace_path) -> None:
    """``--rw-video``: the test's run, drawn from its trace, as an MP4 beside it and in its report entry."""
    import json
    import warnings

    from .render import render_video

    out = Path(trace_path).with_suffix(".mp4")
    try:
        render_video(trace_path, out, size=(640, 480))
    except ImportError:
        warnings.warn(pytest.PytestWarning('--rw-video needs robowright[video] (pip install "robowright[video]")'), stacklevel=1)
        return
    except Exception as e:  # noqa: BLE001 - no GL, or a trace without a camera: the test's own result stands
        warnings.warn(pytest.PytestWarning(f"--rw-video: {item.nodeid}: no video ({type(e).__name__}: {e})"), stacklevel=1)
        return
    record = {"name": out.name, "path": str(out), "type": "video/mp4"}
    item.user_properties.append(("robowright_attachment", json.dumps(record)))


def _say(config, msg: str) -> None:
    """``msg`` on the terminal now, past pytest's output capture (the live view's address, the debugger's pauses)."""
    capman = config.pluginmanager.getplugin("capturemanager")
    tr = config.pluginmanager.getplugin("terminalreporter")
    if capman is None or tr is None:
        print(msg, flush=True)
        return
    with capman.global_and_fixture_disabled():
        tr.write_line(msg)
        tr._tw.flush()  # a pipe (an IDE, CI) would otherwise get it only at the end


def _time_limit(item) -> float:
    if item.config.getoption("--rw-debug", False):  # a person is stepping through it
        return 0.0
    m = item.get_closest_marker("rw_timeout")
    return float(m.args[0]) if m else float(item.config.getoption("--rw-timeout", 0.0) or 0.0)


HANG_GRACE = 60.0  # seconds past --rw-timeout after which a test stuck in a C call ends the process (see below);
# ROBOWRIGHT_HANG_GRACE overrides it (robowright's own test of it does not wait a minute)


def _other_threads(limit: int = 6) -> str:
    """Where the other threads are, for a time limit's error: what the test waits on is often one of them."""
    import sys
    import threading
    import traceback

    names = {t.ident: t.name for t in threading.enumerate()}
    out = []
    for ident, frame in sys._current_frames().items():
        if ident == threading.main_thread().ident:
            continue
        stack = traceback.extract_stack(frame)[-limit:]
        out.append(f"  thread {names.get(ident, ident)}:\n" + "".join(f"    {s.filename}:{s.lineno} in {s.name}\n" for s in stack))
    return "other threads:\n" + "".join(out) if out else ""


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    """Start the test's clock (``--rw-timeout``): from its fixtures, where a stack that never comes up
    waits, to the end of its body. Checked between Python operations, which a long C call delays.

    A main thread stuck inside a C call never gets back to Python to see the limit pass (a ROS 2 node's destroy that
    waits for ever, a join on a thread blocked in C). For that, faulthandler's own watchdog, which runs in C: if the test
    is still running HANG_GRACE seconds after its limit, every thread's stack goes to ``hung-<test>.txt`` in the trace
    directory and the process ends, so a hang costs its limit and a minute, not the rest of the CI job."""
    limit = _time_limit(item)
    if limit > 0:
        import faulthandler
        import signal
        import threading

        name = _safe(_plain(item.nodeid).replace("::", "__").replace("/", "_"))
        hung = Path(item.config.getoption("--rw-trace-dir")) / f"hung-{name}.txt"
        try:
            hung.parent.mkdir(parents=True, exist_ok=True)
            item._rw_hung = hung.open("w")
            grace = float(os.environ.get("ROBOWRIGHT_HANG_GRACE", HANG_GRACE))
            faulthandler.dump_traceback_later(limit + grace, exit=True, file=item._rw_hung)
        except OSError:
            pass

        if hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread():

            def expired(signum, frame):
                from . import _current
                from .errors import TimeLimitError

                w = _current.get()
                tr = getattr(w, "trace", None) if w is not None else None
                running = [e for e in (tr.events if tr else []) if e.status == "running"]
                where = f", while {running[-1].type} {running[-1].name!r} was running" if running else ""
                raise TimeLimitError(f"the test ran past its {limit:g} s time limit (wall clock){where}\n{_other_threads()}")

            item._rw_alarm = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, limit)
        else:
            warnings.warn(pytest.PytestWarning(f"{item.nodeid}: --rw-timeout needs SIGALRM in the main thread; no time limit"))
    return (yield)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """The clock runs through teardown too: a fixture that never finishes closing (a ROS 2 executor waiting
    for a thread, as one did) fails the test instead of hanging the run."""
    try:
        return (yield)
    finally:
        if hasattr(item, "_rw_alarm"):
            import signal

            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, item.__dict__.pop("_rw_alarm"))
        f = item.__dict__.pop("_rw_hung", None)
        if f is not None:
            import faulthandler

            faulthandler.cancel_dump_traceback_later()
            f.close()
            Path(f.name).unlink(missing_ok=True)  # it did not hang: nothing to keep


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """A test asking a robot to pick up something wider than its gripper opens is skipped on that
    robot: the robot cannot do it, which is not a failure of the robot or of robowright."""
    try:
        return (yield)
    except TooWideError as e:
        pytest.skip(f"{e}: too wide for this robot's gripper")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """``--rw-retries N``: a test that fails runs again, up to N times, each in a fresh world. The failed
    attempts are reported as reruns, and a test that then passes as flaky, as Playwright reports it.
    A real robot or simulator can fail once for reasons outside the code under test (a stalled clock,
    a slow start), so a retry tells "fails" from "fails sometimes"; it never turns a failure into a pass."""
    m = item.get_closest_marker("rw_retries")  # one test's own (Playwright's per-test retries), over --rw-retries
    retries = int(m.args[0]) if m else item.config.getoption("--rw-retries", 0)
    if not retries:
        return None
    from _pytest.runner import runtestprotocol

    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    for attempt in range(retries + 1):
        item._rw_attempt = attempt
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
        if any(r.failed for r in reports) and attempt < retries:
            for r in reports:
                if r.failed:
                    r.outcome = "rerun"
                item.ihook.pytest_runtest_logreport(report=r)
            _forget_failed_fixtures(item)
            item._initrequest()  # fresh fixtures (a new world) for the next attempt
            continue
        for r in reports:
            if attempt and r.when == "call" and r.passed:
                r.user_properties.append(("robowright_flaky", attempt))
            item.ihook.pytest_runtest_logreport(report=r)
        break
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def _forget_failed_fixtures(item) -> None:
    """A fixture whose setup raised keeps the error cached for its scope; the retry sets it up again."""
    for defs in item._fixtureinfo.name2fixturedefs.values():
        for d in defs:
            cached = getattr(d, "cached_result", None)
            if cached is not None and cached[2] is not None:
                d.cached_result = None


def pytest_report_teststatus(report, config):
    if report.outcome == "rerun":
        return "rerun", "R", ("RERUN", {"yellow": True})
    if report.when == "call" and report.passed and any(k == "robowright_flaky" for k, _ in report.user_properties):
        return "flaky", "f", ("FLAKY", {"yellow": True})
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    m = pyfuncitem.get_closest_marker("trials")
    if m is None:
        return None
    n = m.args[0] if m.args else m.kwargs.get("n", 10)
    randomize = m.kwargs.get("randomize", True)
    report = TrialReport(_plain(pyfuncitem.nodeid), n, 0, m.kwargs.get("min_success", 1.0), m.kwargs.get("lower_bound", False))
    random_runs = 0
    request = pyfuncitem._request
    backend = pyfuncitem.funcargs.get("rw_backend", "mujoco")
    argnames = pyfuncitem._fixtureinfo.argnames
    base = _seed(pyfuncitem)
    every = pyfuncitem.config.getoption("--rw-all-trials")
    for i in range(n):
        w = _make_world(pyfuncitem, request, backend, base + i, f"[trial {i}]")
        w.robot.reset_to()
        if randomize:
            w.faults.randomize_scene()  # without it, every seed would run the same scene
        args = {k: pyfuncitem.funcargs[k] for k in argnames}
        args.update({k: v for k, v in (("world", w), ("robot", w.robot), ("scene", w.scene)) if k in args})
        if "seed" in args:
            args["seed"] = base + i
        failed, msg = False, ""
        try:
            pyfuncitem.obj(**args)
        except TooWideError:
            w.close(failed=False)
            raise  # this robot cannot run the test at all (see pytest_runtest_call)
        except (AssertionError, RobowrightError) as e:
            failed, msg = True, f"{type(e).__name__}: {e}"
        random_runs += w.faults.used_randomness()
        path = None
        try:
            path = w.close(
                failed=failed, trace_path=_trace_path(pyfuncitem.config, f"{_plain(pyfuncitem.nodeid)}[trial {i}]", _attempt(pyfuncitem))
            )
        except AssertionError as e:  # soft expectations
            failed, msg = True, str(e)
        if failed:
            report.failures.append((base + i, msg.splitlines()[0] if msg else "", str(path) if path else None))
            if path:
                pyfuncitem.config.stash[_TRACES].append((f"{_plain(pyfuncitem.nodeid)}[seed {base + i}]", str(path)))
        else:
            report.passed += 1
        report.ran = i + 1
        if report.settled and not every:
            break  # same seeds, same order: the verdict all n trials would give
    report.identical = random_runs == 0 and report.runs > 1
    if report.identical:
        warnings.warn(
            pytest.PytestWarning(
                f"{_plain(pyfuncitem.nodeid)}: its {report.runs} trials ran the same scene with nothing random in them, so they "
                "are one run repeated and its rate says nothing. Drop randomize=False, or add a fault that draws from the seed "
                "(jitter, joint_noise, camera_dropout) or draw from world.rng."
            )
        )
    pyfuncitem._rw_trials = report
    pyfuncitem.config.stash[_REPORTS][pyfuncitem.nodeid] = report
    if not report.ok:
        __tracebackhide__ = True
        lines = [f"trials: {report.summary()}"] + [
            f"  seed {s}: {msg}" + (f"\n    trace: {p}" if p else "") for s, msg, p in report.failures[:10]
        ]
        raise AssertionError("\n".join(lines))
    return True


class _Summary:
    """The trials and traces of one session, listed at its end.

    One per session, not module state: a session run inside another (pytester, or a test
    worker running the plugin's own tests) printed the outer run's results and wiped them.
    """

    def __init__(self):
        from .report import Run

        self.trials: list = []
        self.traces: list = []
        self.flaky: list = []
        self.config = None
        self.github = False
        self.annotations: list[str] = []
        self.failed_attempts: dict = {}  # nodeid -> the report of its last failed attempt (--rw-retries)
        self.run = Run()

    def pytest_runtest_logreport(self, report):
        self.run.add(report)
        # Runs on the xdist controller too, so traces from workers are listed.
        flaky = any(k == "robowright_flaky" for k, _ in report.user_properties)
        if flaky:
            self.flaky.append(_plain(report.nodeid))
        if report.outcome == "rerun":
            self.failed_attempts[report.nodeid] = report
        elif report.failed or flaky:
            self._annotate(report, self.failed_attempts.get(report.nodeid) if flaky else None)
        for k, v in report.user_properties:
            if k == "robowright_trace" and (report.failed or report.outcome == "rerun" or flaky):
                self.traces.append((_plain(report.nodeid), v))
            if k == "robowright_trials":
                self.trials.append((_plain(report.nodeid), v, report.outcome))

    def _annotate(self, report, failed_attempt) -> None:
        if self.config is None or not self.github:
            return
        from . import github

        trace = next((v for k, v in report.user_properties if k == "robowright_trace"), None)
        args = (str(self.config.rootpath), str(self.config.invocation_params.dir), trace, failed_attempt)
        self.annotations.append(github.annotation(report, *args))

    def pytest_terminal_summary(self, terminalreporter):
        for line in self.annotations:  # workflow commands: GitHub reads them from the log, so they go first and plain
            terminalreporter.write_line(line)
        out = terminalreporter.config.getoption("--rw-report")
        if out and self.run.tests:
            terminalreporter.section("robowright report")
            terminalreporter.line(str(self.run.write(out)))
        blob = terminalreporter.config.getoption("--rw-blob")
        if blob:  # written even for a shard with no tests: the report job expects one blob from each
            terminalreporter.section("robowright report blob")
            terminalreporter.line(str(self.run.save_blob(blob)))
        if self.flaky and terminalreporter.config.getoption("--rw-fail-on-flaky", False):
            terminalreporter.section("robowright flaky tests (--rw-fail-on-flaky)")
            for nodeid in dict.fromkeys(self.flaky):
                terminalreporter.line(f"FLAKY {nodeid}")
        if self.trials:
            terminalreporter.section("robowright trials")
            for nodeid, summary, outcome in self.trials:
                terminalreporter.line(f"{'PASS' if outcome == 'passed' else 'FAIL'} {nodeid}: {summary}")
        if self.traces:
            terminalreporter.section("robowright traces")
            for nodeid, path in dict.fromkeys(self.traces):
                terminalreporter.line(f"{nodeid}\n    robowright show-trace {path}")
            if terminalreporter.config.getoption("--rw-trace-text"):
                from .trace import Trace

                for _nodeid, path in dict.fromkeys(self.traces):
                    try:
                        text = Trace(path).summary()
                    except Exception as e:  # noqa: BLE001 - a summary never hides the run's own result
                        text = f"(could not read {path}: {e})"
                    terminalreporter.line("\n" + text)
