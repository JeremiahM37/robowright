"""Command line: ``robowright init | test | show-trace | show-report | replay | codegen | render | crosscheck``,
``robots | check | info | mcp``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
import zipfile
from pathlib import Path


def _test(args, rest):
    import pytest

    if "-h" in rest or "--help" in rest:
        # pytest's usage text alone reads as if `robowright test` were pytest itself; say what it is first.
        print(
            "usage: robowright test [--watch | --ui] [pytest arguments]\n\n"
            "Runs this project's robot tests. Every argument goes to pytest unchanged, so the\n"
            "options below are pytest's, plus robowright's --rw-* options (see `pytest --help`).\n\n"
            "  --watch    run again whenever a file changes: the changed test files when only\n"
            "             tests changed, everything otherwise (Enter runs everything)\n"
            "  --ui       open the UI in a browser: the suite by file, run any test or file with a\n"
            "             click, each test's trace in place, the camera live, and watch mode\n"
        )
    if "--ui" in rest and not ("-h" in rest or "--help" in rest):
        from .ui import serve

        return serve([a for a in rest if a not in ("--ui", "--watch")])
    if "--watch" in rest and not ("-h" in rest or "--help" in rest):
        from .watch import watch

        return watch([a for a in rest if a != "--watch"])
    return pytest.main([a for a in rest if a not in ("--watch", "--ui")])


_INIT_TEST = '''"""Robot tests, written like Playwright tests: your code drives the robot, the test watches and
asserts, and ``expect`` retries until the world catches up (in simulated time).

    pytest                                   # run them (robot and engine: pytest.ini)
    pytest --rw-live                         # and watch each one live in a browser
    pytest --rw-robot panda,ur5e             # on other robots too
    robowright show-trace robowright-traces/<test>.zip   # every failure leaves a trace

A pass means: this code did this task on this robot model in this engine. Which changes
robowright made to the model, if any, is in every trace (`robowright fidelity --robot so101`).
"""

import pytest

from robowright import condition, expect

# Your controller or policy: anything that maps an observation to joint targets (a function, a
# class with __call__, or a trained model via robowright.learned.LearnedPolicy). Until you have
# one, robowright's scripted reference policy stands in.
from robowright.policies import ScriptedPickPlace as MyController  # replace with: from my_robot import MyController


def test_my_controller_puts_the_cube_in_the_bin(robot, scene):
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(MyController(), until=done, hold=1.0, timeout=15)
    assert rollout.success, rollout
    expect(scene["cube"]).to_be_at_rest()


@pytest.mark.trials(20)  # 20 seeds, each moving the cube; every one must pass (min_success=0.9 accepts 18)
def test_my_controller_with_the_cube_moved(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(MyController(), until=done, hold=1.0, timeout=15)
    assert rollout.success, rollout


# Your ROS 2 stack instead (needs ROS 2): the simulation is served as a ROS 2 robot and your launch
# file drives it unchanged; the test only watches.
#
# def test_my_stack_puts_the_cube_in_the_bin(scene, ros2):
#     with ros2.run("ros2 launch my_robot pick.launch.py use_sim_time:=true"):
#         expect(scene["cube"]).to_be_inside(scene["bin"], timeout=60)


def test_the_reference_controller(robot, scene):
    """robowright's own pick and place: for setting a scene up, or trying a robot model out."""
    cube, bin = scene["cube"], scene["bin"]
    robot.pick(cube)
    expect(robot.gripper).to_be_holding(cube)
    robot.place(on=bin)
    expect(cube).to_be_inside(bin)
'''

_INIT_TOML = """# robowright settings for this project.
#
# Your own robot, from its model file (MJCF, URDF or xacro), by name in --rw-robot:
#   robowright robots --inspect path/to/arm.urdf     # what robowright makes of it
#   robowright robots add path/to/arm.urdf --name my_arm
# [robots.my_arm]
# file = "models/my_arm.urdf"
#
# A trained policy, by name in LearnedPolicy("pick_v2"):
# [policies.pick_v2]
# model = "checkpoints/pick_v2.onnx"
# state = ["qpos", "objects.cube"]
#
# A robot behind ROS 2 (pytest --rw-backend ros2): a real arm, Gazebo, Isaac Sim. Start a real
# arm observe-only (command = false); see docs/hardware.md.
# [ros2]
# joint_states = "/joint_states"
# command = false
"""

_INIT_ROS2_TEST = '''"""Tests against your ROS 2 stack, written like Playwright tests. robowright starts the stack once
(robowright.toml, [ros2.stack]) and puts it back as it came up before each test, as Playwright gives
each test a fresh browser context; the robot is the one your stack describes.

    pytest                                   # start the stack, run every test against it
    robowright show-trace robowright-traces/<test>.zip   # every failure leaves a trace

Your own code under test: start it in a test with ros2.run("ros2 launch ...") and assert on what it
does, or change what it hears with route= (see the README).
"""

import pytest

from robowright import expect


def test_the_stack_reports_the_robots_joints(world):
    assert world.robot.model.arm_joints or world.robot.model.family == "mobile", "the stack publishes no joints"


def test_the_arm_goes_where_its_controller_is_told(world, robot):
    if not robot.n_arm:
        pytest.skip("no arm controller in the stack")
    target = robot.qpos()[: robot.n_arm].copy()
    target[0] += 0.2  # the first joint, a little: robowright plans the move and sends it through your controller
    robot.arm.move_joints(target)
    expect(robot).to_have_joint(robot.model.arm_joints[0], float(target[0]), tol=0.02)
'''

_INIT_ROS2_TOML = """# robowright settings for this project: a robot behind ROS 2.
[ros2]
use_sim_time = {sim_time}  # the stack runs on simulated time ({sim_why})
gazebo = {gazebo}

# The stack the tests run against, started once per test worker and put back before each test.
[ros2.stack]
command = {command}
# env = {{ DISPLAY = false }}  # a variable to set for the stack (false unsets it)
# timeout = 240              # seconds it may take to come up
# speed = 2.0                # Gazebo faster than real time (a stack on simulated time)

# For a real arm, start observe-only (command = false under [ros2]) and set [ros2.limits];
# see docs/hardware.md.
"""

_INIT_PYTEST = """[pytest]
testpaths = tests
# The robots and engines every test runs on (each is a separate test, like Playwright's projects).
addopts = --rw-robot {robot} --rw-backend {backend}
"""

_INIT_CI = """name: robot tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        shard: [1, 2]  # more machines: more numbers here, and the same count in --rw-shard below
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      # Rendering (image snapshots, videos) uses the CPU through Mesa's EGL: no GPU or display needed.
      - run: sudo apt-get update && sudo apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libosmesa6
      - run: pip install __ROBOWRIGHT__ pytest-xdist
      # Traces record state, not pictures: tests need no rendering of their own. --rw-trace-text
      # prints each failure's trace in the log. --rw-retries 2 runs a failed test again: one that
      # then passes is reported flaky, not passed. Each shard keeps its results, traces included,
      # in one zip (--rw-blob) for the report job.
      - run: pytest -n auto --rw-retries 2 --rw-trace-text --rw-shard ${{ matrix.shard }}/2 --rw-blob blob-${{ matrix.shard }}.zip
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: robowright-blob-${{ matrix.shard }}
          path: blob-${{ matrix.shard }}.zip
  report:
    # One report of every shard, each failure with its trace (open report.html, or robowright show-report).
    needs: test
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install __ROBOWRIGHT__
      - uses: actions/download-artifact@v4
        with:
          pattern: robowright-blob-*
          merge-multiple: true
      - run: robowright merge-reports blob-*.zip -o report/report.html
      - uses: actions/upload-artifact@v4
        with:
          name: robowright-report
          path: report/
"""


REPO = "https://github.com/JeremiahM37/robowright"


def _install_spec() -> str:
    """How CI installs robowright, the way it is installed here: from PyPI at this version when it came from PyPI,
    else from its repository (a git or local install, which a CI machine cannot reach)."""
    from importlib import metadata

    try:
        dist = metadata.distribution("robowright")
    except metadata.PackageNotFoundError:
        return f'"robowright @ git+{REPO}"'
    if dist.read_text("direct_url.json"):  # installed from a URL or a path, not from an index
        return f'"robowright @ git+{REPO}"'
    return f'"robowright=={dist.version}"'


def _init_ci() -> str:
    return _INIT_CI.replace("__ROBOWRIGHT__", _install_spec())


def _init_mcp() -> str:
    """The .mcp.json entry. The interpreter that ran init, not a bare ``robowright``: that command only
    resolves with this venv on PATH, so a client started from anywhere else would find no server."""
    entry = {"command": sys.executable, "args": ["-m", "robowright.cli", "mcp"]}
    return json.dumps({"mcpServers": {"robowright": entry}}, indent=2) + "\n"


_AGENTS = {
    "robowright-planner": (
        "Explores a robot and its scene through robowright and writes a test plan: the scenarios worth testing, "
        "each with its steps and what must be true after. Use before writing robot tests for a new robot, scene or behaviour.",
        """You plan robot tests, as Playwright's planner agent plans web tests. You do not write test code.

1. Read the project: robowright.toml, pytest.ini, the tests that exist, and the code under test (a controller, a
   policy, a ROS 2 stack) so you know what is being tested and what is already covered.
2. Explore with the robowright MCP tools: robot_list_robots, robot_launch (the robot and engine the project uses),
   robot_snapshot and robot_screenshot to see the scene, then try what the code under test should achieve
   (robot_run_policy for a policy, robot_start_process for a stack, or robot_pick / robot_place / robot_move_to)
   and check outcomes with robot_expect. Try what could go wrong too: robot_fault (pushes, delays, sensor
   noise, weak joints) and robot_move_object for other starting positions.
3. Write the plan to specs/<feature>.md: one section per scenario, each with its starting state (robot, engine,
   seed, object positions, faults), its steps, and its expected outcome written as robowright matchers
   (to_be_inside, to_be_at_rest, to_be_holding, always.to_have_no_collisions, ...) with the tolerance and
   timeout you measured to be right. Include the scenarios that should fail and say why.

Base every expectation on something you observed in the session, never on a guess about the physics.""",
    ),
    "robowright-generator": (
        "Turns a robowright test plan (specs/*.md) into pytest tests by carrying out each scenario in a live "
        "robowright session and saving it as a test. Use after the planner, or with a scenario described in words.",
        """You write robot tests from a plan, as Playwright's generator agent does: by doing each scenario, not by
imagining it.

For each scenario in the plan (or the one you are given):
1. robot_launch with its robot, engine and seed; set its starting state (robot_move_object, robot_fault).
2. Carry out its steps with the MCP tools, and check each expected outcome with robot_expect as you go. When an
   expectation does not hold, look (robot_snapshot, robot_screenshot) before changing anything: either the step
   was wrong, or the code under test does not do what the plan says. Record which.
3. robot_generate_test with a descriptive test name and a path under tests/: it writes a pytest test that
   reproduces the session exactly. Tidy it into the project's style (fixtures, names) without changing what it
   does or asserts.
4. robot_run_tests on the new file. It must pass on the engines the project runs (pytest.ini's --rw-backend).

Never loosen a tolerance or lengthen a timeout beyond what you measured just to make a test pass. A scenario
the code under test fails is a finding: write that test with @pytest.mark.xfail(strict=True, reason=...) naming
what you measured, and report it.""",
    ),
    "robowright-healer": (
        "Runs the robowright tests, reads each failure's trace, and fixes the tests that are wrong (renamed objects, "
        "changed scenes, stale timeouts). Use when robot tests fail after a change.",
        """You repair failing robot tests, as Playwright's healer agent does, without hiding real failures.

1. robot_run_tests (the whole suite, or what you are pointed at). Each failure comes with its trace as text.
2. For each failure, robot_read_trace on its trace: what ran, the call that failed, the joints, the object poses
   and the contacts at that moment. Decide which is wrong:
   - the TEST: an object or joint that was renamed, a scene that changed on purpose, a step that no longer
     matches the code's interface, a timeout shorter than the motion now takes for a reason you can name;
   - the CODE UNDER TEST: it does not achieve what the test asks (the cube is not in the bin, the arm collided).
   Reproduce the failing moment in a session (robot_launch with the test's robot, engine and seed) when the
   trace alone does not settle it.
3. Fix a test that is wrong, the smallest change that makes it test the same thing again, and rerun it.
   A failure of the code under test is not yours to fix here: mark that test
   @pytest.mark.xfail(strict=True, reason=...) with what the trace showed, and report it.
4. Rerun the suite and report: each test fixed and why it was wrong, each failure left for the code's owners.

Never weaken an assertion, widen a tolerance or delete a test to make the suite pass.""",
    ),
}


def _init_agents(args, rest):
    """Write Claude Code agent definitions for robot tests, as ``npx playwright init-agents`` does: a planner, a
    generator and a healer, each using robowright's MCP tools (registered in .mcp.json, written if missing)."""
    root = Path(args.dir)
    # No tools line: an agent then has every tool, robowright's MCP ones included.
    files = {
        root / ".claude" / "agents" / f"{name}.md": f"---\nname: {name}\ndescription: {desc}\n---\n\n{body}\n"
        for name, (desc, body) in _AGENTS.items()
    }
    files[root / ".mcp.json"] = _init_mcp()
    for path, text in files.items():
        if path.exists():
            print(f"kept     {path} (exists)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote    {path}")
    print(
        "\nnext, in Claude Code (claude):\n"
        '  "use robowright-planner to plan tests for picking the cube"\n'
        '  "use robowright-generator to write the tests in specs/pick.md"\n'
        '  "use robowright-healer to fix the failing robot tests"'
    )
    return 0


def _init(args, rest):
    """Set up a project the way ``npm init playwright`` does: an example test, settings, CI, and
    the MCP server registered for AI coding agents. Files that exist are left alone."""
    root = Path(args.dir)
    if args.ros2:
        import json as _json

        gz = args.gazebo
        files = {
            root / "tests" / "test_robot.py": _INIT_ROS2_TEST,
            root / "robowright.toml": _INIT_ROS2_TOML.format(
                command=_json.dumps(args.ros2),  # a TOML basic string is a JSON string
                sim_time="true" if gz else "false",
                sim_why="Gazebo publishes /clock" if gz else "set true for a simulator that publishes /clock",
                gazebo="true" if gz else "false",
            ),
            root
            / "pytest.ini": "[pytest]\ntestpaths = tests\n# Every test against the stack in robowright.toml\naddopts = --rw-backend ros2\n",
            root / ".mcp.json": _init_mcp(),
        }
    else:
        files = {
            root / "tests" / "test_robot.py": _INIT_TEST,
            root / "robowright.toml": _INIT_TOML,
            root / "pytest.ini": _INIT_PYTEST.format(robot=args.robot, backend=args.backend),
            root / ".github" / "workflows" / "robowright.yml": _init_ci(),
            root / ".mcp.json": _init_mcp(),
        }
    for path, text in files.items():
        if path.exists():
            print(f"kept     {path} (exists)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote    {path}")
    ignore = root / ".gitignore"
    lines = ignore.read_text().splitlines() if ignore.exists() else []
    if "robowright-traces/" not in lines:
        ignore.write_text("\n".join([*lines, "robowright-traces/"]) + "\n")
        print(f"updated  {ignore}")
    if args.ros2:
        print(
            "\nnext (in your ROS 2 environment):\n"
            "  pytest                            start the stack once, run the tests against it\n"
            "  robowright show-trace <trace>     look into a failure\n"
            "  claude                            an AI agent that can drive the robot and run the tests (.mcp.json)"
        )
        return 0
    print(
        "\nnext:\n"
        "  pytest                            run the tests\n"
        "  robowright show-trace <trace>     look into a failure (--text for a plain summary)\n"
        "  claude                            an AI agent that can drive the robot and run the tests (.mcp.json)"
    )
    return 0


def _show(args, rest):
    if args.perfetto:
        import json as _json

        from .trace import Trace

        out = Path(args.perfetto)
        out.write_text(_json.dumps(Trace(args.trace).chrome_trace()))
        print(f"{out}: open it in https://ui.perfetto.dev or chrome://tracing")
        return 0
    if args.text:
        from .trace import Trace

        print(Trace(args.trace).summary())
        return 0
    from .viewer import write_html

    out = write_html(args.trace, args.out)
    print(out)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _show_report(args, rest):
    """Playwright's ``show-report``: open the HTML report a run wrote (``pytest --rw-report``)."""
    path = Path(args.report)
    if path.is_dir():
        path = path / "report.html"
    if not path.exists():
        print(f"robowright show-report: no report at {path}: write one with `pytest --rw-report {path}`", file=sys.stderr)
        return 2
    if args.port is None:
        print(path)
        if not args.no_open:
            webbrowser.open(path.resolve().as_uri())
        return 0
    import functools
    import http.server

    # Served, for a report on another machine (CI, a remote box): the page links its traces by relative path.
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(path.resolve().parent))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{httpd.server_address[1]}/{path.name}"
        print(f"serving {url} (Ctrl-C to stop)", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def _replay(args, rest):
    from .replay import load_factory, replay

    try:
        make_env = load_factory(args.factory) if args.factory else None
        r = replay(args.trace, backend=args.backend, tol=args.tol, make_env=make_env)
    except ValueError as e:  # an attached run whose environment has no spec to rebuild it from
        print(f"robowright replay: {e}", file=sys.stderr)
        return 2
    print(r.summary())
    return 0 if r.matches else 1


def _codegen(args, rest):
    from .codegen import companion, generate

    if args.trace is None:  # record: the test is written from what is done in the browser
        from .recorder import Recorder

        rec = Recorder(robot=args.robot, backend=args.backend, seed=args.seed, output=args.output or "test_recorded.py", name=args.name)

        def ready(url):
            print(f"robowright codegen: {url} (Save writes {rec.output}; Ctrl+C to stop)", flush=True)

        try:
            rec.serve(port=args.port, ready=ready)
        except KeyboardInterrupt:
            pass
        finally:
            rec.close()
        return 0

    try:
        code = generate(args.trace, test_name=args.name, stop_at_failure=not args.full, factory=args.factory)
    except ValueError as e:  # an attached run that cannot be turned into a test, or a factory that does not load
        print(f"robowright codegen: {e}", file=sys.stderr)
        return 2
    extra = companion(args.trace, test_name=args.name)  # a long attached run's actions, read by the test beside it
    if args.output:
        Path(args.output).write_text(code)
        print(args.output)
        if extra:
            path = Path(args.output).with_name(extra[0])
            path.write_bytes(extra[1])
            print(path)
    else:
        if extra:
            print(f"robowright codegen: pass -o to write {extra[0]} next to the test", file=sys.stderr)
            return 2
        sys.stdout.write(code)
    return 0


def _merge_reports(args, rest):
    from .report import merge

    try:
        print(merge(args.blobs, args.output))
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as e:
        print(f"robowright merge-reports: {e}", file=sys.stderr)
        return 2
    return 0


def _info(args, rest):
    import platform

    import numpy

    from . import __version__
    from .backends.base import _REQUIRES, available

    print(f"robowright {__version__}")
    print(f"python {platform.python_version()}, numpy {numpy.__version__}")
    from .plugins import BACKENDS, ROBOTS, _entry_points

    plugins = {ep.name: ep.value for ep in _entry_points(BACKENDS)}
    for name in available():
        from importlib.metadata import PackageNotFoundError, version

        if name not in _REQUIRES:
            print(f"backend {name}: plugin ({plugins[name]})")
            continue
        dist = {"genesis": "genesis-world", "drake": "drake"}.get(name, _REQUIRES[name])
        try:
            v = version(dist)
        except PackageNotFoundError:
            v = "installed"
        print(f"backend {name}: {v}")
    for ep in _entry_points(ROBOTS):
        print(f"robot plugin {ep.name}: {ep.value}")
    from .plugins import POLICIES, project_policies

    for ep in _entry_points(POLICIES):
        print(f"policy loader {ep.name}: {ep.value}")
    for name, (model, _) in project_policies().items():
        print(f"project policy {name}: {model}")
    try:
        from . import launch

        w = launch(settings=__import__("robowright").Settings(trace="off"))
        img = w.backend.render("front", 64, 48)
        w.close()
        print(f"offscreen rendering: ok ({img.shape[1]}x{img.shape[0]})")
    except Exception as e:  # rendering is optional: traces just lose their frames
        print(f"offscreen rendering: unavailable ({type(e).__name__}: {e})")
        if sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):
            print("  no EGL or OSMesa library was found: install one (Debian/Ubuntu: apt install libegl1, or libosmesa6)")
    return 0


def _check(argv) -> int:
    """Run the contract (robowright.contract) on robots and engines; other arguments go to pytest."""
    import pytest

    p = argparse.ArgumentParser(prog="robowright check", description=_check.__doc__)
    p.add_argument("--robot", default="so101", help="robots (names, model files, 'all', 'legged'), comma-separated")
    p.add_argument("--backend", default="mujoco", help="engines, comma-separated")
    args, rest = p.parse_known_args(argv)
    # A ROS 2 install's launch_testing plugins do not load under pytest 9 (they declare hook
    # arguments pytest removed), which stops every run in a sourced ROS environment. The
    # contract has no launch tests: leave them out.
    from importlib.metadata import entry_points

    skip = [f"-p no:{ep.name}" for ep in entry_points(group="pytest11") if ep.name in ("launch_testing", "launch_ros")]
    skip = [a for s in skip for a in s.split(" ", 1)]
    # No .pytest_cache: pytest's rootdir is the working directory when the model file is beside it,
    # so a check run from a robot's folder left a cache in that folder.
    return int(
        pytest.main(
            [
                "--pyargs",
                "robowright.contract",
                "--rw-robot",
                args.robot,
                "--rw-backend",
                args.backend,
                *skip,
                "-p",
                "no:cacheprovider",
                *rest,
            ]
        )
    )


def _robots(args, rest):
    from . import robots

    if args.action == "add":
        return _add(args)
    if args.inspect:
        return _inspect(args.inspect, bare=args.no_gripper)
    rows = []
    for name in robots.names(args.family):
        m = robots.get(name)
        kind = m.family if m.family == "arm" else next((t for t in m.tags if t in ("quadruped", "humanoid")), m.family)
        rows.append((name, m.title, m.maker, kind, str(m.n_arm), m.license))
    head = ("name", "robot", "maker", "kind", "joints", "model licence")
    if args.markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in rows:
            print(f"| `{r[0]}` | " + " | ".join(r[1:]) + " |")
        return 0
    widths = [max(len(x) for x in col) for col in zip(head, *rows)]
    for r in (head, *rows):
        print("  ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip())
    from .plugins import find_config, project_robots

    mine = project_robots()
    if mine and not args.family:
        print(f"\nthis project's robots ({find_config()[0]}), detected from their files on first use:")
        for name, (file, overrides) in mine.items():
            print(f"  {name}  {file}" + (f"  (+ {', '.join(overrides)})" if overrides else ""))
    return 0


def _add(args) -> int:
    """Check that robowright can drive the robot in a model file, then name it in robowright.toml."""
    from .plugins import add_project_robot

    if not args.file or not args.name:
        print("usage: robowright robots add FILE --name NAME", file=sys.stderr)
        return 2
    from .plugins import project_robots

    if args.name in project_robots():
        print(f"robowright: this project already names a robot {args.name!r}", file=sys.stderr)
        return 1
    if _inspect(args.file, hint=False, bare=args.no_gripper):
        return 1
    try:
        target = add_project_robot(args.file, args.name, fields={"gripper": False} if args.no_gripper else None)
    except ValueError as e:
        print(f"robowright: {e}", file=sys.stderr)
        return 1
    print(f"\nadded {args.name!r} to {target}; run its tests with:  pytest --rw-robot {args.name}")
    print(f"or the contract every robot meets:  robowright check --robot {args.name}")
    return 0


def _inspect(path, hint: bool = True, bare: bool = False) -> int:
    """What robowright works out about the robot in a model file, and why."""
    import warnings

    from .robots.detect import DetectionError, arm_reach, build

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MuJoCo's notes on attaching a gripper are not the user's business
        try:
            built = build(path, **({"gripper": False} if bare else {}))
        except (DetectionError, FileNotFoundError) as e:
            print(f"robowright cannot drive {path}: {e}")
            return 1
    m = built.model
    print(f"{path}: {m.family}, {m.n_arm} joints, named {m.name!r}")
    for note in built.notes:
        print(f"  {note}")
    if m.family == "arm":
        print(f"  reach: {arm_reach(m):.2f} m (the farthest the tool point gets from the base over its joint ranges)")
    if hint and bare:
        print("\nan arm with no gripper: for tests of a robot reached over ROS 2 (--rw-backend ros2), not pick-and-place")
        print(f"give it a name:  robowright robots add {path} --name NAME --no-gripper")
    elif hint:
        print(f"\nrun its tests with:  pytest --rw-robot {path}")
        print(f"or give it a name:  robowright robots add {path} --name NAME")
        print("override any of the above in robowright.toml: [robots.NAME] file = ..., <field> = ...")
    return 0


def _render(args, rest) -> int:
    from .render import render_video

    w, h = (int(x) for x in args.size.lower().split("x"))
    out = args.output or str(Path(args.trace).with_suffix(".mp4"))
    try:
        path = render_video(args.trace, out, camera=args.camera, size=(w, h), fps=args.fps)
    except ImportError:
        print("mp4 output needs imageio: pip install 'robowright[video]' (or write a .gif)", file=sys.stderr)
        return 1
    print(path)
    return 0


def _crosscheck(args, rest) -> int:
    from .crosscheck import crosscheck

    r = crosscheck(args.trace, args.backend)
    print(r.summary())
    return 0 if r.agrees else 1


def _mcp(args, rest) -> int:
    from .mcp_server import main as serve

    try:
        import mcp  # noqa: F401
    except ImportError:
        print("robowright mcp needs the MCP SDK: pip install 'robowright[mcp]'", file=sys.stderr)
        return 1
    serve()
    return 0


def _sim(args, rest):
    """Run a simulation on its own, in real time, for code outside robowright to drive and people to watch."""
    import robowright as rw

    from .live import LiveView, serve_forever
    from .scene import default_scene

    robot = args.robot
    if "/" in robot or robot.endswith((".xml", ".urdf", ".xacro")):
        from . import robots

        robot = robots.load(robot).name
    w = rw.launch(default_scene(robot), backend=args.backend, seed=args.seed, settings=rw.Settings(trace="off", fidelity=args.fidelity))
    print(f"{robot} on {args.backend} (fidelity {w.fidelity}), running in real time; Ctrl-C stops it")
    bridge = view = None
    try:
        if args.ros2:
            from .ros2_bridge import Ros2Bridge

            bridge = Ros2Bridge(w, namespace=args.namespace, realtime=False)
            ns = args.namespace.rstrip("/")
            print(f"ROS 2: {ns}/joint_states, /clock, TF ({', '.join(w.object_names)} in {bridge.frame}),")
            print(f"       {ns}/arm_controller/follow_joint_trajectory, {ns}/gripper_controller/gripper_cmd,")
            print(f"       {ns}/forward_position_controller/commands")
        if args.live is not None:
            view = LiveView(w, port=args.live)
        serve_forever(w)
    finally:
        if bridge is not None:
            bridge.close()
        if view is not None:
            view.close()
        w.close()
    return 0


def _fidelity(args, rest):
    """What robowright changes between a robot's model file and what an engine simulates."""
    import robowright as rw

    from . import fidelity
    from .scene import default_scene

    robot = args.robot
    if "/" in robot or robot.endswith((".xml", ".urdf", ".xacro")):
        from . import robots

        robot = robots.load(robot).name
    w = rw.launch(default_scene(robot), backend=args.backend, settings=rw.Settings(trace="off", fidelity=args.mode))
    try:
        print(f"{robot} on {args.backend}, fidelity {w.fidelity}:")
        print("  " + fidelity.summary(w.model_changes).replace("\n", "\n  "))
        if w.fidelity == "published":
            print("(--mode adjusted shows what robowright's own tuning adds on top)")
    finally:
        w.close()
    return 0


def _ros2(args, _rest) -> int:
    try:
        from . import ros2_discover as discover

        graph = discover.inspect(args.namespace, timeout=args.timeout)
    except ImportError as e:
        print(f"robowright ros2 needs a ROS 2 environment (rclpy): {e}", file=sys.stderr)
        return 2
    print(discover.report(graph, discover.suggest(graph)))
    return 0 if graph.joints else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="robowright", description="Playwright-style testing for robots.")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="set up a project: an example test, settings, CI and the MCP server for AI agents")
    i.add_argument("dir", nargs="?", default=".")
    i.add_argument("--robot", default="so101", help="the robots tests run on (default so101)")
    i.add_argument("--backend", default="mujoco", help="the engines tests run on (default mujoco)")
    i.add_argument(
        "--ros2", metavar="COMMAND", help='a ROS 2 project: the command that starts your stack ("ros2 launch my_robot sim.launch.py")'
    )
    i.add_argument("--gazebo", action="store_true", help="with --ros2: the stack runs Gazebo (simulated time, Gazebo's ground truth)")
    ia = sub.add_parser("init-agents", help="AI agents that plan, write and repair robot tests (Claude Code), as playwright init-agents")
    ia.add_argument("dir", nargs="?", default=".")
    sub.add_parser("test", help="run tests (arguments go to pytest; --watch: again on every change; --ui: in a browser)", add_help=False)
    s = sub.add_parser("show-trace", help="open a trace in the HTML viewer")
    s.add_argument("trace")
    s.add_argument("-o", "--out")
    s.add_argument("--no-open", action="store_true", help="write the HTML but do not open a browser")
    s.add_argument("--text", action="store_true", help="print what ran, what failed and the state at the failure, as text")
    s.add_argument("--perfetto", metavar="OUT.json", help="write the trace for ui.perfetto.dev / chrome://tracing instead")
    mr = sub.add_parser("merge-reports", help="one HTML report from the blobs of a sharded run (pytest --rw-blob)")
    mr.add_argument("blobs", nargs="+", help="the blobs, one per shard")
    mr.add_argument("-o", "--output", default="report.html")
    sr = sub.add_parser("show-report", help="open the HTML report a run wrote (pytest --rw-report)")
    sr.add_argument("report", nargs="?", default="report.html", help="the report, or the folder it is in (default report.html)")
    sr.add_argument("--port", type=int, help="serve it over HTTP on this port (0: any free one) instead of opening the file")
    sr.add_argument("--host", default="127.0.0.1", help="the address to serve on (default 127.0.0.1)")
    sr.add_argument("--no-open", action="store_true", help="do not open a browser")
    r = sub.add_parser("replay", help="re-simulate a trace and report divergence")
    r.add_argument("trace")
    r.add_argument("--backend")
    r.add_argument("--tol", type=float, default=1e-6)
    r.add_argument(
        "--factory", metavar="MODULE:FUNC", help="for an attached run: the function that builds its simulation (or path/to/file.py:func)"
    )
    c = sub.add_parser(
        "codegen",
        help="generate a pytest test from a trace, or, with no trace, write one by doing it in your browser (--robot)",
    )
    c.add_argument("trace", nargs="?", help="a trace .zip; leave it out to record a test in your browser")
    c.add_argument("-o", "--output")
    c.add_argument("--robot", default="so101", help="recording: the robot (a name or model file)")
    c.add_argument("--backend", default="mujoco", help="recording: the engine")
    c.add_argument("--seed", type=int, default=0, help="recording: the scene's seed")
    c.add_argument("--port", type=int, default=8790, help="recording: the page's port (0: any free one)")
    c.add_argument("--name")
    c.add_argument("--full", action="store_true", help="include events after the first failure")
    c.add_argument(
        "--factory", metavar="MODULE:FUNC", help="for an attached run: the test imports this function as make_env (or path/to/file.py:func)"
    )
    sub.add_parser("info", help="versions, backends and rendering support")
    sm = sub.add_parser("sim", help="run a simulation in real time for your code to drive (--ros2) and you to watch (--live)")
    sm.add_argument("--robot", default="so101", help="a robot name or model file")
    sm.add_argument("--backend", default="mujoco")
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--fidelity", choices=["published", "adjusted"], default=None)
    sm.add_argument("--ros2", action="store_true", help="serve it as a ROS 2 robot (needs rclpy)")
    sm.add_argument("--namespace", default="", help="ROS 2 namespace for its topics and actions")
    sm.add_argument("--live", type=int, nargs="?", const=8765, default=None, metavar="PORT", help="watch it in a browser (port 8765)")
    fd = sub.add_parser("fidelity", help="what robowright changes between a robot's model file and what an engine simulates")
    fd.add_argument("--robot", default="so101", help="a robot name or model file")
    fd.add_argument("--backend", default="mujoco")
    fd.add_argument("--mode", choices=["published", "adjusted"], default=None, help="default: $ROBOWRIGHT_FIDELITY, else published")
    rb = sub.add_parser("robots", help="list the robots tests can run on")
    rb.add_argument("--family", choices=["arm", "legged"])
    rb.add_argument("--markdown", action="store_true")
    rb.add_argument("--inspect", metavar="FILE", help="show what robowright makes of the robot in a model file (MJCF or URDF), and why")
    rb.add_argument("action", nargs="?", choices=["add"], help="add: name the robot in FILE in this project's robowright.toml")
    rb.add_argument("file", nargs="?", help="the model file to add (MJCF, URDF or xacro)")
    rb.add_argument("--name", help="the name to give it")
    rb.add_argument(
        "--no-gripper",
        action="store_true",
        help="the arm as its file has it, with no gripper (else a gripperless arm gets a Robotiq 2F-85): for a robot over ROS 2",
    )
    sub.add_parser("check", help="run the contract every robot and engine meets (robowright check --robot R --backend B)", add_help=False)
    rd = sub.add_parser("render", help="render a trace to video (mp4, or gif) from its recorded state")
    rd.add_argument("trace")
    rd.add_argument("-o", "--output", help="output file (.mp4 or .gif); default: next to the trace")
    rd.add_argument("--camera", default="front")
    rd.add_argument("--size", default="1280x720", help="WIDTHxHEIGHT")
    rd.add_argument("--fps", type=float, help="default: the run's own pace (50 fps)")
    cc = sub.add_parser("crosscheck", help="make a trace's calls again on another engine and compare the outcome")
    cc.add_argument("trace")
    cc.add_argument("--backend", required=True)
    sub.add_parser("mcp", help="run the MCP server (stdio) that lets an AI agent drive a simulated robot")
    ros = sub.add_parser("ros2", help="look at a robot behind ROS 2: its joints and controllers, and the settings to connect with")
    ros.add_argument("action", choices=["inspect"])
    ros.add_argument("--namespace", default="", help="the robot's ROS namespace")
    ros.add_argument("--timeout", type=float, default=5.0, help="seconds to look for (default 5)")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["test"]:
        return _test(None, argv[1:])
    if argv[:1] == ["check"]:
        return _check(argv[1:])
    args = p.parse_args(argv)
    commands = {
        "show-trace": _show,
        "show-report": _show_report,
        "merge-reports": _merge_reports,
        "replay": _replay,
        "codegen": _codegen,
        "render": _render,
        "crosscheck": _crosscheck,
        "info": _info,
        "fidelity": _fidelity,
        "sim": _sim,
        "robots": _robots,
        "mcp": _mcp,
        "init": _init,
        "init-agents": _init_agents,
        "ros2": _ros2,
    }
    return commands[args.cmd](args, [])


if __name__ == "__main__":
    sys.exit(main())
