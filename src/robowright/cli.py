"""Command line: ``robowright init | test | show-trace | replay | codegen | render | crosscheck | robots | check | info | mcp``."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


def _test(args, rest):
    import pytest

    return pytest.main(rest)


_INIT_TEST = '''"""Robot tests, written like Playwright tests: actions wait until they are done, and
``expect`` retries until the world catches up (in simulated time).

    pytest                                   # run them (robot and engine: pytest.ini)
    pytest --rw-robot panda,ur5e             # on other robots too
    robowright show-trace robowright-traces/<test>.zip   # every failure leaves a trace
"""

import pytest

from robowright import condition, expect
from robowright.policies import ScriptedPickPlace


def test_pick_and_place(robot, scene):
    cube, bin = scene["cube"], scene["bin"]
    robot.pick(cube)
    expect(robot.gripper).to_be_holding(cube)
    robot.place(on=bin)
    expect(cube).to_be_inside(bin)
    expect(cube).to_be_at_rest()


@pytest.mark.trials(20)  # 20 seeds; every one must pass (min_success=0.9 would accept 18)
def test_a_policy_with_the_cube_moved(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(ScriptedPickPlace(), until=done, hold=1.0, timeout=15)
    assert rollout.success, rollout
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
# A robot behind ROS 2 (pytest --rw-backend ros2):
# [ros2]
# joint_states = "/joint_states"
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
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      # robowright is not on PyPI yet: install it from its repository until it is, e.g.
      #   pip install "robowright @ git+https://github.com/JeremiahM37/robowright" pytest-xdist
      - run: pip install robowright pytest-xdist
      # Traces record state, not pictures: no GPU or display is needed to test. --rw-trace-text
      # prints each failure's trace in the log; the traces themselves are kept as an artifact.
      - run: pytest -n auto --rw-trace-text
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: robowright-traces
          path: robowright-traces/
"""

_INIT_MCP = """{
  "mcpServers": {
    "robowright": {"command": "robowright", "args": ["mcp"]}
  }
}
"""


def _init(args, rest):
    """Set up a project the way ``npm init playwright`` does: an example test, settings, CI, and
    the MCP server registered for AI coding agents. Files that exist are left alone."""
    root = Path(args.dir)
    files = {
        root / "tests" / "test_robot.py": _INIT_TEST,
        root / "robowright.toml": _INIT_TOML,
        root / "pytest.ini": _INIT_PYTEST.format(robot=args.robot, backend=args.backend),
        root / ".github" / "workflows" / "robowright.yml": _INIT_CI,
        root / ".mcp.json": _INIT_MCP,
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
    print(
        "\nnext:\n"
        "  pytest                            run the tests\n"
        "  robowright show-trace <trace>     look into a failure (--text for a plain summary)\n"
        "  claude                            an AI agent that can drive the robot and run the tests (.mcp.json)"
    )
    return 0


def _show(args, rest):
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


def _replay(args, rest):
    from .replay import replay

    r = replay(args.trace, backend=args.backend, tol=args.tol)
    print(r.summary())
    return 0 if r.first_divergent_step is None else 1


def _codegen(args, rest):
    from .codegen import generate

    code = generate(args.trace, test_name=args.name, stop_at_failure=not args.full)
    if args.output:
        Path(args.output).write_text(code)
        print(args.output)
    else:
        sys.stdout.write(code)
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
    return int(pytest.main(["--pyargs", "robowright.contract", "--rw-robot", args.robot, "--rw-backend", args.backend, *skip, *rest]))


def _robots(args, rest):
    from . import robots

    if args.action == "add":
        return _add(args)
    if args.inspect:
        return _inspect(args.inspect)
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
    if _inspect(args.file, hint=False):
        return 1
    try:
        target = add_project_robot(args.file, args.name)
    except ValueError as e:
        print(f"robowright: {e}", file=sys.stderr)
        return 1
    print(f"\nadded {args.name!r} to {target}; run its tests with:  pytest --rw-robot {args.name}")
    print(f"or the contract every robot meets:  robowright check --robot {args.name}")
    return 0


def _inspect(path, hint: bool = True) -> int:
    """What robowright works out about the robot in a model file, and why."""
    import warnings

    from .robots.detect import DetectionError, build

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MuJoCo's notes on attaching a gripper are not the user's business
        try:
            built = build(path)
        except (DetectionError, FileNotFoundError) as e:
            print(f"robowright cannot drive {path}: {e}")
            return 1
    m = built.model
    print(f"{path}: {m.family}, {m.n_arm} joints, named {m.name!r}")
    for note in built.notes:
        print(f"  {note}")
    if hint:
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="robowright", description="Playwright-style testing for robots.")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="set up a project: an example test, settings, CI and the MCP server for AI agents")
    i.add_argument("dir", nargs="?", default=".")
    i.add_argument("--robot", default="so101", help="the robots tests run on (default so101)")
    i.add_argument("--backend", default="mujoco", help="the engines tests run on (default mujoco)")
    sub.add_parser("test", help="run tests (all arguments are passed to pytest)", add_help=False)
    s = sub.add_parser("show-trace", help="open a trace in the HTML viewer")
    s.add_argument("trace")
    s.add_argument("-o", "--out")
    s.add_argument("--no-open", action="store_true", help="write the HTML but do not open a browser")
    s.add_argument("--text", action="store_true", help="print what ran, what failed and the state at the failure, as text")
    r = sub.add_parser("replay", help="re-simulate a trace and report divergence")
    r.add_argument("trace")
    r.add_argument("--backend")
    r.add_argument("--tol", type=float, default=1e-6)
    c = sub.add_parser("codegen", help="generate a pytest regression test from a trace")
    c.add_argument("trace")
    c.add_argument("-o", "--output")
    c.add_argument("--name")
    c.add_argument("--full", action="store_true", help="include events after the first failure")
    sub.add_parser("info", help="versions, backends and rendering support")
    rb = sub.add_parser("robots", help="list the robots tests can run on")
    rb.add_argument("--family", choices=["arm", "legged"])
    rb.add_argument("--markdown", action="store_true")
    rb.add_argument("--inspect", metavar="FILE", help="show what robowright makes of the robot in a model file (MJCF or URDF), and why")
    rb.add_argument("action", nargs="?", choices=["add"], help="add: name the robot in FILE in this project's robowright.toml")
    rb.add_argument("file", nargs="?", help="the model file to add (MJCF, URDF or xacro)")
    rb.add_argument("--name", help="the name to give it")
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
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["test"]:
        return _test(None, argv[1:])
    if argv[:1] == ["check"]:
        return _check(argv[1:])
    args = p.parse_args(argv)
    commands = {
        "show-trace": _show,
        "replay": _replay,
        "codegen": _codegen,
        "render": _render,
        "crosscheck": _crosscheck,
        "info": _info,
        "robots": _robots,
        "mcp": _mcp,
        "init": _init,
    }
    return commands[args.cmd](args, [])


if __name__ == "__main__":
    sys.exit(main())
