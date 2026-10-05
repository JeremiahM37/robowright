"""Command line: ``robowright test | show-trace | replay | codegen | robots | info``."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


def _test(args, rest):
    import pytest

    return pytest.main(rest)


def _show(args, rest):
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
    for name in available():
        from importlib.metadata import PackageNotFoundError, version

        dist = {"genesis": "genesis-world", "drake": "drake"}.get(name, _REQUIRES[name])
        try:
            v = version(dist)
        except PackageNotFoundError:
            v = "installed"
        print(f"backend {name}: {v}")
    try:
        from . import launch

        w = launch(settings=__import__("robowright").Settings(trace="off"))
        img = w.backend.render("front", 64, 48)
        w.close()
        print(f"offscreen rendering: ok ({img.shape[1]}x{img.shape[0]})")
    except Exception as e:  # rendering is optional: traces just lose their frames
        print(f"offscreen rendering: unavailable ({type(e).__name__}: {e})")
    return 0


def _robots(args, rest):
    from . import robots

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
    return 0


def _inspect(path) -> int:
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
    print(f"\nrun its tests with:  pytest --rw-robot {path}")
    print("override any of the above with robowright.robots.detect.load(path, <field>=...) in a conftest.py")
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
    sub.add_parser("test", help="run tests (all arguments are passed to pytest)", add_help=False)
    s = sub.add_parser("show-trace", help="open a trace in the HTML viewer")
    s.add_argument("trace")
    s.add_argument("-o", "--out")
    s.add_argument("--no-open", action="store_true", help="write the HTML but do not open a browser")
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
    rb.add_argument("--inspect", metavar="FILE", help="show what robowright works out about the robot in an MJCF file, and why")
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
    }
    return commands[args.cmd](args, [])


if __name__ == "__main__":
    sys.exit(main())
