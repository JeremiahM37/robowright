"""Command line: ``robowright test | show-trace | replay | codegen | info``."""

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
    from .backends.base import available

    print(f"robowright {__version__}")
    print(f"python {platform.python_version()}, numpy {numpy.__version__}")
    for name in available():
        mod = __import__(name)
        print(f"backend {name}: {getattr(mod, '__version__', 'installed')}")
    try:
        from . import launch

        w = launch(settings=__import__("robowright").Settings(trace="off"))
        img = w.backend.render("front", 64, 48)
        w.close()
        print(f"offscreen rendering: ok ({img.shape[1]}x{img.shape[0]})")
    except Exception as e:  # rendering is optional: traces just lose their frames
        print(f"offscreen rendering: unavailable ({type(e).__name__}: {e})")
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
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["test"]:
        return _test(None, argv[1:])
    args = p.parse_args(argv)
    return {"show-trace": _show, "replay": _replay, "codegen": _codegen, "info": _info}[args.cmd](args, [])


if __name__ == "__main__":
    sys.exit(main())
