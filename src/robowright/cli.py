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
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["test"]:
        return _test(None, argv[1:])
    args = p.parse_args(argv)
    return {"show-trace": _show, "replay": _replay, "codegen": _codegen, "info": _info, "robots": _robots}[args.cmd](args, [])


if __name__ == "__main__":
    sys.exit(main())
