"""``robowright test --ui``: Playwright's UI mode, for robot tests.

One page in your browser lists the suite by file. A click runs everything, a file or one test. Each test's status
shows as it runs, and its trace opens in place (every run records one). The camera can be watched live while a test
runs, and Watch runs again on every save. Each run is a fresh ``python -m pytest`` with the arguments ``test --ui``
was given, so a run here is the run on the command line.

The page and its API are served on 127.0.0.1 only. Running a test runs code, so every API call must carry the
page's own token, and requests naming another host are refused (a web page elsewhere cannot start one).
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import watch as _watch

_PAGE = Path(__file__).with_name("ui.html")


def _free_port(host: str) -> int:
    import socket

    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _without_xdist(args: list[str]) -> list[str]:
    """``args`` without pytest-xdist's worker options (collection lists tests in one process)."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in ("-n", "--numprocesses", "--dist", "--maxprocesses"):
            skip = True
        elif (a.startswith("-n") and not a.startswith("--")) or a.startswith(("--numprocesses=", "--dist=", "--maxprocesses=")):
            pass  # -n4, -nauto, --dist=load
        else:
            out.append(a)
    return out


class UI:
    """The state behind the page: the tests, their last results, the run in progress and the watcher."""

    def __init__(self, args: list[str], *, root: Path | str = ".", live: bool = False, host: str = "127.0.0.1"):
        self.args = list(args)
        self.root = Path(root)
        self.host = host
        self.dir = Path(tempfile.mkdtemp(prefix="robowright-ui-"))
        self.trace_dir = self.dir / "traces"
        self.live = live
        self.live_port = _free_port(host)
        self.watching = False
        self.tests: dict[str, dict] = {}
        # pytest's rootdir as collection found it: every run is pinned to it, or one test id run alone can find another
        # (with no ini file, a lone id's rootdir is worked out from its path) and report under ids the list does not know
        self.rootdir: str | None = None
        self.collect_error = ""
        self.running: list[str] | None = None  # what the run in progress was asked for ([] = everything)
        self.last: list[str] = []
        self.status: int | None = None
        self.log: deque[str] = deque(maxlen=2000)
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.version = 0  # bumped on every change, so the page knows when to redraw
        self._stop = threading.Event()

    # --- pytest ---------------------------------------------------------------------------------
    def _env(self, events: Path) -> dict:
        return {**_watch.run_env(), "ROBOWRIGHT_UI_EVENTS": str(events), "PYTHONUNBUFFERED": "1"}

    def _options(self) -> list[str]:
        return [a for a in self.args if not _watch._is_path(a)]

    def collect(self) -> None:
        """List the suite (``pytest --collect-only``) and keep each known test's last result."""
        events = self.dir / "collect.jsonl"
        events.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "robowright.ui_events", *_without_xdist(self.args)]
        done = subprocess.run(cmd, cwd=self.root, env=self._env(events), capture_output=True, text=True)
        collected = next((e for e in _read(events) if e["event"] == "collected"), {})
        found = collected.get("tests", [])
        with self.lock:
            self.rootdir = collected.get("rootdir") or self.rootdir
            self.tests = {n: self.tests.get(n, {"outcome": None}) for n in found}
            self.collect_error = "" if done.returncode in (0, 5) else (done.stdout + done.stderr)[-6000:]
            self.version += 1

    def run(self, ids: list[str] | None = None) -> bool:
        """Run ``ids`` (node ids or files; None or [] runs everything ``test --ui`` was given). False if one is running."""
        ids = list(ids or [])
        with self.lock:
            if self.proc is not None:
                return False
            events = self.dir / "run.jsonl"
            events.unlink(missing_ok=True)
            targets = ids or [a for a in self.args if _watch._is_path(a)]
            cmd = [sys.executable, "-m", "pytest", *self._options(), *targets, "-p", "robowright.ui_events"]
            if self.rootdir and not any(a.startswith("--rootdir") for a in self.args):
                cmd += ["--rootdir", self.rootdir]
            cmd += ["--rw-trace", "on", "--rw-trace-dir", str(self.trace_dir)]
            if self.live:
                cmd += ["--rw-live", str(self.live_port)]
            for n, t in self.tests.items():
                if not ids or any(n == i or n.startswith(i.rstrip("/") + "::") or n.startswith(i + "[") for i in ids):
                    t.update(outcome="queued")
            self.running, self.last, self.status = ids, ids, None
            self.log.clear()
            self.log.append("$ " + " ".join(cmd[1:]))
            self.proc = subprocess.Popen(
                cmd, cwd=self.root, env=self._env(events), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            self.version += 1
        threading.Thread(target=self._follow, args=(self.proc, events), daemon=True, name="robowright-ui-run").start()
        return True

    def _follow(self, proc: subprocess.Popen, events: Path) -> None:
        seen = 0

        def take():
            nonlocal seen
            new = _read(events)[seen:]
            seen += len(new)
            with self.lock:
                for e in new:
                    if e["event"] == "start" and e["nodeid"] in self.tests:
                        self.tests[e["nodeid"]]["outcome"] = "running"
                    elif e["event"] == "start":
                        self.tests[e["nodeid"]] = {"outcome": "running"}
                    elif e["event"] == "result":
                        self.tests[e["nodeid"]] = {k: v for k, v in e.items() if k not in ("event", "nodeid")}
                if new:
                    self.version += 1

        reader = threading.Thread(target=self._read_output, args=(proc,), daemon=True)
        reader.start()
        while proc.poll() is None:
            take()
            time.sleep(0.2)
        reader.join(timeout=5)
        take()
        with self.lock:
            for t in self.tests.values():
                if t.get("outcome") in ("queued", "running"):
                    t["outcome"] = None  # stopped, or never reached: not run
            self.status, self.proc, self.running = proc.returncode, None, None
            self.version += 1

    def _read_output(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            with self.lock:
                self.log.append(line.rstrip("\n"))
                self.version += 1

    def stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # --- watching -------------------------------------------------------------------------------
    def watch_loop(self, interval: float = 0.5, settle: float = 0.2) -> None:
        before = _watch.snapshot(self.root)
        while not self._stop.wait(interval):
            files = _watch.changed(before, _watch.snapshot(self.root))
            if not files:
                continue
            time.sleep(settle)  # take the whole save
            after = _watch.snapshot(self.root)
            files, before = _watch.changed(before, after), after
            if not self.watching or self.proc is not None:
                continue
            with self.lock:
                self.log.append("robowright: " + ", ".join(os.path.relpath(f, self.root) for f in files[:5]) + " changed")
            tests_only = all(_watch._is_test_file(f) for f in files)
            alive = [os.path.relpath(f, self.root) for f in files if os.path.exists(f)]
            self.collect()  # a new or renamed test shows in the list
            self.run(alive if tests_only and alive else self.last)

    # --- the page's view of it -------------------------------------------------------------------
    def state(self) -> dict:
        with self.lock:
            return {
                "version": self.version,
                "tests": [{"id": n, **t} for n, t in self.tests.items()],
                "collect_error": self.collect_error,
                "running": self.proc is not None,
                "status": self.status,
                "watching": self.watching,
                "live": self.live,
                "live_url": f"http://{self.host}:{self.live_port}/" if self.live else None,
                "live_ready": self.live and self.proc is not None and _listening(self.host, self.live_port),
                "log": "\n".join(self.log),
                "args": self.args,
            }

    def trace_file(self, path: str) -> Path | None:
        """``path``, if it is a trace this UI's runs wrote (nothing else is served)."""
        p = Path(path).resolve()
        return p if p.suffix == ".zip" and p.is_file() and self.trace_dir.resolve() in p.parents else None

    def close(self) -> None:
        self._stop.set()
        self.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def _listening(host: str, port: int) -> bool:
    """Whether the run's live view has started (it starts with the first test that builds a world)."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


_ICON = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#2f6fde"/>'
    b'<path d="M11 8v16l13-8z" fill="#fff"/></svg>'
)


def _read(events: Path) -> list[dict]:
    try:
        lines = events.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            break
    return out


def _handler(ui: UI, token: str, port_of):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _allowed(self) -> bool:
            host = self.headers.get("Host", "")
            if host not in (f"127.0.0.1:{port_of()}", f"localhost:{port_of()}", f"{ui.host}:{port_of()}"):
                return False  # another name for this address (DNS rebinding): refused
            q = parse_qs(urlparse(self.path).query)
            return secrets.compare_digest(self.headers.get("X-RW-Token", "") or q.get("token", [""])[0], token)

        def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/favicon.ico":
                return self._send(200, _ICON, "image/svg+xml")
            if url.path == "/":
                host = self.headers.get("Host", "")
                if host not in (f"127.0.0.1:{port_of()}", f"localhost:{port_of()}", f"{ui.host}:{port_of()}"):
                    return self._send(403, b"forbidden", "text/plain")
                page = _PAGE.read_text(encoding="utf-8").replace("__TOKEN__", token)
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if not self._allowed():
                return self._send(403, b"forbidden", "text/plain")
            if url.path == "/api/state":
                return self._json(ui.state())
            if url.path == "/trace":
                p = ui.trace_file(parse_qs(url.query).get("path", [""])[0])
                if p is None:
                    return self._send(404, b"no such trace", "text/plain")
                from .viewer import build_html

                return self._send(200, build_html(p).encode(), "text/html; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._allowed():
                return self._send(403, b"forbidden", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "not JSON"}, 400)
            path = urlparse(self.path).path
            if path == "/api/run":
                ids = body.get("ids") or []
                if not isinstance(ids, list) or not all(isinstance(i, str) and not i.startswith("-") for i in ids):
                    return self._json({"error": "ids: a list of test ids or files"}, 400)
                return self._json({"started": ui.run(ids)}, 200)
            if path == "/api/stop":
                threading.Thread(target=ui.stop, daemon=True).start()
                return self._json({"ok": True})
            if path == "/api/watch":
                ui.watching = bool(body.get("on"))
                ui.version += 1
                return self._json({"watching": ui.watching})
            if path == "/api/live":
                ui.live = bool(body.get("on"))
                ui.version += 1
                return self._json({"live": ui.live})
            if path == "/api/collect":
                ui.collect()
                return self._json({"ok": True})
            self._send(404, b"not found", "text/plain")

    return Handler


def serve(args: list[str], *, port: int = 0, host: str = "127.0.0.1", open_browser: bool = True, root=".") -> int:
    """Serve the UI until Ctrl-C."""
    ui = UI(args, root=root, host=host)
    token = secrets.token_urlsafe(24)
    httpd = ThreadingHTTPServer((host, port), _handler(ui, token, lambda: httpd.server_address[1]))
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"robowright UI: {url} (Ctrl-C to stop)", flush=True)
    ui.collect()
    threading.Thread(target=ui.watch_loop, daemon=True, name="robowright-ui-watch").start()
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        ui.close()
    return 0
