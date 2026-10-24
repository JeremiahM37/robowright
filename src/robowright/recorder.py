"""``robowright codegen --robot panda``: write a test by doing it, as ``npx playwright codegen`` writes one from what
you do in a browser.

A page in your browser shows the robot's camera and the scene's objects. Each thing you choose (pick the cube, put it
in the bin, check it landed) runs on the robot at once, and the test it makes so far is shown beside it; Save writes
it. The test is what robowright's codegen makes of the session's trace, so it is the same test the MCP tools save for
an agent: it runs the same calls, and expects the same outcomes.

The server is single-threaded on purpose: one simulation, driven by one request at a time.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .errors import RobowrightError
from .session import Session

# What the page may ask for: the session's own actions and its expect. Anything else is refused.
ACTIONS = ("pick", "place", "gripper", "home", "move_to", "wait", "push", "expect")
MATCHERS = (
    ("to_be_inside", "container"),
    ("to_be_near", "target"),
    ("to_be_above", "target"),
    ("to_be_touching", "other"),
    ("to_be_holding", "obj"),
    ("to_be_upright", None),
    ("to_be_at_rest", None),
    ("to_be_visible", None),
)


class Recorder:
    """A session driven from a browser page, its test kept up to date (see the module)."""

    def __init__(
        self, robot: str = "so101", backend: str = "mujoco", seed: int = 0, output: str | Path = "test_recorded.py", name: str | None = None
    ):
        self.session = Session()
        self.session.launch(robot=robot, backend=backend, seed=seed)
        self.output = Path(output)
        self.test_name = name or ("test_" + self.output.stem.removeprefix("test_"))
        self.log: list[dict] = []

    # --- what the page calls ---------------------------------------------------------
    def act(self, action: str, args: dict) -> dict:
        """Run one action on the robot. A failure is reported, not raised: the session goes on, as it does for an agent."""
        if action not in ACTIONS:
            return self._note(action, args, False, f"unknown action {action!r}; choose from {', '.join(ACTIONS)}")
        try:
            out = getattr(self.session, action)(**args)
        except (RobowrightError, KeyError, ValueError, TypeError) as e:
            return self._note(action, args, False, f"{type(e).__name__}: {e}")
        if action == "expect":
            return self._note(action, args, str(out).startswith("PASS"), str(out).splitlines()[0])
        return self._note(action, args, True, f"done at t={self.session.world.time:.2f} s")

    def state(self) -> dict:
        w = self.session.world
        return {
            "robot": w.backend.robot_model.title,
            "time": round(w.time, 3),
            "objects": [o.name for o in w.spec.objects],
            "cameras": [c.name for c in w.spec.cameras],
            "matchers": [m for m, _ in MATCHERS],
            "log": self.log,
            "code": self.code(),
            "output": str(self.output),
            **self.picks(),
        }

    def picks(self) -> dict:
        """For the page's locator picker, as the live view has it: where each object is in each camera's image and
        the locators that find it (see robowright.live)."""
        from .live import _picks

        w = self.session.world
        return _picks(w, (640, 480)) if w.has_ground_truth else {}

    def picked(self, locator: str) -> str:
        """A locator chosen in the page: printed, for an editor waiting on it (VS Code's Pick locator)."""
        print(f"robowright picked: {locator}", flush=True)
        return locator

    def code(self) -> str:
        return self.session.generate_test(test_name=self.test_name)

    def save(self) -> str:
        self.session.generate_test(test_name=self.test_name, path=str(self.output))
        return str(self.output)

    def frame(self, camera: str | None = None) -> bytes:
        cams = self.state()["cameras"]
        return self.session.screenshot(camera=camera if camera in cams else cams[0])

    def close(self) -> None:
        self.session.close()

    def _note(self, action, args, ok, text) -> dict:
        entry = {"action": action, "args": args, "ok": ok, "result": text}
        self.log.append(entry)
        return entry

    # --- the server ------------------------------------------------------------------
    def serve(self, port: int = 8790, host: str = "127.0.0.1", ready=None) -> None:
        """Serve the page until interrupted (``ready(url)`` is called once it listens)."""
        rec = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, ctype: str, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the browser asked for a newer frame and dropped this one

            def _json(self, obj, code: int = 200):
                self._send(code, "application/json", json.dumps(obj).encode())

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
                elif u.path == "/state":
                    self._json(rec.state())
                elif u.path == "/frame.png":
                    cam = parse_qs(u.query).get("camera", [None])[0]
                    self._send(200, "image/png", rec.frame(cam))
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    req = json.loads(body or b"{}")
                except ValueError:
                    return self._json({"error": "not JSON"}, 400)
                if self.path == "/act":
                    self._json(rec.act(str(req.get("action", "")), dict(req.get("args") or {})))
                elif self.path == "/save":
                    self._json({"saved": rec.save()})
                elif self.path == "/picked":
                    self._json({"picked": rec.picked(str(req.get("locator", "")))})
                else:
                    self._send(404, "text/plain", b"not found")

        server = HTTPServer((host, port), Handler)
        url = f"http://{host}:{server.server_address[1]}/"
        if ready is not None:
            ready(url)
        try:
            server.serve_forever()
        finally:
            server.server_close()


PAGE = (Path(__file__).parent / "recorder.html").read_text(encoding="utf-8")
