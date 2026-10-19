"""A stand-in HTTP service for the code under test, as Playwright's ``route.fulfill`` answers a page's requests: a
robot's controller or stack that calls a fleet manager, a map server or a cloud API gets your answers instead, and
the test can assert on what it asked. Any process can use it (a ROS 2 stack started by the test included): point
the code's base URL at ``stub.url``::

    with rw.http_stub({
        "GET /missions/next": {"id": 7, "pick": "cube", "place": "bin"},
        "POST /missions/7/done": lambda req: (204, None),
        "GET /map": rw.http_stub.status(503),               # the map server is down
    }) as fleet:
        run_my_controller(fleet_url=fleet.url)
        assert fleet.called("POST /missions/7/done")

A route is ``"METHOD /path"`` (a ``*`` at the end matches the rest of the path) and its answer is a JSON body, an
``http_stub.status(code, body)``, or a function of the request returning one of those or ``(status, body)``. A
request no route matches gets 404 and is recorded, so a test can find it with ``stub.unmatched``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Request:
    method: str
    path: str  # without the query
    query: str
    headers: dict
    body: bytes
    t: float = field(default_factory=time.monotonic)
    route: str | None = None  # the route that answered it, None if none did

    @property
    def json(self):
        return json.loads(self.body or b"null")

    def __str__(self) -> str:
        return f"{self.method} {self.path}" + (f"?{self.query}" if self.query else "")


@dataclass
class Reply:
    code: int
    body: object = None
    delay: float = 0.0  # seconds before answering (a slow service)


def status(code: int, body=None, delay: float = 0.0) -> Reply:
    """An answer with this status (``status(503)``: the service is down), and a delay if given (a slow service)."""
    return Reply(code, body, delay)


class HttpStub:
    """A local HTTP server answering from ``routes`` (see the module). ``url``, ``requests`` (every one received),
    ``called(route)``, ``calls(route)``, ``unmatched``; a context manager that stops it at the end."""

    def __init__(self, routes: dict, host: str = "127.0.0.1", port: int = 0):
        self.routes = {self._key(k): v for k, v in routes.items()}
        self.requests: list[Request] = []
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _any(self):
                path, _, query = self.path.partition("?")
                n = int(self.headers.get("Content-Length") or 0)
                req = Request(self.command, path, query, dict(self.headers), self.rfile.read(n) if n else b"")
                code, body, ctype = stub._answer(req)
                with stub._lock:
                    stub.requests.append(req)
                data = b"" if body is None else body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                if data:
                    self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _any

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.url = f"http://{host}:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, name="robowright-http-stub", daemon=True)
        self._thread.start()

    @staticmethod
    def _key(route: str) -> tuple[str, str]:
        method, _, path = route.strip().partition(" ")
        if not path.startswith("/"):
            raise ValueError(f"a route is 'METHOD /path', e.g. 'GET /missions/next', not {route!r}")
        return method.upper(), path

    def _match(self, req: Request):
        for (method, path), answer in self.routes.items():
            if method != req.method:
                continue
            if path == req.path or (path.endswith("*") and req.path.startswith(path[:-1])):
                return f"{method} {path}", answer
        return None, None

    def _answer(self, req: Request) -> tuple[int, object, str]:
        route, answer = self._match(req)
        req.route = route
        if route is None:
            return 404, {"error": f"no route for {req}"}, "application/json"
        if callable(answer) and not isinstance(answer, Reply):
            try:
                answer = answer(req)
            except Exception as e:  # noqa: BLE001 - the handler's error goes back as a 500, and is recorded
                return 500, {"error": f"{type(e).__name__}: {e}"}, "application/json"
        if isinstance(answer, tuple):
            answer = Reply(*answer)
        if not isinstance(answer, Reply):
            answer = Reply(200, answer)
        if answer.delay:
            time.sleep(answer.delay)
        ctype = "application/octet-stream" if isinstance(answer.body, bytes) else "application/json"
        return answer.code, answer.body, ctype

    def calls(self, route: str | None = None) -> list[Request]:
        """The requests ``route`` (``"POST /missions/7/done"``) answered, or all of them."""
        with self._lock:
            reqs = list(self.requests)
        if route is None:
            return reqs
        method, path = self._key(route)
        return [r for r in reqs if r.route == f"{method} {path}"]

    def called(self, route: str) -> bool:
        return bool(self.calls(route))

    @property
    def unmatched(self) -> list[Request]:
        """Requests no route answered (they got 404): a call the test did not expect."""
        with self._lock:
            return [r for r in self.requests if r.route is None]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def http_stub(routes: dict, host: str = "127.0.0.1", port: int = 0) -> HttpStub:
    """Start a stand-in HTTP service answering from ``routes`` (see the module)."""
    return HttpStub(routes, host, port)


http_stub.status = status
