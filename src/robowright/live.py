"""Watch a world live in a browser while a test, an agent or your own code drives it.

Playwright's headed mode lets you watch the browser do what a test says. ``LiveView`` serves the
same for a robot: a page with the camera streaming as the simulation runs, the robot's joints,
the objects and their contacts, and what the robot was last asked to do::

    view = LiveView(world)          # http://127.0.0.1:8765, printed and in view.url
    ... drive the world ...
    view.close()

``pytest --rw-live`` serves every test's world on one port, ``robowright sim --live`` the
simulation it runs, and the MCP server's ``robot_watch`` the agent's session. Frames are drawn
on the world's own thread (MuJoCo's GL context belongs to it) every few control steps, so
watching costs a little time per step and nothing when nobody is connected.
"""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>robowright live</title>
<link rel="icon" href="/favicon.ico" type="image/svg+xml">
<style>
:root{--bg:#f6f7f9;--fg:#15181d;--mut:#5b6472;--card:#fff;--line:#e3e6eb;--acc:#0891b2}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--fg:#e8ebf0;--mut:#9aa3b2;--card:#171b21;--line:#2a3039;--acc:#22d3a0}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:minmax(0,3fr) minmax(0,2fr)}
@media (max-width:800px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow:hidden}
h1{font-size:16px;margin:0 0 8px} h2{font-size:13px;color:var(--mut);margin:12px 0 4px;text-transform:uppercase;letter-spacing:.04em}
img{width:100%;height:auto;border-radius:6px;background:#000;display:block}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums} td{padding:2px 6px 2px 0;border-bottom:1px solid var(--line)}
.mut{color:var(--mut)} .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--acc);margin-right:6px}
select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:2px 6px}
</style></head><body><main>
<section class="card"><h1><span class="dot"></span><span id="title">robowright live</span></h1>
<img id="cam" alt="the simulation, live"><div class="mut" id="time"></div>
<label class="mut">camera <select id="camsel"></select></label></section>
<section class="card"><h2>Last action</h2><div id="action" class="mut">-</div>
<h2>Joints</h2><table id="joints"></table><h2>Objects</h2><table id="objects"></table>
<h2>Contacts</h2><div id="contacts" class="mut"></div></section></main>
<script>
const $=id=>document.getElementById(id);let cam=null;
function setCam(c){cam=c;$('cam').src='/stream?camera='+encodeURIComponent(c)+'&t='+Date.now()}
$('camsel').onchange=e=>setCam(e.target.value);
async function poll(){try{const s=await (await fetch('/state')).json();
 $('title').textContent=s.name+' - '+s.robot+' on '+s.backend+(s.fidelity?' ('+s.fidelity+')':'');
 $('time').textContent='t = '+s.t.toFixed(2)+' s simulated, step '+s.step;
 if(!cam&&s.cameras.length){s.cameras.forEach(c=>{const o=document.createElement('option');
  o.textContent=c;$('camsel').append(o)});setCam(s.cameras[0])}
 $('action').textContent=s.action||'-';
 $('joints').innerHTML=s.joints.map(j=>'<tr><td>'+j[0]+'</td><td>'+j[1].toFixed(3)+'</td></tr>').join('');
 $('objects').innerHTML=s.objects.map(o=>'<tr><td>'+o[0]+'</td><td>('+o[1].map(v=>v.toFixed(3)).join(', ')+')</td></tr>').join('');
 $('contacts').textContent=s.contacts.length?s.contacts.join(', '):'none';}catch(e){}
 setTimeout(poll,250)}
poll();
</script></body></html>"""


_ICON = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="#0891b2"/></svg>'


class LiveView:
    """Serve ``world`` live over HTTP (see the module). ``world`` can be swapped with :meth:`watch`."""

    def __init__(self, world=None, port: int = 8765, host: str = "127.0.0.1", every: int = 2, size=(640, 480), quiet: bool = False):
        self.every, self.size = every, tuple(size)
        self._frames: dict[str, bytes] = {}
        self._state: dict = {}
        self._lock = threading.Condition()
        self._world = None
        self._clients = 0
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path == "/":
                    body = _PAGE.encode()
                    self._send(200, "text/html; charset=utf-8", body)
                elif path == "/favicon.ico":
                    self._send(200, "image/svg+xml", _ICON)
                elif path == "/state":
                    with view._lock:
                        body = json.dumps(view._state).encode()
                    self._send(200, "application/json", body)
                elif path == "/frame.jpg":
                    camera = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("camera")
                    frame = view._latest(camera)
                    self._send(200 if frame else 404, "image/jpeg", frame or b"")
                elif path == "/stream":
                    camera = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("camera")
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    view._clients += 1
                    try:
                        last = None
                        while True:
                            with view._lock:
                                view._lock.wait(1.0)
                                frame = view._latest(camera)
                            if frame is None or frame is last:
                                continue
                            last = frame
                            head = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode()
                            self.wfile.write(head + b"\r\n\r\n")
                            self.wfile.write(frame + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        view._clients -= 1
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.url = f"http://{host}:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, name="robowright-live", daemon=True)
        self._thread.start()
        if not quiet:
            print(f"robowright: watch live at {self.url}", flush=True)
        if world is not None:
            self.watch(world)

    def _latest(self, camera):
        if camera in self._frames:
            return self._frames[camera]
        return next(iter(self._frames.values()), None)

    def watch(self, world) -> None:
        """Show ``world`` from now on (the previous one is let go)."""
        if self._world is not None and self._hook in self._world._step_hooks:
            self._world._step_hooks.remove(self._hook)
        self._world = world
        world._step_hooks.append(self._hook)
        self._update(world, draw=True)

    def _hook(self, world):
        if world.step_count % self.every == 0:
            self._update(world, draw=self._clients > 0 or not self._frames)

    def _update(self, w, draw: bool):
        state = {
            "name": w.name,
            "robot": w.backend.robot_model.name,
            "backend": w.backend.name,
            "fidelity": getattr(w, "fidelity", None),
            "t": float(w.time),
            "step": int(w.step_count),
            "cameras": [c.name for c in w.spec.cameras],
            "joints": [[n, float(q)] for n, q in zip(w.backend.joint_names, w.backend.qpos())],
            "objects": [[n, [float(x) for x in w.backend.object_pose(n)[0]]] for n in w.object_names] if w.has_ground_truth else [],
            "contacts": [],
            "action": _last_action(w),
        }
        if w.has_contacts:
            pairs = {}
            for c in w.backend.contacts():
                pairs[(c.a, c.b)] = pairs.get((c.a, c.b), 0.0) + c.force
            state["contacts"] = [f"{a}-{b} {f:.1f} N" for (a, b), f in sorted(pairs.items()) if "floor" not in (a, b) or f > 1.0]
        frames = {}
        if draw and "render" in w.backend.capabilities:
            for c in w.spec.cameras:
                frames[c.name] = _jpeg(w.backend.render(c.name, *self.size))
        with self._lock:
            self._state = state
            if frames:
                self._frames = frames
            self._lock.notify_all()

    def close(self):
        if self._world is not None and self._hook in self._world._step_hooks:
            self._world._step_hooks.remove(self._hook)
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _last_action(w) -> str:
    tr = getattr(w, "trace", None)
    events = getattr(tr, "events", None) if tr is not None else None
    if not events:
        return ""
    e = events[-1]
    args = ", ".join(f"{k}={v}" for k, v in (getattr(e, "args", None) or {}).items())
    return f"{e.name}({args})"[:200] + ("  [running]" if getattr(e, "status", "") == "running" else "")


def _jpeg(img: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[..., :3])).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def serve_forever(world, view: LiveView | None = None) -> None:
    """Keep ``world`` running at real-time pace (for ``robowright sim``) until interrupted."""
    start_wall, start_sim = time.perf_counter(), world.time
    try:
        while True:
            world.step()
            ahead = (world.time - start_sim) - (time.perf_counter() - start_wall)
            if ahead > 0:
                time.sleep(ahead)
    except KeyboardInterrupt:
        pass
