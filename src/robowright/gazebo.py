"""Drive a running Gazebo (gz sim) the way Playwright drives a browser: robowright does not
simulate anything here, it controls a simulator someone else built and runs.

:class:`Gazebo` talks to the simulator through Gazebo's own transport (``gz.transport``), not
through the robot's ROS 2 interfaces, so it reaches what a robot cannot: every model's true
pose, moving and spawning objects, pausing the world and stepping it a set number of physics
iterations, and resetting it. The ROS 2 backend uses it for ground truth and for scene edits
(``[ros2] gazebo = true``); a test, an agent (the ``sim_*`` MCP tools) or a script can use it
directly::

    from robowright.gazebo import Gazebo

    gz = Gazebo()                      # the one world running (or Gazebo(world="empty"))
    gz.pause(); gz.step(500)           # exactly 500 physics iterations, then still
    gz.spawn_object(ObjectSpec("cube", pos=(0.4, 0.1, None)))
    gz.set_pose("cube", (0.3, -0.2, 0.0125))
    gz.pose("cube")                    # (position, quaternion w x y z), in Gazebo's world frame
    gz.add_camera("front", pos=(1.0, -0.6, 0.6), lookat=(0.3, 0.0, 0.1))
    gz.image("front")                  # what Gazebo renders, as an RGB array

Needs Gazebo's Python bindings (``gz.transport13``/``gz.msgs10``, with Gazebo Harmonic), and a
world with the ``UserCommands`` and ``SceneBroadcaster`` systems (gz sim's ``empty.sdf`` has both).
Several simulations on one host keep apart by ``GZ_PARTITION``.

Service requests go through a helper process of their own. The bindings hold Python's GIL for the
whole of a request, and a reply is read by the same transport thread that runs subscription
callbacks: a pose message arriving mid-request waits for the GIL, and the reply behind it is
never read (the request times out, though Gazebo answered at once). A process with no
subscriptions has no callbacks to wait on.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time as _time

import numpy as np

from .scene import ObjectSpec

_TIMEOUT_MS = 5000
_VELOCITY_SPAN = 0.02  # s of Gazebo's clock a velocity is measured over


def _modules():
    try:
        from gz.msgs10 import boolean_pb2, entity_factory_pb2, entity_pb2, pose_pb2, pose_v_pb2, world_control_pb2, world_stats_pb2
        from gz.transport13 import Node
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("robowright.gazebo needs Gazebo Harmonic's Python bindings (gz.transport13, gz.msgs10)") from e
    return Node, boolean_pb2, entity_factory_pb2, entity_pb2, pose_pb2, pose_v_pb2, world_control_pb2, world_stats_pb2


class _Requests:
    """Gazebo service requests, made by a helper process (``python -m robowright.gazebo``)."""

    def __init__(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "robowright.gazebo", "requests"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._lock = threading.Lock()

    def __call__(self, service: str, req, rep_type, timeout_ms: int) -> tuple[bool, object]:
        line = {
            "service": service,
            "request": base64.b64encode(req.SerializeToString()).decode(),
            "request_type": req.DESCRIPTOR.full_name,
            "response_type": rep_type.DESCRIPTOR.full_name,
            "timeout": int(timeout_ms),
        }
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError(f"robowright's Gazebo request helper exited ({self._proc.returncode})")
            self._proc.stdin.write(json.dumps(line) + "\n")
            out = json.loads(self._proc.stdout.readline())
        rep = rep_type()
        if out["ok"]:
            rep.ParseFromString(base64.b64decode(out["response"]))
        return out["ok"], rep

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


def _serve_requests() -> None:
    """The helper: one request per line on stdin, its result on stdout. No subscriptions."""
    from gz.transport13 import Node

    node = Node()
    discovered: set[str] = set()
    for line in sys.stdin:
        r = json.loads(line)
        if r["service"] not in discovered:  # a request reaches only a discovered service
            deadline = _time.monotonic() + r["timeout"] / 1000
            while not node.service_info(r["service"]) and _time.monotonic() < deadline:
                node.service_list()
            discovered.add(r["service"])
        ok, data = node.request_raw(r["service"], base64.b64decode(r["request"]), r["request_type"], r["response_type"], r["timeout"])
        sys.stdout.write(json.dumps({"ok": bool(ok), "response": base64.b64encode(data).decode()}) + "\n")
        sys.stdout.flush()


class Gazebo:
    """A running Gazebo world. ``world``: its name (default: the only one running)."""

    def __init__(self, world: str | None = None, timeout: float = 30.0):
        (Node, self._Boolean, self._Factory, self._Entity, self._Pose, PoseV, self._Control, Stats) = _modules()
        self._node = Node()  # subscriptions only (see _Requests)
        self._request = _Requests()
        self._lock = threading.Lock()
        self._poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._history: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}  # recent (time, position, orientation)
        self._stamp = 0.0
        self._removing: set[str] = set()  # models removed that Gazebo's poses may still show
        self._stats: dict = {}
        self._images: dict[str, np.ndarray] = {}
        self._cameras: dict[str, str] = {}  # camera -> image topic
        self._sensors = False  # whether robowright added the Sensors system to the world
        self._fresh = threading.Condition(self._lock)
        deadline = _time.monotonic() + timeout
        while True:
            worlds = sorted({s.split("/")[2] for s in self._node.service_list() if s.startswith("/world/") and s.endswith("/control")})
            if world in worlds or (world is None and len(worlds) == 1):
                self.world = world or worlds[0]
                break
            if world is None and len(worlds) > 1:
                raise RuntimeError(f"several Gazebo worlds are running ({', '.join(worlds)}): pass world=")
            if _time.monotonic() > deadline:
                found = f"found {', '.join(worlds)}" if worlds else "none found"
                raise TimeoutError(f"no Gazebo world {world or ''} within {timeout} s ({found}; is gz sim running, in this GZ_PARTITION?)")
            _time.sleep(0.1)
        w = f"/world/{self.world}"
        # A request goes nowhere until its service is discovered (it waits out its timeout and
        # fails): wait for each one this uses first.
        services = [f"{w}/{s}" for s in ("control", "create/blocking", "set_pose/blocking", "remove/blocking")]
        while missing := [s for s in services if not self._node.service_info(s)]:
            if _time.monotonic() > deadline:
                raise TimeoutError(f"Gazebo world {self.world!r} offers no {', '.join(missing)} (does it load the UserCommands system?)")
            self._node.service_list()
            _time.sleep(0.1)
        self._node.subscribe(PoseV.Pose_V, f"{w}/pose/info", self._on_poses)
        self._node.subscribe(Stats.WorldStatistics, f"{w}/stats", self._on_stats)
        with self._lock:
            if not self._fresh.wait_for(lambda: self._stats, timeout=max(1.0, deadline - _time.monotonic())):
                raise TimeoutError(f"Gazebo world {self.world!r} publishes no statistics")
            if not self._stats["paused"]:  # (a paused world may publish no poses until it steps)
                self._fresh.wait_for(lambda: self._poses, timeout=max(1.0, deadline - _time.monotonic()))

    # --- what Gazebo says ------------------------------------------------------------------
    def _on_poses(self, msg) -> None:
        got = {}
        for p in msg.pose:
            if p.name and p.name not in got:  # a model's entry comes before its links' (which can share a name)
                o = p.orientation
                got[p.name] = (np.array([p.position.x, p.position.y, p.position.z]), np.array([o.w, o.x, o.y, o.z]))
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        with self._lock:
            if t < self._stamp:
                return  # older than what is known (or from before a reset): not the world as it is
            self._stamp = t
            # Poses published before a removal can arrive after it: the first message without the
            # model shows it gone, and the topic is in order
            self._removing &= set(got)
            for name in self._removing:
                got.pop(name)
            for name, (p, q) in got.items():
                h = self._history.setdefault(name, [])
                h.append((t, p, q))
                while len(h) > 2 and t - h[1][0] >= _VELOCITY_SPAN:
                    h.pop(0)  # keep the newest sample at least _VELOCITY_SPAN old, and what is newer
            self._poses.update(got)
            self._fresh.notify_all()

    def _on_stats(self, msg) -> None:
        with self._lock:
            self._stats = {
                "time": msg.sim_time.sec + msg.sim_time.nsec * 1e-9,
                "iterations": int(msg.iterations),
                "paused": bool(msg.paused),
                "real_time_factor": float(msg.real_time_factor),
            }
            self._fresh.notify_all()

    @property
    def time(self) -> float:
        """Simulated seconds since the world started (or was reset)."""
        with self._lock:
            return self._stats["time"]

    @property
    def iterations(self) -> int:
        with self._lock:
            return self._stats["iterations"]

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._stats["paused"]

    @property
    def real_time_factor(self) -> float:
        with self._lock:
            return self._stats["real_time_factor"]

    def models(self) -> list[str]:
        """Names Gazebo publishes poses for: its models (and their links, where named apart)."""
        with self._lock:
            return sorted(self._poses)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._poses

    def pose(self, name: str, timeout: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """``name``'s position and orientation (w, x, y, z) in the world frame, as Gazebo has it."""
        with self._lock:
            if not self._fresh.wait_for(lambda: name in self._poses, timeout=timeout):
                raise KeyError(f"Gazebo world {self.world!r} has no model {name!r}; it has {sorted(self._poses)}")
            p, q = self._poses[name]
        return p.copy(), q.copy()

    def velocity(self, name: str) -> np.ndarray:
        """``name``'s linear then angular velocity (world frame): the change in its pose over the
        last ``_VELOCITY_SPAN`` or more of Gazebo's clock (zero until there are two poses since it
        was last placed)."""
        self.pose(name)
        with self._lock:
            h = list(self._history.get(name, ()))
        if len(h) < 2:
            return np.zeros(6)
        (t0, p0, q0), (t1, p, q) = h[0], h[-1]
        dt = t1 - t0
        if dt <= 0:
            return np.zeros(6)
        dq = q if np.dot(q, q0) >= 0 else -q
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = dq
        # The turn from q0 to q (q * conj(q0)), as an axis times its angle, over dt.
        rel = np.array(
            [
                w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0,
                -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0,
                -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0,
                -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0,
            ]
        )
        s = np.linalg.norm(rel[1:])
        ang = rel[1:] / s * 2 * np.arctan2(s, rel[0]) if s > 1e-12 else np.zeros(3)
        return np.concatenate([(p - p0) / dt, ang / dt])

    # --- seeing it ---------------------------------------------------------------------------
    def add_camera(self, name: str, pos, lookat, fovy: float = 45.0, width: int = 320, height: int = 240, hz: float = 10.0) -> None:
        """A camera in the world (a static model with a camera sensor, ``rw_camera_<name>``) at
        ``pos`` looking at ``lookat``, whose frames :meth:`image` returns: Gazebo's own rendering.
        A world that renders no sensors gets Gazebo's Sensors system added (ogre2, headless)."""
        from gz.msgs10 import image_pb2

        model = f"rw_camera_{name}"
        pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
        d = lookat - pos
        yaw, pitch = np.arctan2(d[1], d[0]), np.arctan2(-d[2], np.hypot(d[0], d[1]))  # a camera looks along its +x
        cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
        quat = (cy * cp, -sy * sp, cy * sp, sy * cp)  # turned yaw about z, then pitch about y
        topic = f"rw/{self.world}/{name}/image"
        hfov = 2 * np.arctan(np.tan(np.radians(fovy) / 2) * width / height)
        if self.has(model):
            self.set_pose(model, pos, quat)
        else:
            sdf = (
                f"<?xml version='1.0'?><sdf version='1.9'><model name='{model}'><static>true</static><link name='link'>"
                f"<sensor name='camera' type='camera'><topic>{topic}</topic><update_rate>{hz}</update_rate><always_on>1</always_on>"
                f"<camera><horizontal_fov>{hfov}</horizontal_fov><image><width>{width}</width><height>{height}</height></image>"
                "<clip><near>0.02</near><far>20</far></clip></camera></sensor></link></model></sdf>"
            )
            self.spawn(model, sdf, pos, quat)
        if name not in self._cameras:
            self._node.subscribe(image_pb2.Image, topic, lambda msg, n=name: self._on_image(n, msg))
            self._cameras[name] = topic

    def _on_image(self, name: str, msg) -> None:
        ch = len(msg.data) // max(msg.width * msg.height, 1)
        if ch not in (3, 4):
            return  # a pixel format robowright does not read
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, ch)[..., :3]
        with self._lock:
            self._images[name] = img
            self._fresh.notify_all()

    def image(self, name: str, timeout: float = 10.0) -> np.ndarray:
        """The latest frame from camera ``name`` (see :meth:`add_camera`), RGB."""
        if name not in self._cameras:
            raise KeyError(f"no camera {name!r}; robowright added {sorted(self._cameras)}")
        with self._lock:
            ok = self._fresh.wait_for(lambda: name in self._images, timeout=min(timeout, 3.0) if not self._sensors else timeout)
        if not ok and not self._sensors:
            self._add_sensors()
            with self._lock:
                ok = self._fresh.wait_for(lambda: name in self._images, timeout=timeout)
        if not ok:
            raise TimeoutError(
                f"no image from Gazebo camera {name!r} within {timeout} s (can this Gazebo render? see its log: a server "
                "started with DISPLAY set to an X server without GLX crashes; start it with --headless-rendering or without DISPLAY)"
            )
        with self._lock:
            return self._images[name].copy()

    def _add_sensors(self) -> None:
        """Gazebo's Sensors system, added to the running world (renders camera sensors)."""
        from gz.msgs10 import entity_plugin_v_pb2

        req = entity_plugin_v_pb2.EntityPlugin_V()
        req.entity.name, req.entity.type = self.world, self._Entity.Entity.WORLD
        req.entity.id = 1  # the world is entity 1
        plugin = req.plugins.add()
        plugin.name, plugin.filename = "gz::sim::systems::Sensors", "gz-sim-sensors-system"
        plugin.innerxml = "<render_engine>ogre2</render_engine>"
        self._call("entity/system/add", req, entity_plugin_v_pb2.EntityPlugin_V, "add the Sensors system")
        self._sensors = True

    # --- changing the world ----------------------------------------------------------------
    def _call(self, service: str, req, req_type, what: str) -> None:
        name = f"/world/{self.world}/{service}"
        ok, rep = self._request(name, req, self._Boolean.Boolean, _TIMEOUT_MS)
        if not ok:
            raise TimeoutError(f"Gazebo did not answer {name} within {_TIMEOUT_MS / 1000:g} s (asked to {what})")
        if not rep.data:
            raise RuntimeError(f"Gazebo refused to {what} ({name})")

    def _control(self, what: str, **fields) -> None:
        req = self._Control.WorldControl(**{k: v for k, v in fields.items() if k != "reset"})
        if fields.get("reset"):
            req.reset.all = True
        self._call("control", req, self._Control.WorldControl, what)

    def _wait(self, check, what: str, timeout: float = 10.0) -> None:
        with self._lock:
            if not self._fresh.wait_for(lambda: check(self._stats), timeout=timeout):
                raise TimeoutError(f"Gazebo did not {what} within {timeout} s")

    def pause(self) -> None:
        self._control("pause", pause=True)
        self._wait(lambda s: s["paused"], "pause")

    def play(self) -> None:
        self._control("play", pause=False)
        self._wait(lambda s: not s["paused"], "play")

    def step(self, iterations: int = 1, timeout: float = 30.0) -> None:
        """Advance a paused world exactly ``iterations`` physics steps, and wait until it has."""
        if not self.paused:
            self.pause()
        target = self.iterations + int(iterations)
        self._control(f"step {iterations} iterations", pause=True, multi_step=int(iterations))
        # Gazebo reports the world unpaused while it runs the steps, and a statistics message can
        # show the last of them before the world is paused again: wait for both.
        self._wait(lambda s: s["iterations"] >= target and s["paused"], f"step {iterations} iterations", timeout)

    def reset(self) -> None:
        """Put the world back as it was loaded (time, models, joints). Gazebo leaves models spawned
        since in place; the ROS 2 backend puts the scene's objects back itself."""
        self._control("reset", pause=self.paused, reset=True)
        self._wait(lambda s: s["iterations"] < 10 or s["time"] < 0.05, "reset")

    def set_pose(self, name: str, pos, quat=None) -> None:
        """Move model ``name`` (teleport it, at rest) to ``pos`` turned ``quat`` (w, x, y, z)."""
        p = self._Pose.Pose(name=name)
        p.position.x, p.position.y, p.position.z = map(float, pos)
        w, x, y, z = (1.0, 0.0, 0.0, 0.0) if quat is None else map(float, quat)
        p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = w, x, y, z
        self._call("set_pose/blocking", p, self._Pose.Pose, f"move {name!r}")
        self._placed(name, pos, (w, x, y, z))

    def spawn(self, name: str, sdf: str, pos=(0.0, 0.0, 0.0), quat=None, timeout: float = 10.0) -> None:
        """Add a model (``sdf``: an SDF ``<model>`` document) named ``name`` at ``pos``."""
        req = self._Factory.EntityFactory(sdf=sdf, name=name, allow_renaming=False)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = map(float, pos)
        w, x, y, z = (1.0, 0.0, 0.0, 0.0) if quat is None else map(float, quat)
        req.pose.orientation.w, req.pose.orientation.x, req.pose.orientation.y, req.pose.orientation.z = w, x, y, z
        with self._lock:
            self._removing.discard(name)
        self._call("create/blocking", req, self._Factory.EntityFactory, f"spawn {name!r}")
        self._placed(name, pos, (w, x, y, z))

    def _placed(self, name: str, pos, quat) -> None:
        """A model put somewhere (not moved there): its pose is that, and its motion starts again
        from there (a teleport is not a velocity). A paused world publishes no new pose."""
        with self._lock:
            pose = (np.array(pos, float), np.array(quat, float))
            self._poses[name] = pose
            self._history[name] = [(self._stamp, *pose)]
            self._fresh.notify_all()

    def spawn_object(self, obj: ObjectSpec) -> None:
        """A robowright scene object (box, cylinder, sphere or bin) as a Gazebo model."""
        pos = (obj.pos[0], obj.pos[1], obj.half_height if obj.kind != "bin" else 0.0) if obj.pos[2] is None else obj.pos
        self.spawn(obj.name, model_sdf(obj), pos, (np.cos(obj.yaw / 2), 0.0, 0.0, np.sin(obj.yaw / 2)))

    def remove(self, name: str, timeout: float = 5.0) -> None:
        """Remove model ``name``, and wait until Gazebo's poses show it gone (a paused world
        publishes none)."""
        req = self._Entity.Entity(name=name, type=self._Entity.Entity.MODEL)
        with self._lock:
            self._removing.add(name)
        self._call("remove/blocking", req, self._Entity.Entity, f"remove {name!r}")
        with self._lock:
            self._poses.pop(name, None)
            self._history.pop(name, None)
            if not self._stats.get("paused", False):
                self._fresh.wait_for(lambda: name not in self._removing, timeout=timeout)
            self._removing.discard(name)

    def close(self) -> None:
        for topic in list(self._node.subscribed_topics()):
            self._node.unsubscribe(topic)
        self._request.close()


def _box(hx, hy, hz) -> str:
    return f"<geometry><box><size>{2 * hx} {2 * hy} {2 * hz}</size></box></geometry>"


def model_sdf(o: ObjectSpec) -> str:
    """An SDF document with the ``<model>`` of a robowright scene object (see :func:`model`)."""
    return f"<?xml version='1.0'?><sdf version='1.9'>{model(o)}</sdf>"


def model(o: ObjectSpec, plugins: str = "", pose: bool = False) -> str:
    """The SDF ``<model>`` of a robowright scene object: its shape, mass, inertia, friction and
    colour. A bin is a static open box (a floor and four walls). ``plugins``: SDF to put in the
    model; ``pose``: give the model its scene pose (in a world file)."""
    r, g, b, a = o.rgba
    material = f"<material><diffuse>{r} {g} {b} {a}</diffuse><ambient>{r} {g} {b} {a}</ambient></material>"
    if o.kind == "bin":
        from .backends.mujoco_backend import bin_walls

        parts = "".join(
            f'<collision name="w{i}"><pose>{px} {py} {pz} 0 0 0</pose>{_box(hx, hy, hz)}</collision>'
            f'<visual name="v{i}"><pose>{px} {py} {pz} 0 0 0</pose>{_box(hx, hy, hz)}{material}</visual>'
            for i, ((px, py, pz), (hx, hy, hz)) in enumerate(bin_walls(o.size))
        )
        body = f"<static>true</static><link name='link'>{parts}</link>"
    else:
        if o.kind == "box":
            geo = f"<box><size>{2 * o.size[0]} {2 * o.size[1]} {2 * o.size[2]}</size></box>"
            ixx = o.mass / 3 * (o.size[1] ** 2 + o.size[2] ** 2)
            iyy = o.mass / 3 * (o.size[0] ** 2 + o.size[2] ** 2)
            izz = o.mass / 3 * (o.size[0] ** 2 + o.size[1] ** 2)
        elif o.kind == "cylinder":
            geo = f"<cylinder><radius>{o.size[0]}</radius><length>{2 * o.size[1]}</length></cylinder>"
            ixx = iyy = o.mass * (3 * o.size[0] ** 2 + (2 * o.size[1]) ** 2) / 12
            izz = o.mass * o.size[0] ** 2 / 2
        else:
            geo = f"<sphere><radius>{o.size[0]}</radius></sphere>"
            ixx = iyy = izz = 0.4 * o.mass * o.size[0] ** 2
        body = (
            "<link name='link'>"
            f"<inertial><mass>{o.mass}</mass><inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz></inertia></inertial>"
            f"<collision name='c'><geometry>{geo}</geometry><surface><friction><ode>"
            f"<mu>{o.friction}</mu><mu2>{o.friction}</mu2></ode></friction></surface></collision>"
            f"<visual name='v'><geometry>{geo}</geometry>{material}</visual></link>"
        )
    at = ""
    if pose:
        x, y, z = o.initial_pos
        at = f"<pose>{x} {y} {z} 0 0 {o.yaw}</pose>"
    return f"<model name='{o.name}'>{at}{body}{plugins}</model>"


if __name__ == "__main__" and sys.argv[1:2] == ["requests"]:
    _serve_requests()
