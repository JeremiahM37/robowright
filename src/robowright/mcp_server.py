"""MCP server: lets an AI agent drive a simulated robot the way Playwright MCP drives a browser.

Run ``robowright mcp`` (stdio) and register it with an MCP client, e.g. for
Claude Code::

    claude mcp add robowright -- robowright mcp

The agent launches a world, reads its snapshot (the robot's state and every
object by name), acts on it - ``pick``, ``place``, ``move_to``, pushes and
faults - checks matchers with ``expect``, looks through cameras with
``screenshot``, and finally saves what it did as a pytest test that
reproduces the session bit for bit.

Every call runs on one worker thread: a world is not thread-safe, and
MuJoCo's offscreen GL context belongs to the thread that made it.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from .session import Session

INSTRUCTIONS = """\
Drive a simulated robot (MuJoCo by default; deterministic physics).
Start with robot_launch, then robot_snapshot to see the world: objects are named
(those names are the refs other tools take). Arms: robot_pick, robot_place,
robot_move_to, robot_gripper, robot_move_joints, robot_home. Legged robots:
robot_stand, robot_crouch, robot_move_joints. Disturb the world with robot_push,
robot_move_object and robot_fault. Check outcomes with robot_expect. Look with
robot_screenshot (cameras: front, top, side). A failed action reports why and
leaves the session running. robot_generate_test turns the session into a pytest
test that reproduces it exactly; robot_crosscheck re-runs it on another engine.
robot_run_tests runs a project's tests and returns each failure with its trace as
text; robot_read_trace reads any trace that way. Positions are metres in the world frame, z up."""


def build_server(session: Session | None = None):
    try:  # mcp >= 2
        from mcp.server.mcpserver import Image
        from mcp.server.mcpserver import MCPServer as Server
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server
        from mcp.server.fastmcp import Image
        from mcp.server.fastmcp.exceptions import ToolError

    s = session or Session()
    server = Server(name="robowright", instructions=INSTRUCTIONS)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robowright")

    async def run(fn, *args, **kwargs):
        try:
            return await asyncio.get_running_loop().run_in_executor(worker, functools.partial(fn, *args, **kwargs))
        except Exception as e:
            # Pass the reason on: the SDK hides any other exception's message, and an agent
            # cannot correct a call it is not told is wrong.
            msg = e.args[0] if isinstance(e, KeyError) and e.args else e
            raise ToolError(f"{type(e).__name__}: {msg}") from e

    def tool(fn):
        return server.tool(name=fn.__name__)(fn)

    @tool
    async def robot_list_robots(family: str | None = None) -> str:
        """List the robots available. family: 'arm' or 'legged' (default: all)."""
        from . import robots

        rows = []
        for name in robots.names(family):
            m = robots.get(name)
            rows.append(f"{name}: {m.title} ({m.family}{', ' + ', '.join(m.tags) if m.tags else ''})")
        return "\n".join(rows)

    @tool
    async def robot_launch(robot: str = "so101", backend: str = "mujoco", seed: int = 0, objects: list[dict] | None = None) -> str:
        """Start a fresh simulated world, closing any open one, and return its snapshot.

        robot: a name from robot_list_robots. backend: mujoco (default, fastest), pybullet,
        drake, genesis or isaac, whichever are installed. seed: same seed and same calls
        give the same world. objects: replace the default scene's objects, each
        {"name", "kind": box|cylinder|sphere|bin, "size": half-extents in m (box/bin xyz,
        cylinder [radius, half_height], sphere [radius]), "pos": [x, y] to rest it on the
        floor, or [x, y, z] (z is a free object's centre, a bin's base),
        "color": red|green|blue|yellow|..., "mass": kg}.
        """
        return await run(s.launch, robot=robot, backend=backend, seed=seed, objects=objects)

    @tool
    async def robot_snapshot() -> str:
        """The world as text: robot state, every object by name with pose and contacts, active faults."""
        return await run(s.snapshot)

    @tool
    async def robot_screenshot(camera: str = "front", width: int = 640, height: int = 480):
        """A camera image of the world. Cameras: front (whole robot), top (task area from above), side."""
        return Image(data=await run(s.screenshot, camera, width, height), format="png")

    @tool
    async def robot_pick(object: str, approach: str | list[float] = "top") -> str:
        """Arm: grasp the named object and lift it. approach: "top" (from above), "side"
        (horizontally, for a tall object), or a direction [x, y, z] to come in along, level or
        tilted down ([1, 0, -1]: 45 degrees); side and tilted grasps plan their way round the
        table and objects."""
        return await run(s.pick, object, approach)

    @tool
    async def robot_place(on: str | list[float], height: float | None = None) -> str:
        """Arm: put the held object on/in a named object (e.g. a bin) or at [x, y, z], and let go."""
        return await run(s.place, on, height)

    @tool
    async def robot_move_to(target: str | list[float], linear: bool = False, speed: float = 0.15) -> str:
        """Arm: move the tool centre point to a named object or [x, y, z] (gripper pointing down).

        linear: straight-line path at speed m/s; otherwise a joint-space move.
        """
        return await run(s.move_to, target, linear, speed)

    @tool
    async def robot_gripper(action: str, amount: float = 1.0) -> str:
        """Arm: 'open' (amount 0..1 of fully open) or 'close' (until it stalls on an object or shuts)."""
        return await run(s.gripper, action, amount)

    @tool
    async def robot_home() -> str:
        """Arm: return to the home pose."""
        return await run(s.home)

    @tool
    async def robot_move_joints(joints: dict[str, float]) -> str:
        """Move named joints to targets in radians, e.g. {"shoulder_pan": 0.3}; the rest hold."""
        return await run(s.move_joints, joints)

    @tool
    async def robot_stand() -> str:
        """Legged: return to the standing pose."""
        return await run(s.stand)

    @tool
    async def robot_crouch(depth: float = 0.5) -> str:
        """Legged: bend toward the folded pose, depth 0 (standing) to 1 (fully folded)."""
        return await run(s.crouch, depth)

    @tool
    async def robot_wait(seconds: float) -> str:
        """Let simulated time pass with the robot holding its targets."""
        return await run(s.wait, seconds)

    @tool
    async def robot_push(target: str, force: list[float], duration: float = 0.1) -> str:
        """Apply a force [fx, fy, fz] in newtons (world frame) to a named object, or 'robot' (a legged robot's base)."""
        return await run(s.push, target, force, duration)

    @tool
    async def robot_move_object(object: str, position: list[float], yaw: float | None = None) -> str:
        """Teleport a named object to [x, y, z] (yaw in radians), e.g. to set up a scenario."""
        return await run(s.move_object, object, position, yaw)

    @tool
    async def robot_fault(kind: str, params: dict | None = None) -> str:
        """Inject a fault for the rest of the session.

        kind/params: joint_noise {std}, action_delay {steps}, weak_joint {joint, scale},
        jitter {object, xy_std, yaw_std}, camera_dropout {p}.
        """
        return await run(s.fault, kind, **(params or {}))

    @tool
    async def robot_expect(subject: str, matcher: str, args: dict | None = None, negate: bool = False, timeout: float | None = None) -> str:
        """Check an outcome; returns PASS or FAIL with the reason. Simulated time runs while it retries.

        subject: an object name, or robot, gripper, tcp, base. matcher and args, e.g.
        to_be_inside {container}, to_be_holding {obj}, to_be_near {target, tol},
        to_be_above {target, by}, to_be_at_rest {}, to_be_upright {tol_deg},
        to_be_touching {other}, to_be_open {}, to_be_closed {}, to_have_joint {joint, value, tol},
        to_have_no_collisions {}. Object names in args are resolved.
        """
        return await run(s.expect, subject, matcher, args, negate, timeout)

    @tool
    async def robot_generate_test(test_name: str = "test_session", path: str | None = None) -> str:
        """The session so far as a pytest test that reproduces it exactly (written to path if given)."""
        return await run(s.generate_test, test_name, path)

    @tool
    async def robot_crosscheck(backend: str) -> str:
        """Re-run the session's calls on another physics engine (e.g. drake, genesis, isaac, if installed)
        from the same scene and seed, and report whether the outcome holds and how far objects end up
        from where they did here. Use it before trusting a result that may depend on one contact model."""
        return await run(s.crosscheck, backend)

    @tool
    async def robot_save_trace(path: str | None = None) -> str:
        """Save the session's trace (for `robowright show-trace` / `replay`) and return its path."""
        return await run(s.save_trace, path)

    @tool
    async def robot_read_trace(path: str) -> str:
        """A trace (.zip, from a failed test or robot_save_trace) as text: what ran, which expectation
        or action failed and why, and the robot's joints, every object's pose and the contacts at
        the moment it failed. Read this before changing a test or the code under test."""
        from .trace import Trace

        return await run(lambda: Trace(path).summary())

    @tool
    async def robot_run_tests(
        target: str = "", robot: str | None = None, backend: str | None = None, select: str | None = None, cwd: str | None = None
    ) -> str:
        """Run robot tests with pytest and report the result: the pass/fail summary, each failure's
        message, and each failing test's trace as text (as robot_read_trace gives it).

        target: test file, directory or node id (default: the project's tests). robot: e.g. 'panda'
        or 'panda,ur5e' or a model file. backend: e.g. 'mujoco' or 'mujoco,pybullet'. select: a
        pytest -k expression. cwd: the project directory (default: the server's). Runs in its own
        process; the session's world is untouched."""
        import sys

        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rw-trace-text", "-rfE"]
        cmd += [target] if target else []
        cmd += ["--rw-robot", robot] if robot else []
        cmd += ["--rw-backend", backend] if backend else []
        cmd += ["-k", select] if select else []
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            proc.kill()
            return "the tests did not finish within 30 minutes"
        text = out.decode(errors="replace")
        if len(text) > 20000:  # keep the end: the summary, the failures and their traces
            text = "...\n" + text[-20000:]
        return f"exit code {proc.returncode} ({'passed' if proc.returncode == 0 else 'failed'})\n{text}"

    @tool
    async def robot_close() -> str:
        """Close the world and keep its trace."""
        return await run(s.close)

    return server


def main(argv=None) -> None:
    build_server().run()
