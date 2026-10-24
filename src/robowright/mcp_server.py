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
import inspect
from concurrent.futures import ThreadPoolExecutor

from . import __version__
from .session import Session

INSTRUCTIONS = """\
Test robot software against a simulated robot (MuJoCo by default; deterministic physics),
or a robot behind ROS 2 (robot_launch backend="ros2": Gazebo, Isaac Sim, a real arm).
Start with robot_launch, then robot_snapshot to see the world: objects are named (those
names are the refs other tools take). For a robot behind ROS 2, start with ros2_inspect: it
says what is running and gives the settings robot_launch needs.

The code under test is the user's: run their controller or policy with robot_run_policy
("package.module:name"), or serve the simulation as a ROS 2 robot with robot_serve_ros2 and
start their stack with robot_start_process; then let time pass (robot_wait, robot_step) and
check what happened with robot_expect. robot_set_targets sends raw joint/gripper targets.
robot_watch gives a URL where the user can watch the world live in a browser.

A robot in Gazebo (ros2={"gazebo": true}): robowright drives the simulator as well as watching
the robot, the way a browser is driven. sim_state, sim_pause, sim_play, sim_step (exact physics
iterations), sim_spawn and sim_remove act on Gazebo itself; robot_move_object moves a model there;
object positions are Gazebo's own; robot_screenshot is Gazebo's rendering, from cameras robowright
adds to the world. robot_options={"gripper": false} loads an arm's own description as it is.

An environment someone else built (gymnasium, dm_control, ManiSkill, panda-gym): sim_attach it, then
sim_subjects names what can be checked. sim_env_step steps their own step, which robowright records and
checks. robot_snapshot, robot_expect, robot_push, robot_move_object, robot_screenshot and robot_save_trace work
there; the calls that drive a robot robowright built (robot_pick, robot_run_policy, ...) say why they cannot.

robowright's own reference controller sets up situations and explores: robot_pick,
robot_place, robot_move_to, robot_gripper, robot_move_joints, robot_home (arms), robot_stand,
robot_crouch (legged). Disturb the world with robot_push, robot_move_object and robot_fault.
Look with robot_screenshot (cameras: front, top, side). A failed action reports why and
leaves the session running. robot_generate_test turns the session into a pytest test;
robot_crosscheck re-runs it on another engine. robot_run_tests runs a project's tests and
returns each failure with its trace as text; robot_read_trace reads any trace that way.
Positions are metres in the world frame, z up. Simulated robots are their model files as
published plus cited specs (fidelity "published"); a pass there is evidence about that model in
that engine, not about the real robot."""


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
    # The SDK reports this as the server's version; left blank, a client shows none. mcp 1.x's FastMCP has no such argument.
    version = {"version": __version__} if "version" in inspect.signature(Server).parameters else {}
    server = Server(name="robowright", instructions=INSTRUCTIONS, **version)
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
    async def robot_launch(
        robot: str | None = None,
        backend: str = "mujoco",
        seed: int = 0,
        objects: list[dict] | None = None,
        fidelity: str | None = None,
        ros2: dict | None = None,
        robot_options: dict | None = None,
    ) -> str:
        """Start a fresh world, closing any open one, and return its snapshot.

        robot: a name from robot_list_robots, or a model file (default: the SO-101). backend:
        mujoco (default, fastest), pybullet, drake, genesis or isaac, whichever are installed; or
        ros2 to connect to a robot behind ROS 2. There, call ros2_inspect first: it returns the
        ros2 settings to pass here. Leave robot out and the robot is read from the description
        its own stack publishes; leave the controllers out and they are found. ros2 settings,
        e.g. {"frame": "base_link", "use_sim_time": true, "gazebo": true (the robot is in Gazebo:
        drive Gazebo too), "command": false (observe only: the user's own code moves the robot),
        "objects": {"cube": "cube"} (objects seen as TF frames)}. robot_options: for a robot
        given as a model file, e.g. {"gripper": false, "base_pos": [0, 0, 0]} (an arm with no
        gripper, standing at the world origin). fidelity: "published" (default:
        the model as its makers published it) or "adjusted" (robowright's tuning too). seed:
        same seed and same calls give the same world. objects: replace the default scene's objects, each
        {"name", "kind": box|cylinder|sphere|bin, "size": half-extents in m (box/bin xyz,
        cylinder [radius, half_height], sphere [radius]), "pos": [x, y] to rest it on the
        floor, or [x, y, z] (z is a free object's centre, a bin's base),
        "color": red|green|blue|yellow|..., "mass": kg}. On ros2 there are no objects unless
        objects names some, and the robot is not moved to a start pose.
        """
        return await run(
            s.launch, robot=robot, backend=backend, seed=seed, objects=objects, fidelity=fidelity, ros2=ros2, robot_options=robot_options
        )

    @tool
    async def ros2_inspect(namespace: str = "", timeout: float = 5.0) -> str:
        """What is running behind ROS 2 before connecting to it: the robot's description, its
        joints, its controllers and how each is driven, whether it runs on simulation time,
        whether Gazebo is there, and the ros2 settings to give robot_launch(backend="ros2")."""
        return await run(s.ros2_inspect, namespace, timeout)

    @tool
    async def ros2_topic(name: str, timeout: float = 2.0) -> str:
        """The newest message on a ROS 2 topic, with its rate and how many have arrived. Works on a
        robot behind ROS 2 (robot_launch backend="ros2") or a world served with robot_serve_ros2."""
        return await run(s.ros_topic, name, timeout)

    @tool
    async def ros2_call(service: str, request: dict | None = None, timeout: float = 10.0) -> str:
        """Call a ROS 2 service and return its response. request: the request's fields as a dict
        (its type is found from the graph), or omitted for an empty request. Time passes while it waits."""
        return await run(s.ros_call, service, request, timeout)

    @tool
    async def ros2_send_goal(action: str, goal: dict | None = None, timeout: float = 60.0) -> str:
        """Send a goal to a ROS 2 action server and wait for its result: succeeded, aborted,
        canceled or rejected, and the result. goal: its fields as a dict. A goal on the arm's own
        controller takes the arm from robowright until robowright next moves it."""
        return await run(s.ros_send_goal, action, goal, timeout)

    @tool
    async def ros2_publish(topic: str, message: dict, type: str | None = None) -> str:
        """Publish one message on a ROS 2 topic: its fields as a dict; the type comes from the
        graph, or pass type="pkg/msg/Name" for a topic nobody has published yet."""
        return await run(s.ros_publish, topic, message, type)

    @tool
    async def robot_run_policy(
        policy: str, kwargs: dict | None = None, until: dict | None = None, timeout: float = 20.0, hold: float = 0.0
    ) -> str:
        """Run the user's controller or policy and report whether it succeeded.

        policy: "package.module:name" (a callable obs -> joint targets, or a class built with
        kwargs), or a learned policy reference (a checkpoint path, "plugin:ref"). until: the goal,
        {"subject": "cube", "matcher": "to_be_inside", "args": {"container": "bin"}}; without it,
        it runs for timeout seconds. hold: seconds the goal must stay true."""
        return await run(s.run_policy, policy, kwargs, until, timeout, hold)

    @tool
    async def robot_set_targets(joints: dict[str, float] | None = None, gripper: float | None = None) -> str:
        """Set joint targets (radians, by name) and/or the gripper opening (0 closed, 1 open) without
        waiting: the raw command a controller sends. Time does not pass: follow with robot_step."""
        return await run(s.set_targets, joints, gripper)

    @tool
    async def robot_step(steps: int = 1) -> str:
        """Advance the world this many control periods (50 per second) with the current targets."""
        return await run(s.step, steps)

    @tool
    async def sim_state() -> str:
        """The simulator's own state (Gazebo): its clock, iterations, paused or running, real-time
        factor, and every model in it. Also the world's own time, which snapshots report as t and
        which counts from launch; Gazebo's clock need not match it."""
        return await run(s.sim_state)

    @tool
    async def sim_pause() -> str:
        """Pause the simulator (Gazebo). Its clock stops: so do the ROS 2 stack's timers on sim time."""
        return await run(s.sim_pause)

    @tool
    async def sim_play() -> str:
        """Let the paused simulator (Gazebo) run again."""
        return await run(s.sim_play)

    @tool
    async def sim_step(iterations: int = 1) -> str:
        """Advance the simulator (Gazebo) exactly this many physics iterations, paused before and
        after: to look at a moment closely."""
        return await run(s.sim_step, iterations)

    @tool
    async def sim_spawn(object: dict) -> str:
        """Add an object to the simulator (Gazebo): {"name", "kind": box|cylinder|sphere|bin,
        "size", "pos", "color", "mass"}, as robot_launch's objects."""
        return await run(s.sim_spawn, object)

    @tool
    async def sim_remove(name: str) -> str:
        """Remove a model from the simulator (Gazebo)."""
        return await run(s.sim_remove, name)

    @tool
    async def sim_attach(env_id: str = "", kwargs: dict | None = None, seed: int | None = 0, entry_point: str | None = None) -> str:
        """Attach to a simulation someone else built, and watch and drive it. env_id: a registered gymnasium id
        (e.g. "FetchPickAndPlace-v4", "gym_aloha/AlohaTransferCube-v0"; kwargs as gymnasium.make takes them), or
        gymnasium's own "module:EnvId" form, which imports the module first. If the id is not registered yet,
        the installed environment packages (gymnasium-robotics, ManiSkill, panda-gym, gym-aloha, ...) are
        imported to register theirs, then it is retried. entry_point: "module:callable" that returns an
        environment, called with kwargs (Meta-World: entry_point="metaworld:make_mt_envs",
        kwargs={"name": "pick-place-v3", "seed": 0}). It is reset with seed (None: not seeded). Closes any open
        world. Time moves only when sim_env_step steps the environment: their own loop is not robowright's.
        Returns the snapshot, or why it could not attach: the ids that look like env_id, which packages were
        registered, and which imports failed."""
        return await run(s.sim_attach, env_id, kwargs, seed, entry_point)

    @tool
    async def sim_env_step(action: list[float] | None = None, steps: int = 1, sample: bool = False) -> str:
        """Step the attached environment's own step this many times. action: its action, with the shape of its
        action space; None is the zero (no-op) action, or with sample=true a random one from the action space.
        Returns the reward summed over the steps, whether the episode ended, which named bodies moved, and the
        snapshot. Every step is recorded (robot_save_trace keeps it)."""
        return await run(s.sim_env_step, action, steps, sample)

    @tool
    async def sim_subjects() -> str:
        """The names an attached simulation takes: bodies (free bodies are objects, starred), sites, joints and
        cameras. Bodies and sites go to robot_expect, robot_push, robot_move_object; a joint to robot_expect with
        matcher to_be_between {lo, hi}; a camera to robot_screenshot."""
        return await run(s.sim_subjects)

    @tool
    async def robot_serve_ros2(
        namespace: str = "",
        arm_controller: str = "arm_controller",
        gripper_controller: str = "gripper_controller",
    ) -> str:
        """Serve the simulated world as a ROS 2 robot (/joint_states, /clock, TF for the objects,
        FollowJointTrajectory and GripperCommand actions, forward position commands), so the user's
        ROS 2 stack drives it. Then robot_start_process their stack, and robot_wait / robot_expect."""
        return await run(s.serve_ros2, namespace, arm_controller, gripper_controller)

    @tool
    async def robot_start_process(command: str, cwd: str | None = None) -> str:
        """Start the code under test as a process (e.g. "ros2 launch my_pkg pick.launch.py
        use_sim_time:=true"). Its output is kept for robot_stop_process."""
        return await run(s.start_process, command, cwd)

    @tool
    async def robot_stop_process() -> str:
        """Stop the process robot_start_process started, and return the end of its output."""
        return await run(s.stop_process)

    @tool
    async def robot_watch(port: int = 8765) -> str:
        """A URL where the user can watch the world live in a browser while you drive it."""
        return await run(s.watch, port)

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
    async def robot_drive(linear: float, angular: float, duration: float) -> str:
        """Mobile: drive the base at linear m/s and angular rad/s (turning left is positive) for duration seconds, then stop."""
        return await run(s.drive, linear, angular, duration)

    @tool
    async def robot_go_to(x: float, y: float, yaw: float | None = None) -> str:
        """Mobile: send the base to map-frame (x, y) with heading yaw (radians, default 0). Its own navigation
        stack plans and drives; the reply says where it ended up, or why the stack could not get it there."""
        return await run(s.go_to, x, y, yaw)

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
        to_have_no_collisions {}, to_be_visible {camera, min_pixels}, to_be_hidden {camera}. Object names in args
        are resolved.
        """
        return await run(s.expect, subject, matcher, args, negate, timeout)

    @tool
    async def robot_generate_test(test_name: str = "test_session", path: str | None = None, keep_failed: bool = False) -> str:
        """The session so far as a pytest test that reproduces it exactly (written to path if given).

        keep_failed: false (default) leaves out the steps that failed (a wrong attempt is not something
        the test should assert), each as a comment; the first line of the result says how many were left
        out. A failed step can still have moved the world, so the test may then not reproduce the
        session. true keeps each failed step, asserting it fails again."""
        return await run(s.generate_test, test_name, path, keep_failed)

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
