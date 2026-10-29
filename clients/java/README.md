# robowright for Java

The same simulated robots, matchers and traces as the Python library, from Java 17 or later. The client starts the
robowright MCP server (`robowright mcp`) and talks to it over stdio, so every tool an AI agent gets is a method
here, and a world you drive from Java is the same deterministic world a pytest test sees. It has no dependencies.

It needs robowright itself, with its MCP extra, where the server will run:

```sh
pip install "robowright[mcp] @ git+https://github.com/JeremiahM37/robowright"
```

```java
import robowright.Robowright;
import robowright.RobotTools;

try (Robowright rw = Robowright.launch()) {
  rw.robot().launch(new RobotTools.LaunchOptions().setRobot("so101"));
  rw.robot().pick("cube");
  rw.expect("gripper").toBeHolding("cube");
  rw.robot().place("bin");
  rw.expect("cube", 2).toBeInside("bin");
  rw.expect("gripper").not().toBeHolding("cube");
  rw.robot().saveTrace(new RobotTools.SaveTraceOptions().setPath("trace.zip"));
}
```

- **Tools** are grouped as `rw.robot()`, `rw.ros2()` and `rw.sim()`, named after the MCP tools in camelCase
  (`robot_move_to` is `moveTo`). As in Playwright for Java, a tool's required arguments are parameters and the rest
  go in its options class (`new RobotTools.LaunchOptions().setRobot("panda").setKeepOpen(true)`). Two names
  differ, because Java will not take them: `robot_switch` is `switchTo` and `robot_wait` is `waitSeconds`. Each
  returns the tool's text: a snapshot of the world, a result or a report.
- **`expect(subject)`** has the Python matchers in camelCase (`toBeInside`, `toBeHolding`, `toBeNear`,
  `toHaveJoint`, `toBeVisible`, ...), `not()`, and `expect(subject, timeout)` in seconds of simulated time, retried
  as in Python. A matcher's first argument can be given bare (`toBeInside("bin")`) or as a map
  (`toBeNear(Map.of("target", "bin", "tol", 0.02))`), and a second map takes the rest. A failing one throws
  `ExpectationException`, an `AssertionError`, so JUnit and TestNG report it as a failed test.
- **Errors** from a tool throw `RobowrightException`, with `tool()` naming it.
- **`call(tool, args)`** calls any tool by its MCP name, with the server's own argument names.
- **`Robowright.launch(new Robowright.LaunchOptions().setCommand(...))`** picks the server command; default
  `robowright mcp`, or `ROBOWRIGHT_COMMAND`.

It is not on Maven Central yet. Build a jar from a checkout:

```sh
javac -d out clients/java/src/robowright/*.java && jar cf robowright.jar -C out .
```

`RobotTools`, `Ros2Tools`, `SimTools` and `Assertions` are generated from the server's own tool list by
`python scripts/gen_java_client.py`; the Python test suite fails when they fall behind, and runs the client's own
tests (`test/robowright/ClientTest.java`, no framework) against a stand-in server and the real one.
