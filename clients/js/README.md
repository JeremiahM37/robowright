# robowright for JavaScript and TypeScript

The same simulated robots, matchers and traces as the Python library, from Node. The client starts the
robowright MCP server (`robowright mcp`) and talks to it over stdio, so every tool an AI agent gets is a typed
method here, and a world you drive from JavaScript is the same deterministic world a pytest test sees. It has no
dependencies.

It needs robowright itself, with its MCP extra, where the server will run:

```sh
pip install "robowright[mcp] @ git+https://github.com/JeremiahM37/robowright"
```

```js
import { test } from "node:test";
import { launch } from "robowright";

test("the arm puts the cube in the bin", async () => {
  const rw = await launch();
  try {
    await rw.robot.launch({ robot: "so101" });
    await rw.robot.pick({ object: "cube" });
    await rw.expect("gripper").toBeHolding("cube");
    await rw.robot.place({ on: "bin" });
    await rw.expect("cube").toBeInside("bin", { timeout: 2 });
    await rw.expect("gripper").not.toBeHolding("cube");
  } finally {
    await rw.robot.saveTrace({ path: "trace.zip" });
    await rw.close();
  }
});
```

- **Tools** are grouped as `rw.robot.*`, `rw.ros2.*` and `rw.sim.*`, named after the MCP tools in camelCase
  (`robot_move_to` is `rw.robot.moveTo`). Arguments are one object; camelCase names are sent in snake_case. Each
  returns the tool's text: a snapshot of the world, a result or a report.
- **`expect(subject)`** has the Python matchers in camelCase (`toBeInside`, `toBeHolding`, `toBeNear`,
  `toHaveJoint`, `toBeVisible`, ...), `.not`, and a `timeout` in seconds of simulated time, retried as in
  Python. A matcher's first argument can be given bare (`toBeInside("bin")`) or as an object
  (`toBeNear({ target: "bin", tol: 0.02 })`). A failing one throws `ExpectationError` with the reason.
- **Errors** from a tool throw `RobowrightError`, with `.tool` naming it.
- **`launch({ command })`** picks the server command; default `robowright mcp`, or `ROBOWRIGHT_COMMAND`.
- `rw.robot.generateTest()` writes what you did as a pytest test that reproduces it, and `rw.robot.saveTrace()`
  keeps the trace for `robowright show-trace`.

`src/tools.js` and `index.d.ts` are generated from the server's own tool list by
`python scripts/gen_js_client.py`; the Python test suite fails when they fall behind. `npm test` runs the client's
tests, against a stand-in server and, when robowright is installed, the real one.
