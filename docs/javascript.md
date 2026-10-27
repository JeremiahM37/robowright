# JavaScript and TypeScript

Playwright has one API in four languages. robowright's test runner is pytest, but its robots can be driven from
Node as well: `clients/js` is a client, with TypeScript types, for the robowright MCP server. Every tool an AI agent
gets is a method there, and `expect` has the same matchers.

```js
import { launch } from "robowright";

const rw = await launch(); // starts `robowright mcp`
await rw.robot.launch({ robot: "so101" });
await rw.robot.pick({ object: "cube" });
await rw.expect("gripper").toBeHolding("cube");
await rw.robot.place({ on: "bin" });
await rw.expect("cube").toBeInside("bin", { timeout: 2 });
await rw.close();
```

It is not on npm yet; install it from the repository (`npm install ./clients/js` from a checkout). Its README in
[clients/js](https://github.com/JeremiahM37/robowright/tree/main/clients/js) has the details.

What it does not have yet: the pytest plugin's runner features (retries, projects, shards, the HTML report) are
Python's. From Node you run the world and assert on it with Node's own test runner, and `rw.robot.generateTest()`
writes the session as a pytest test when you want it in the suite.
