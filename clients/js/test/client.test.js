import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { ExpectationError, RobowrightError, TOOLS, launch } from "../src/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fake = () => launch({ command: [process.execPath, join(here, "fake-server.mjs")] });

test("every server tool is a method, in its namespace", async () => {
  const rw = await fake();
  try {
    assert.deepEqual(rw.server, { name: "fake", version: "9" });
    for (const t of TOOLS) assert.equal(typeof rw[t.ns][t.method], "function", t.tool);
    assert.ok(TOOLS.some((t) => t.tool === "robot_move_to" && t.method === "moveTo"));
    assert.deepEqual(JSON.parse(await rw.robot.moveTo({ target: [0.2, 0, 0.1] })), { name: "robot_move_to", args: { target: [0.2, 0, 0.1] } });
  } finally {
    await rw.close();
  }
});

test("camelCase argument names reach the server in snake_case, and undefined ones are left out", async () => {
  const rw = await fake();
  try {
    const out = JSON.parse(await rw.robot.launch({ robot: "panda", keepOpen: true, seed: undefined }));
    assert.deepEqual(out.args, { robot: "panda", keep_open: true });
  } finally {
    await rw.close();
  }
});

test("a tool error throws RobowrightError naming the tool", async () => {
  const rw = await fake();
  try {
    await assert.rejects(rw.robot.pick({ object: "nope" }), (e) => e instanceof RobowrightError && e.tool === "robot_pick" && /no object named/.test(e.message));
  } finally {
    await rw.close();
  }
});

test("expect passes, fails with the reason, negates with .not, and takes the matcher's first argument bare", async () => {
  const rw = await fake();
  try {
    assert.match(await rw.expect("cube").toBeInside("bin"), /^PASS/);
    await assert.rejects(rw.expect("cube").toBeInside("box"), ExpectationError);
    assert.match(await rw.expect("cube").not.toBeInside("box"), /^PASS/);
    const sent = JSON.parse((await rw.expect("cube", { timeout: 2 }).toBeInside({ container: "bin" }, { minFraction: 0.5 })).slice(6));
    assert.deepEqual(sent, { subject: "cube", matcher: "to_be_inside", args: { container: "bin", min_fraction: 0.5 }, negate: false, timeout: 2 });
    assert.equal(typeof rw.expect("cube").toBeAtRest, "function");
  } finally {
    await rw.close();
  }
});

test("a matcher with no first argument refuses a bare value", async () => {
  const rw = await fake();
  try {
    await assert.rejects(rw.expect("cube").toBeAtRest(3), TypeError);
  } finally {
    await rw.close();
  }
});

test("calls in flight fail, not hang, when the server dies", async () => {
  const rw = await fake();
  await assert.rejects(rw.call("robot_crash"), /exited \(code 3\)/);
  await assert.rejects(rw.robot.snapshot(), /exited/);
  await rw.close();
});

test("a command that does not exist fails launch", async () => {
  await assert.rejects(launch({ command: ["robowright-no-such-command"] }), /ENOENT/);
});

// The real server, when there is one: ROBOWRIGHT_COMMAND, or robowright on PATH.
function realCommand() {
  if (process.env.ROBOWRIGHT_COMMAND) return process.env.ROBOWRIGHT_COMMAND.split(" ").filter(Boolean);
  const venv = join(here, "..", "..", "..", ".venv", "bin", "robowright");
  if (existsSync(venv)) return [venv, "mcp"];
  try {
    execFileSync("robowright", ["--help"], { stdio: "ignore" });
    return ["robowright", "mcp"];
  } catch {
    return null;
  }
}

const real = realCommand();
test("against the real server: pick the cube, hold it, and fail to find it in the bin", { skip: !real && "robowright is not installed" }, async () => {
  const rw = await launch({ command: real, stderr: "ignore" });
  try {
    assert.equal(rw.server.name, "robowright");
    assert.match(await rw.robot.launch(), /cube: red box/);
    await rw.robot.pick({ object: "cube" });
    assert.match(await rw.expect("gripper").toBeHolding("cube"), /^PASS/);
    await assert.rejects(rw.expect("cube").toBeInside("bin", { timeout: 0.2 }), (e) => e instanceof ExpectationError && /bin spans/.test(e.message));
    await assert.rejects(rw.robot.pick({ object: "nope" }), (e) => e instanceof RobowrightError && /scene has/.test(e.message));
  } finally {
    await rw.close();
  }
});
