import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

function run(env, file = "sample.test.mjs", cwd = mkdtempSync(join(tmpdir(), "rw-runner-"))) {
  const r = spawnSync(process.execPath, ["--test", "--test-reporter=tap", join(here, "fixtures", file)], {
    cwd,
    encoding: "utf8",
    // NODE_TEST_CONTEXT would make the inner run think it is already inside one, and run nothing.
    env: { ...Object.fromEntries(Object.entries(process.env).filter(([k]) => k !== "NODE_TEST_CONTEXT")), ROBOWRIGHT_COMMAND: `${process.execPath} ${join(here, "fake-server.mjs")}`, ...env },
  });
  return { ...r, cwd, out: r.stdout + r.stderr };
}

test("robowright/test runs every test on each robot, saves a failure's trace, and retries a flaky one", () => {
  const r = run({ ROBOWRIGHT_ROBOT: "panda,ur5e" });
  assert.notEqual(r.status, 0, r.out); // one test fails, on purpose
  assert.match(r.out, /^ok \d+ - passes \[mujoco-panda\]$/m);
  assert.match(r.out, /^ok \d+ - passes \[mujoco-ur5e\]$/m);
  assert.match(r.out, /^not ok \d+ - fails with its trace \[mujoco-panda\]$/m);
  assert.match(r.out, /trace: robowright-traces\/fails_with_its_trace_mujoco-panda\.zip/);
  assert.match(r.out, /robowright show-trace robowright-traces\/fails_with_its_trace_mujoco-panda\.zip/);
  // the test's own robot wins over the default, but ROBOWRIGHT_ROBOT wins over both, as a flag does
  assert.match(r.out, /^ok \d+ - flaky \[mujoco-panda\]$/m);
  assert.match(r.out, /# flaky: failed 1 time, then passed on retry 1/);
  assert.match(r.out, /# pass 4\n# fail 2/);
});

test("one robot and engine: test names are as written", () => {
  const r = run({ ROBOWRIGHT_ROBOT: "", ROBOWRIGHT_BACKEND: "" });
  assert.match(r.out, /^ok \d+ - passes$/m);
  assert.match(r.out, /^ok \d+ - flaky$/m);
});

// The real server, when there is one (as in client.test.js).
const venv = join(here, "..", "..", "..", ".venv", "bin", "robowright");
const real = process.env.ROBOWRIGHT_COMMAND || (existsSync(venv) ? `${venv} mcp` : null);

test("against the real server: a pick and place passes, and a failure leaves a trace that opens", { skip: !real && "robowright is not installed" }, () => {
  const r = run({ ROBOWRIGHT_COMMAND: real, ROBOWRIGHT_ROBOT: "", ROBOWRIGHT_BACKEND: "" }, "real.test.mjs");
  assert.match(r.out, /^ok \d+ - puts the cube in the bin$/m, r.out);
  assert.match(r.out, /^not ok \d+ - leaves the cube where it was$/m);
  assert.match(r.out, /bin spans/); // the matcher's own reason
  const trace = join(r.cwd, "robowright-traces", "leaves_the_cube_where_it_was.zip");
  assert.ok(existsSync(trace), r.out);
  const exe = real.split(" ")[0];
  const text = spawnSync(exe, ["show-trace", "--text", trace], { encoding: "utf8" });
  assert.equal(text.status, 0, text.stderr);
  assert.match(text.stdout, /to_be_inside/);
});
