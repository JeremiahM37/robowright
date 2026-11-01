import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, readdirSync } from "node:fs";
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
  // trials: seeds 7, 8, 9 of 7..10 settle it (two passes of four meet 50%), so the fourth never runs
  assert.match(r.out, /^ok \d+ - trials \[mujoco-panda\]$/m);
  assert.match(r.out, /# 2\/3 passed \(67%, 95% CI 21%-94%\); required rate >= 50% of 4, settled after 3/);
  assert.match(r.out, /# pass 6\n# fail 2/);
});

test("ROBOWRIGHT_RESULTS writes each file's results in the report's format", () => {
  const r = run({ ROBOWRIGHT_ROBOT: "panda", ROBOWRIGHT_RESULTS: "results" });
  const files = readdirSync(join(r.cwd, "results"));
  assert.equal(files.length, 1, r.out);
  const run_ = JSON.parse(readFileSync(join(r.cwd, "results", files[0]), "utf8"));
  assert.equal(run_.format, 1);
  const byName = Object.fromEntries(Object.entries(run_.tests).map(([k, v]) => [k.split("::").pop(), v]));
  assert.equal(byName["passes"].outcome, "passed");
  assert.equal(byName["fails with its trace"].outcome, "failed");
  assert.match(byName["fails with its trace"].message, /FAIL/);
  assert.ok(byName["fails with its trace"].traces[0].endsWith("fails_with_its_trace.zip"));
  assert.equal(byName["flaky"].outcome, "flaky");
  assert.match(byName["trials"].trials, /^2\/3 passed/);
});

test("one robot and engine: test names are as written", () => {
  const r = run({ ROBOWRIGHT_ROBOT: "", ROBOWRIGHT_BACKEND: "" });
  assert.match(r.out, /^ok \d+ - passes$/m);
  assert.match(r.out, /^ok \d+ - flaky$/m);
});

test("ROBOWRIGHT_PROJECT runs in robowright.toml's projects, each test in each", () => {
  const cli = `${process.execPath} ${join(here, "fake-projects.mjs")}`;
  const r = run({ ROBOWRIGHT_PROJECT: "nightly,smoke", ROBOWRIGHT_CLI: cli, ROBOWRIGHT_ROBOT: "", ROBOWRIGHT_BACKEND: "" }, "projects.test.mjs");
  assert.equal(r.status, 0, r.out);
  assert.match(r.out, /^ok \d+ - launches \[nightly-mujoco-panda\]$/m);
  assert.match(r.out, /^ok \d+ - launches \[nightly-mujoco-ur5e\]$/m);
  assert.match(r.out, /^ok \d+ - launches \[smoke\]$/m);
  const bad = run({ ROBOWRIGHT_PROJECT: "weekly", ROBOWRIGHT_CLI: cli }, "projects.test.mjs");
  assert.notEqual(bad.status, 0);
  assert.match(bad.out, /ROBOWRIGHT_PROJECT weekly: no \[projects.weekly\] in robowright.toml; it has nightly, smoke/);
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
  // and robowright's HTML report, from the results file
  const again = run({ ROBOWRIGHT_COMMAND: real, ROBOWRIGHT_ROBOT: "", ROBOWRIGHT_BACKEND: "", ROBOWRIGHT_RESULTS: "results" }, "real.test.mjs");
  const results = readdirSync(join(again.cwd, "results")).map((f) => join(again.cwd, "results", f));
  const report = join(again.cwd, "report.html");
  const merged = spawnSync(exe, ["merge-reports", ...results, "-o", report], { encoding: "utf8" });
  assert.equal(merged.status, 0, merged.stderr);
  const html = readFileSync(report, "utf8");
  assert.match(html, /puts the cube in the bin/);
  assert.match(html, /leaves the cube where it was/);
  assert.match(html, /open the trace/);
});
