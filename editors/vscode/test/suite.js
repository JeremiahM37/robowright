// The extension in a real VS Code: the tests are found, each sits on its line, a run reports each outcome, a failure
// links its trace, and the trace opens in an editor tab.
"use strict";

const assert = require("assert");
const path = require("path");
const vscode = require("vscode");

async function run() {
  const ext = vscode.extensions.getExtension("jeremiahm37.robowright");
  const api = await ext.activate();
  await api.discover();

  const files = [];
  api.controller.items.forEach(f => files.push(f));
  const tests = files.flatMap(f => { const out = []; f.children.forEach(t => out.push(t)); return out; });
  const ids = tests.map(t => t.id).sort();
  assert.deepStrictEqual(ids, [
    "tests/test_arm.py::test_lifts_the_cube[mujoco]",
    "tests/test_arm.py::test_reaches_the_shelf[mujoco]",
    "tests/test_plain.py::test_sum",
  ]);
  const shelf = tests.find(t => t.id.includes("shelf"));
  assert.strictEqual(shelf.range.start.line, 8, "the test sits on its def line (0-based)");

  const code = await api.run(undefined); // everything
  assert.strictEqual(code, 1, "pytest's status: one test fails");
  assert.strictEqual(api.results.get("tests/test_plain.py::test_sum"), "passed");
  assert.strictEqual(api.results.get("tests/test_arm.py::test_lifts_the_cube[mujoco]"), "passed");
  assert.strictEqual(api.results.get(shelf.id), "failed");
  assert.ok(api.traces.get(shelf.id).endsWith(".zip"), "the failure's trace is known");
  assert.ok(api.traces.get("tests/test_arm.py::test_lifts_the_cube[mujoco]"), "a passing test's trace too");

  // one test, chosen in the explorer
  api.results.clear();
  assert.strictEqual(await api.run([tests.find(t => t.id.endsWith("test_sum"))]), 0);
  assert.deepStrictEqual([...api.results.keys()], ["tests/test_plain.py::test_sum"]);

  // the Debug profile: the same test under the Python Debugger, stopping at a breakpoint in it
  const sum = tests.find(t => t.id.endsWith("test_sum"));
  api.results.clear();
  if (vscode.extensions.getExtension("ms-python.debugpy")) {
    const stops = [];
    const tracker = vscode.debug.registerDebugAdapterTrackerFactory("debugpy", {
      createDebugAdapterTracker: session => ({
        onDidSendMessage: async m => {
          if (m.type !== "event" || m.event !== "stopped") return;
          const st = await session.customRequest("stackTrace", { threadId: m.body.threadId, levels: 1 });
          stops.push(`${path.basename(st.stackFrames[0].source.path)}:${st.stackFrames[0].line}`);
          await session.customRequest("continue", { threadId: m.body.threadId });
        },
      }),
    });
    const bp = new vscode.SourceBreakpoint(new vscode.Location(sum.uri, new vscode.Position(1, 0)));
    vscode.debug.addBreakpoints([bp]);
    try {
      assert.strictEqual(await api.run([sum], false, true), 0, "the test passes under the debugger");
    } finally {
      vscode.debug.removeBreakpoints([bp]);
      tracker.dispose();
    }
    assert.deepStrictEqual(stops, ["test_plain.py:2"], "it stopped at the breakpoint, then ran on");
    assert.strictEqual(api.results.get(sum.id), "passed");
  } else {
    assert.strictEqual(await api.run([sum], false, true), -1, "no Python Debugger: the run says so");
    assert.ok(!api.results.has(sum.id));
  }

  const panel = await api.showTrace(api.traces.get(shelf.id));
  assert.ok(panel.webview.html.includes("test_reaches_the_shelf"), "the trace viewer, for this test");
  panel.dispose();
  console.log(`robowright extension: ${ids.length} tests found, results ${JSON.stringify([...api.results])}, debugger ${vscode.extensions.getExtension("ms-python.debugpy") ? "stopped at the breakpoint" : "absent, said so"}, trace viewer opened`);
  if (process.env.ROBOWRIGHT_TEST_MUST_FAIL) assert.fail("ROBOWRIGHT_TEST_MUST_FAIL is set");
}

module.exports = { run };
