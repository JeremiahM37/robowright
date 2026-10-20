// robowright for VS Code: the project's robot tests in the Test Explorer, run with pytest and robowright's own
// events plugin (robowright.ui_events, the one `robowright test --ui` uses), each failure on its line with a link to
// its trace, which opens in an editor tab.
"use strict";

const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const os = require("os");
const net = require("net");
const path = require("path");

let controller, output, storage;
const traces = new Map(); // test id -> the trace its last run wrote
const results = new Map(); // test id -> its last outcome

async function python() {
  const set = vscode.workspace.getConfiguration("robowright").get("python");
  if (set) return set;
  try {
    const ext = vscode.extensions.getExtension("ms-python.python");
    if (ext) {
      const api = ext.isActive ? ext.exports : await ext.activate();
      const env = api.environments && api.environments.getActiveEnvironmentPath();
      if (env && env.path) return env.path;
    }
  } catch (e) {
    output.appendLine(`robowright: could not ask the Python extension for its interpreter (${e.message})`);
  }
  return "python3";
}

function root() {
  const f = vscode.workspace.workspaceFolders;
  return f && f.length ? f[0].uri.fsPath : undefined;
}

function args() {
  return vscode.workspace.getConfiguration("robowright").get("pytestArgs") || [];
}

function events(file) {
  try {
    return fs.readFileSync(file, "utf8").split("\n").filter(Boolean).map(l => {
      try { return JSON.parse(l); } catch { return null; } // a line still being written
    }).filter(Boolean);
  } catch {
    return [];
  }
}

function eventsFile(name) {
  const f = path.join(storage, `${name}-${process.pid}-${Date.now()}.jsonl`);
  try { fs.unlinkSync(f); } catch {}
  return f;
}

function pytest(py, argv, eventsPath, cwd) {
  const env = { ...process.env, ROBOWRIGHT_UI_EVENTS: eventsPath, PYTHONUNBUFFERED: "1", PYTHONDONTWRITEBYTECODE: "1" };
  return cp.spawn(py, ["-m", "pytest", ...argv, "-p", "robowright.ui_events"], { cwd, env });
}

// --- finding the tests ---------------------------------------------------------------------------------------------
let rootdir;

async function discover() {
  const cwd = root();
  if (!cwd) return;
  const py = await python();
  const file = eventsFile("collect");
  const argv = ["--collect-only", "-q", ...withoutXdist(args())];
  const text = await new Promise(resolve => {
    const p = pytest(py, argv, file, cwd);
    let out = "";
    p.stdout.on("data", d => (out += d));
    p.stderr.on("data", d => (out += d));
    p.on("error", e => resolve(`could not start ${py}: ${e.message}`));
    p.on("close", code => resolve(code === 0 || code === 5 ? "" : out));
  });
  const got = events(file).find(e => e.event === "collected");
  try { fs.unlinkSync(file); } catch {}
  if (!got) {
    output.appendLine("robowright: the tests could not be listed:\n" + text);
    return;
  }
  rootdir = got.rootdir;
  const files = new Map();
  for (const id of got.tests) {
    const [rel, line] = (got.locations && got.locations[id]) || [id.split("::")[0], 0];
    const fid = id.split("::")[0];
    if (!files.has(fid)) {
      const uri = vscode.Uri.file(path.join(rootdir, fid));
      files.set(fid, controller.createTestItem(fid, fid, uri));
    }
    const name = id.split("::").slice(1).join("::") || id;
    const item = controller.createTestItem(id, name, vscode.Uri.file(path.join(rootdir, rel)));
    item.range = new vscode.Range(line, 0, line, 0);
    files.get(fid).children.add(item);
  }
  controller.items.replace([...files.values()]);
}

// --- running them ------------------------------------------------------------------------------------------------
function leaves(items) {
  const out = [];
  const walk = it => (it.children.size ? it.children.forEach(walk) : out.push(it));
  items.forEach(walk);
  return out;
}

function failureMessage(item, result) {
  const md = new vscode.MarkdownString();
  md.isTrusted = { enabledCommands: ["robowright.showTrace"] };
  md.appendCodeblock(result.message || "failed", "text");
  if (result.traces && result.traces.length) {
    const a = encodeURIComponent(JSON.stringify([result.traces[0]]));
    md.appendMarkdown(`\n[Show trace](command:robowright.showTrace?${a})`);
  }
  const m = new vscode.TestMessage(md);
  if (item.uri && item.range) {
    const line = lineOf(result.message, item.uri.fsPath);
    m.location = new vscode.Location(item.uri, line !== undefined ? new vscode.Position(line, 0) : item.range);
  }
  return m;
}

// pytest's failure text ends with "path/to/test_x.py:13: Error": that line, in this test's file
function lineOf(message, file) {
  const base = path.basename(file).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const found = [...String(message || "").matchAll(new RegExp(`${base}:(\\d+):`, "g"))];
  return found.length ? Number(found[found.length - 1][1]) - 1 : undefined;
}

async function run(request, token, live, debug = false) {
  const cwd = root();
  const run = controller.createTestRun(request);
  const chosen = request.include ? leaves(request.include) : leaves(controller.items);
  const skipped = new Set((request.exclude || []).flatMap(x => leaves([x]).map(i => i.id)));
  const items = chosen.filter(i => !skipped.has(i.id));
  const byId = new Map(items.map(i => [i.id, i]));
  items.forEach(i => run.enqueued(i));
  const ids = request.include ? items.map(i => i.id) : [];
  const traceDir = path.join(storage, "traces");
  let argv = [...args().filter(a => !(request.include && !a.startsWith("-") && fs.existsSync(path.join(cwd, a.split("::")[0])))), ...ids];
  // under the debugger: one process (breakpoints are in this one) and no time limit while stopped at one
  if (debug) argv = [...withoutXdist(argv), "--rw-timeout", "0"];
  argv.push("--rw-trace", "on", "--rw-trace-dir", traceDir);
  if (rootdir && !args().some(a => a.startsWith("--rootdir"))) argv.push("--rootdir", rootdir);
  const livePort = live ? await freePort() : 0;
  if (live) argv.push("--rw-live", String(livePort));
  const file = eventsFile("run");
  const py = await python();
  // the live view starts with the first test that builds a world: open it once it answers
  let opened = false;
  const watchLive = live && setInterval(() => {
    if (opened) return;
    const s = net.connect(livePort, "127.0.0.1", () => {
      s.end();
      if (!opened) { opened = true; vscode.env.openExternal(vscode.Uri.parse(`http://127.0.0.1:${livePort}/`)); }
    });
    s.on("error", () => {});
  }, 500);
  let seen = 0;
  const outcomes = new Map(); // this run's
  const take = () => {
    const all = events(file);
    for (const e of all.slice(seen)) {
      const item = byId.get(e.nodeid);
      if (!item) continue;
      if (e.event === "start") run.started(item);
      if (e.event !== "result") continue;
      if (e.traces && e.traces.length) traces.set(item.id, e.traces[0]);
      results.set(item.id, e.outcome);
      outcomes.set(item.id, e.outcome);
      const ms = (e.duration || 0) * 1000;
      if (e.outcome === "passed" || e.outcome === "xfailed") run.passed(item, ms);
      else if (e.outcome === "skipped") run.skipped(item);
      else run.failed(item, failureMessage(item, e), ms); // failed, flaky, xpassed
    }
    seen = all.length;
  };
  const timer = setInterval(take, 250);
  const code = debug ? await underDebugger(py, argv, file, cwd, run, token) : await new Promise(resolve => {
    const p = pytest(py, argv, file, cwd);
    token.onCancellationRequested(() => p.kill("SIGTERM"));
    p.stdout.on("data", d => run.appendOutput(String(d).replace(/\r?\n/g, "\r\n")));
    p.stderr.on("data", d => run.appendOutput(String(d).replace(/\r?\n/g, "\r\n")));
    p.on("error", e => { run.appendOutput(`could not start ${py}: ${e.message}\r\n`); resolve(-1); });
    p.on("close", resolve);
  });
  clearInterval(timer);
  if (watchLive) clearInterval(watchLive);
  take();
  try { fs.unlinkSync(file); } catch {}
  // a test the run never reached (the debugger could not start, or was stopped first) is not left spinning
  if (code === -1) items.filter(i => !outcomes.has(i.id)).forEach(i => run.errored(i, new vscode.TestMessage("not run: see the output")));
  run.end();
  if (code !== -1 && debug) return [...outcomes.values()].some(o => !["passed", "xfailed", "skipped"].includes(o)) ? 1 : 0;
  return code;
}

function withoutXdist(argv) {
  const out = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "-n" || argv[i] === "--numprocesses" || argv[i] === "--dist" || argv[i] === "--maxprocesses") { i++; continue; }
    if (/^-n\S/.test(argv[i]) || /^--(numprocesses|dist|maxprocesses)=/.test(argv[i])) continue;
    out.push(argv[i]);
  }
  return out;
}

// the launch configuration the Python Debugger (ms-python.debugpy) runs: the same pytest, events and arguments as a run
function debugConfig(py, argv, eventsPath, cwd) {
  return {
    type: "debugpy",
    request: "launch",
    name: "robowright: debug tests",
    module: "pytest",
    args: [...argv, "-p", "robowright.ui_events"],
    python: py,
    cwd,
    env: { ROBOWRIGHT_UI_EVENTS: eventsPath, PYTHONUNBUFFERED: "1", PYTHONDONTWRITEBYTECODE: "1" },
    console: "internalConsole",
    justMyCode: false,
    purpose: ["debug-test"],
  };
}

// runs pytest under the Python Debugger: breakpoints in tests (and in robowright) stop there. Resolves when the
// session ends (-1: it could not start)
async function underDebugger(py, argv, file, cwd, run, token) {
  if (!vscode.extensions.getExtension("ms-python.debugpy")) {
    const msg = "robowright: debugging tests needs the Python Debugger extension (ms-python.debugpy)";
    run.appendOutput(msg + "\r\n");
    vscode.window.showErrorMessage(msg, "Install").then(c => { if (c) vscode.commands.executeCommand("workbench.extensions.installExtension", "ms-python.debugpy"); });
    return -1;
  }
  const config = debugConfig(py, argv, file, cwd);
  const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  let session;
  const started = vscode.debug.onDidStartDebugSession(s => { if (!session && s.configuration.name === config.name) session = s; });
  const ended = new Promise(resolve => {
    const sub = vscode.debug.onDidTerminateDebugSession(s => { if (session && s.id === session.id) { sub.dispose(); resolve(0); } });
  });
  token.onCancellationRequested(() => session && vscode.debug.stopDebugging(session));
  const ok = await vscode.debug.startDebugging(folder, config, { testRun: run });
  started.dispose();
  if (!ok || !session) { run.appendOutput("robowright: the debugger did not start\r\n"); return -1; }
  return ended;
}

function freePort() {
  return new Promise(resolve => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => { const { port } = srv.address(); srv.close(() => resolve(port)); });
  });
}

// --- traces ------------------------------------------------------------------------------------------------------
async function showTrace(target) {
  let zip = target instanceof vscode.Uri ? target.fsPath : target;
  if (!zip) {
    const picked = await vscode.window.showOpenDialog({ filters: { "robowright trace": ["zip"] }, defaultUri: vscode.Uri.file(path.join(root() || os.homedir(), "robowright-traces")) });
    if (!picked || !picked.length) return;
    zip = picked[0].fsPath;
  }
  const out = path.join(storage, `trace-${Date.now()}.html`);
  const py = await python();
  const res = cp.spawnSync(py, ["-m", "robowright.cli", "show-trace", zip, "-o", out, "--no-open"], { cwd: root(), encoding: "utf8" });
  if (res.status !== 0 || !fs.existsSync(out)) {
    vscode.window.showErrorMessage(`robowright: could not open the trace: ${(res.stderr || res.stdout || res.error || "").toString().slice(-500)}`);
    return;
  }
  const panel = vscode.window.createWebviewPanel("robowrightTrace", path.basename(zip), vscode.ViewColumn.Active, { enableScripts: true, retainContextWhenHidden: true });
  panel.webview.html = fs.readFileSync(out, "utf8");
  try { fs.unlinkSync(out); } catch {}
  return panel;
}

// --- recording a new test (Playwright's "Record new") ----------------------------------------------------------------
const recorders = new Set();

function robotArg() {
  const a = args();
  const i = a.findIndex(x => x === "--rw-robot" || x.startsWith("--rw-robot="));
  if (i < 0) return undefined;
  const v = a[i].includes("=") ? a[i].split("=")[1] : a[i + 1];
  return v && v.split(",")[0];
}

function newTestFile(cwd) {
  const dir = ["tests", "test"].find(d => fs.existsSync(path.join(cwd, d))) || "";
  for (let n = 1; ; n++) {
    const f = path.join(cwd, dir, n === 1 ? "test_recorded.py" : `test_recorded_${n}.py`);
    if (!fs.existsSync(f)) return f;
  }
}

// `robowright codegen --robot R`: a page where what you do runs on the robot and is written as a test; Save writes
// it, and the test opens here. Resolves once the page is up, with its address and the file it saves to.
async function recordNew(opts = {}) {
  const cwd = root();
  if (!cwd) return;
  const robot = opts.robot || (await vscode.window.showInputBox({ prompt: "Record a test on which robot? (a name or a model file)", value: robotArg() || "so101" }));
  if (!robot) return;
  const file = opts.file || newTestFile(cwd);
  const py = await python();
  const p = cp.spawn(py, ["-m", "robowright.cli", "codegen", "--robot", robot, "--port", "0", "-o", file], { cwd, env: { ...process.env, PYTHONUNBUFFERED: "1" } });
  recorders.add(p);
  p.on("close", () => recorders.delete(p));
  let shown = false;
  const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(path.dirname(file), path.basename(file)));
  const opened = async uri => {
    const doc = await vscode.workspace.openTextDocument(uri);
    await vscode.window.showTextDocument(doc, { preview: false });
    if (!shown) { shown = true; discover(); }
  };
  watcher.onDidCreate(opened); watcher.onDidChange(opened);
  p.on("close", () => watcher.dispose());
  const url = await new Promise(resolve => {
    let out = "";
    const take = d => { out += d; output.append(String(d)); const m = out.match(/https?:\/\/127\.0\.0\.1:\d+\//); if (m) resolve(m[0]); };
    p.stdout.on("data", take);
    p.stderr.on("data", take);
    p.on("close", () => resolve(undefined));
    p.on("error", e => { output.appendLine(`robowright: could not start ${py}: ${e.message}`); resolve(undefined); });
  });
  if (!url) {
    vscode.window.showErrorMessage("robowright: the recorder did not start; see the robowright output");
    output.show(true);
    return;
  }
  if (!opts.noOpen) vscode.env.openExternal(vscode.Uri.parse(url));
  vscode.window.showInformationMessage(`robowright: recording on ${robot}. Save in the page writes ${path.relative(cwd, file)}.`);
  return { url, file, stop: () => p.kill("SIGTERM") };
}

async function openUI() {
  const py = await python();
  const term = vscode.window.createTerminal({ name: "robowright UI", cwd: root() });
  term.sendText([py, "-m", "robowright.cli", "test", "--ui", ...args()].map(a => (/[\s"'$]/.test(a) ? JSON.stringify(a) : a)).join(" "));
  term.show();
  return term;
}

function activate(context) {
  storage = context.storageUri ? context.storageUri.fsPath : path.join(os.tmpdir(), "robowright-vscode");
  fs.mkdirSync(storage, { recursive: true });
  output = vscode.window.createOutputChannel("robowright");
  controller = vscode.tests.createTestController("robowright", "robowright");
  controller.resolveHandler = async item => { if (!item) await discover(); };
  controller.refreshHandler = () => discover();
  controller.createRunProfile("Run", vscode.TestRunProfileKind.Run, (r, t) => run(r, t, false), true);
  controller.createRunProfile("Run with live view", vscode.TestRunProfileKind.Run, (r, t) => run(r, t, true), false);
  controller.createRunProfile("Debug", vscode.TestRunProfileKind.Debug, (r, t) => run(r, t, false, true), true);
  const watcher = vscode.workspace.createFileSystemWatcher("**/{test_*.py,*_test.py,conftest.py,pytest.ini,pyproject.toml,robowright.toml}");
  watcher.onDidCreate(discover); watcher.onDidDelete(discover); watcher.onDidChange(discover);
  context.subscriptions.push(
    controller, output, watcher,
    vscode.commands.registerCommand("robowright.showTrace", showTrace),
    vscode.commands.registerCommand("robowright.openUI", openUI),
    vscode.commands.registerCommand("robowright.refresh", discover),
    vscode.commands.registerCommand("robowright.recordNew", () => recordNew()),
    { dispose: () => recorders.forEach(p => p.kill("SIGTERM")) },
  );
  // what the extension's own tests drive it through
  return { controller, discover, run: (include, live = false, debug = false) => run(new vscode.TestRunRequest(include), new vscode.CancellationTokenSource().token, live, debug), traces, results, showTrace, recordNew };
}

function deactivate() {}

module.exports = { activate, deactivate, lineOf, debugConfig, withoutXdist };
