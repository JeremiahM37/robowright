// robowright/test: robot tests in Node, written as Playwright Test writes web tests, on Node's own test runner.
//
//   import { test, expect } from "robowright/test";
//   test.use({ robot: "panda" });
//   test("puts the cube in the bin", async ({ rw }) => {
//     await rw.robot.pick({ object: "cube" });
//     await rw.robot.place({ on: "bin" });
//     await expect("cube").toBeInside("bin");
//   });
//
// Each test gets a fresh world from one robowright server per test file. ROBOWRIGHT_ROBOT and ROBOWRIGHT_BACKEND
// (comma lists, as --rw-robot and --rw-backend) run every test on each, as Playwright runs every project. A failed
// test saves its trace and says how to open it; ROBOWRIGHT_RETRIES (or `retries`) runs it again, and a test that
// passes on a retry is reported flaky. `trials: 20, minSuccess: 0.9` judges a test as a rate over seeds, each with
// the scene randomized, as @pytest.mark.trials does. ROBOWRIGHT_RESULTS=dir writes each file's results there, which
// `robowright merge-reports dir/*.json` makes into the HTML report. ROBOWRIGHT_PROJECT=nightly (or a,b or all) runs
// in robowright.toml's [projects.nightly], as --rw-project does. Run with `node --test`.

import { AsyncLocalStorage } from "node:async_hooks";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { basename, join, relative, resolve } from "node:path";
import { after, describe, test as nodeTest } from "node:test";

import { launch } from "./index.js";

const current = new AsyncLocalStorage();
const fileOptions = {};
let server = null;

const list = (v) =>
  (Array.isArray(v) ? v : String(v ?? "").split(","))
    .map((s) => String(s).trim())
    .filter(Boolean);

function server_() {
  server ??= launch({ stderr: process.env.ROBOWRIGHT_SERVER_LOG ? "inherit" : "ignore" });
  return server;
}

const results = {};
const started = Date.now() / 1000;

after(async () => {
  if (server) await (await server).close();
  const dir = process.env.ROBOWRIGHT_RESULTS;
  if (dir && Object.keys(results).length) {
    mkdirSync(dir, { recursive: true });
    const run = { format: 1, started, seconds: Date.now() / 1000 - started, tests: results };
    writeFileSync(join(dir, `${slug(basename(process.argv[1] ?? "tests"))}-${process.pid}.json`), JSON.stringify(run, null, 1));
  }
});

// The share of trials that must pass, as robowright.stats does: the rate, or its 95% Wilson lower bound.
function wilson(k, n, z = 1.96) {
  if (!n) return [0, 1];
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

function trialSummary(passed, runs, n, min, lowerBound) {
  const [lo, hi] = wilson(passed, runs);
  const pct = (x) => `${Math.round(x * 100)}%`;
  const req = `${lowerBound ? "95% lower bound" : "rate"} >= ${pct(min)}`;
  let out = `${passed}/${runs} passed (${pct(runs ? passed / runs : 0)}, 95% CI ${pct(lo)}-${pct(hi)}); required ${req}`;
  if (runs < n) out += ` of ${n}, settled after ${runs}`;
  return out;
}

const slug = (s) => s.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");

// [projects.NAME] in robowright.toml, read through `robowright projects --json` (ROBOWRIGHT_CLI, or the server
// command without its `mcp`). A project's keys are those --rw-project applies that mean something here.
const PROJECT_KEYS = { robot: "robot", backend: "backend", retries: "retries", seed: "seed", fidelity: "fidelity" };
let projectTables = null;

function projects() {
  const raw = process.env.ROBOWRIGHT_PROJECT;
  if (!raw) return [null];
  if (!projectTables) {
    const server = (process.env.ROBOWRIGHT_COMMAND ?? "").split(" ").filter(Boolean);
    const cli = (process.env.ROBOWRIGHT_CLI ?? "").split(" ").filter(Boolean);
    const command = cli.length ? cli : server.at(-1) === "mcp" ? server.slice(0, -1) : ["robowright"];
    const found = JSON.parse(execFileSync(command[0], [...command.slice(1), "projects", "--json"], { encoding: "utf8" }));
    projectTables = { config: found.config, tables: found.projects ?? {} };
  }
  const { config, tables } = projectTables;
  const names = raw === "all" ? Object.keys(tables) : list(raw);
  return names.map((name) => {
    const table = tables[name];
    if (!table) throw new Error(`ROBOWRIGHT_PROJECT ${name}: no [projects.${name}] in ${config ?? "any robowright.toml"}; it has ${Object.keys(tables).join(", ") || "none"}`);
    const options = {};
    for (const [k, v] of Object.entries(table)) if (PROJECT_KEYS[k]) options[PROJECT_KEYS[k]] = v;
    if (table.timeout !== undefined) options.timeout = Number(table.timeout) * 1000; // seconds there, ms in node:test
    return { name, options };
  });
}

// Precedence, as in pytest: the environment (its flags), then the test's own options, then test.use, then the
// project, then the defaults.
function settings(own, project) {
  const o = { ...(project?.options ?? {}), ...fileOptions, ...own };
  const robots = list(process.env.ROBOWRIGHT_ROBOT || o.robot || "so101");
  const backends = list(process.env.ROBOWRIGHT_BACKEND || o.backend || "mujoco");
  const retries = Number(process.env.ROBOWRIGHT_RETRIES ?? o.retries ?? 0);
  const traceDir = o.traceDir ?? process.env.ROBOWRIGHT_TRACE_DIR ?? "robowright-traces";
  return { o, robots, backends, retries, traceDir };
}

async function attempt(rw, launchArgs, fn) {
  await rw.robot.launch(launchArgs);
  const fixtures = { rw, robot: rw.robot, expect: (subject, opts) => rw.expect(subject, opts) };
  await current.run(rw, () => fn(fixtures));
}

function register(name, own, fn) {
  for (const project of projects()) registerIn(name, own, fn, project);
}

function registerIn(name, own, fn, project) {
  const { o, robots, backends, retries, traceDir } = settings(own, project);
  const many = robots.length * backends.length > 1;
  for (const backend of backends) {
    for (const robot of robots) {
      const tag = [project?.name, ...(many ? [backend, robot] : [])].filter(Boolean).join("-");
      const full = tag ? `${name} [${tag}]` : name;
      const nodeOpts = {};
      for (const k of ["timeout", "skip", "todo", "only"]) if (o[k] !== undefined) nodeOpts[k] = o[k];
      const id = `${relative(process.cwd(), process.argv[1] ?? "") || "tests"}::${full}`;
      nodeTest(full, nodeOpts, async (t) => {
        const rw = await server_();
        const launchArgs = { robot, backend, seed: o.seed ?? 0 };
        for (const k of ["objects", "fidelity", "ros2", "robotOptions"]) if (o[k] !== undefined) launchArgs[k] = o[k];
        const record = { outcome: "passed", duration: 0, message: "", traces: [], trials: "", attachments: [], tags: [], annotations: [] };
        results[id] = record;
        const t0 = Date.now();
        // Saves the world's trace for a failed run, says where on the error, and keeps it for the report.
        const keep = async (e, suffix) => {
          const path = join(traceDir, `${slug(full)}${suffix}.zip`);
          try {
            mkdirSync(traceDir, { recursive: true });
            const saved = (await rw.robot.saveTrace({ path })).trim().split("\n").pop();
            const at = /\S+\.zip/.exec(saved)?.[0] ?? path;
            if (e) e.message += `\n\ntrace: ${at}\n  robowright show-trace ${at}    (or --text for a plain summary)`;
            t.diagnostic(`trace: ${at}`);
            record.traces.push(resolve(at));
          } catch {
            // no world to save (the launch itself failed): the error says why
          }
        };
        try {
          if (o.trials) await trials(t, rw, launchArgs, fn, o, record, keep);
          else await retried(t, rw, launchArgs, fn, retries, record, keep);
        } catch (e) {
          record.outcome = "failed";
          record.message = String(e?.stack ?? e);
          throw e;
        } finally {
          record.duration = (Date.now() - t0) / 1000;
        }
      });
    }
  }
}

async function retried(t, rw, launchArgs, fn, retries, record, keep) {
  let lastError = null;
  for (let i = 0; i <= retries; i++) {
    try {
      await attempt(rw, launchArgs, fn);
      if (i > 0) {
        t.diagnostic(`flaky: failed ${i} time${i > 1 ? "s" : ""}, then passed on retry ${i}`);
        record.outcome = "flaky";
        record.message = `failed ${i} time${i > 1 ? "s" : ""}, then passed on retry ${i}:\n${lastError?.message ?? ""}`;
      }
      return;
    } catch (e) {
      lastError = e;
      await keep(e, i ? `-retry${i}` : "");
    }
  }
  throw lastError;
}

async function trials(t, rw, launchArgs, fn, o, record, keep) {
  const n = Number(o.trials);
  const min = Number(o.minSuccess ?? 1);
  const lowerBound = Boolean(o.lowerBound);
  const meets = (k) => (lowerBound ? wilson(k, n)[0] : k / n) >= min - 1e-12;
  let need = 0;
  while (need <= n && !meets(need)) need++; // the fewest passes out of n that pass (n + 1: none do)
  const every = Boolean(process.env.ROBOWRIGHT_ALL_TRIALS);
  let passed = 0;
  let runs = 0;
  const failures = [];
  for (let i = 0; i < n; i++) {
    const seed = (launchArgs.seed ?? 0) + i;
    try {
      await attempt(rw, { ...launchArgs, seed, randomize: o.randomize ?? true }, fn);
      passed++;
    } catch (e) {
      failures.push(`seed ${seed}: ${String(e?.message ?? e).split("\n")[0]}`);
      await keep(null, `-trial${i}`);
    }
    runs++;
    if (!every && (passed >= need || runs - passed > n - need)) break; // the rest cannot change the verdict
  }
  const summary = trialSummary(passed, runs, n, min, lowerBound);
  record.trials = summary;
  t.diagnostic(summary);
  if (passed < need) throw new Error(`${summary}\n${failures.join("\n")}`);
}

export function test(name, optionsOrFn, maybeFn) {
  const [own, fn] = typeof optionsOrFn === "function" ? [{}, optionsOrFn] : [optionsOrFn ?? {}, maybeFn];
  register(name, own, fn);
}

// Options for every test in this file (Playwright's test.use): robot, backend, seed, objects, fidelity, ros2,
// robotOptions, retries, timeout, traceDir, and trials, minSuccess, lowerBound, randomize. Call it before the tests
// it applies to.
test.use = (options) => Object.assign(fileOptions, options);
test.only = (name, optionsOrFn, maybeFn) =>
  typeof optionsOrFn === "function" ? test(name, { only: true }, optionsOrFn) : test(name, { ...optionsOrFn, only: true }, maybeFn);
test.skip = (name, optionsOrFn, maybeFn) =>
  typeof optionsOrFn === "function" ? test(name, { skip: true }, optionsOrFn) : test(name, { ...optionsOrFn, skip: true }, maybeFn);
test.fixme = test.skip;
test.describe = describe;

// expect(subject) inside a test: the current test's world, as Playwright's expect needs no page passed.
export function expect(subject, options) {
  const rw = current.getStore();
  if (!rw) throw new Error("expect() from robowright/test is for use inside a test(); outside one, use rw.expect()");
  return rw.expect(subject, options);
}
