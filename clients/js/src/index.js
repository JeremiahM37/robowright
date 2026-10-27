// robowright for JavaScript and TypeScript: the same simulated robots, matchers and traces as the Python
// library, driven through the robowright MCP server (`robowright mcp`) over stdio. No dependencies.

import { spawn } from "node:child_process";
import { TOOLS } from "./tools.js";

const PROTOCOL = "2025-06-18";

export class RobowrightError extends Error {
  constructor(tool, message) {
    super(message);
    this.name = "RobowrightError";
    this.tool = tool;
  }
}

export class ExpectationError extends Error {
  constructor(tool, message) {
    super(message);
    this.name = "ExpectationError";
    this.tool = tool;
  }
}

const snake = (k) => k.replace(/[A-Z]/g, (c) => "_" + c.toLowerCase());

// The argument a matcher takes first, so expect("cube").toBeInside("bin") reads as it does in Python.
const PRIMARY = {
  to_be_inside: "container",
  to_be_holding: "obj",
  to_be_near: "target",
  to_be_above: "target",
  to_be_touching: "other",
  to_have_joint: "joint",
  to_be_visible: "camera",
  to_be_hidden: "camera",
};

class Connection {
  constructor(proc) {
    this.proc = proc;
    this.next = 1;
    this.pending = new Map();
    this.buffer = "";
    this.closed = null;
    proc.stdout.setEncoding("utf8");
    proc.stdout.on("data", (chunk) => this.#read(chunk));
    proc.on("exit", (code, signal) => this.#fail(new Error(`the robowright server exited (${signal ?? `code ${code}`})`)));
    proc.on("error", (e) => this.#fail(e));
  }

  #read(chunk) {
    this.buffer += chunk;
    let i;
    while ((i = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, i).trim();
      this.buffer = this.buffer.slice(i + 1);
      if (!line) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue; // not a protocol message
      }
      const waiter = msg.id !== undefined ? this.pending.get(msg.id) : undefined;
      if (!waiter) continue; // a notification, or a request from the server we do not serve
      this.pending.delete(msg.id);
      if (msg.error) waiter.reject(new Error(msg.error.message ?? JSON.stringify(msg.error)));
      else waiter.resolve(msg.result);
    }
  }

  #fail(err) {
    this.closed ??= err;
    for (const w of this.pending.values()) w.reject(err);
    this.pending.clear();
  }

  #send(msg) {
    if (this.closed) throw this.closed;
    this.proc.stdin.write(JSON.stringify(msg) + "\n");
  }

  request(method, params) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        this.#send({ jsonrpc: "2.0", id, method, params });
      } catch (e) {
        this.pending.delete(id);
        reject(e);
      }
    });
  }

  notify(method, params) {
    this.#send({ jsonrpc: "2.0", method, ...(params ? { params } : {}) });
  }
}

function text(result) {
  return (result.content ?? [])
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("\n");
}

export class Robowright {
  #conn;

  constructor(conn, server) {
    this.#conn = conn;
    this.server = server;
    for (const t of TOOLS) {
      this[t.ns] ??= {};
      this[t.ns][t.method] = (args) => this.call(t.tool, args);
    }
  }

  async call(tool, args = {}) {
    const params = {};
    for (const [k, v] of Object.entries(args ?? {})) if (v !== undefined) params[snake(k)] = v;
    const result = await this.#conn.request("tools/call", { name: tool, arguments: params });
    const out = text(result);
    if (result.isError) throw new RobowrightError(tool, out);
    return out;
  }

  expect(subject, options = {}) {
    const make = (negate) =>
      new Proxy(
        {},
        {
          get: (_, prop) => {
            if (prop === "not") return make(!negate);
            if (typeof prop !== "string" || prop === "then") return undefined;
            const matcher = snake(prop);
            return async (arg, opts = {}) => {
              const { timeout = options.timeout, ...rest } = opts;
              let args = {};
              if (arg !== undefined && arg !== null && typeof arg === "object" && !Array.isArray(arg)) args = arg;
              else if (arg !== undefined) {
                if (!PRIMARY[matcher]) throw new TypeError(`${prop} takes an options object, not ${JSON.stringify(arg)}`);
                args = { [PRIMARY[matcher]]: arg };
              }
              args = Object.fromEntries(Object.entries({ ...args, ...rest }).map(([k, v]) => [snake(k), v]));
              const out = await this.call("robot_expect", { subject, matcher, args, negate, timeout });
              if (!out.startsWith("PASS")) throw new ExpectationError("robot_expect", out);
              return out;
            };
          },
        },
      );
    return make(false);
  }

  async close() {
    const proc = this.#conn.proc;
    if (proc.exitCode !== null || proc.signalCode !== null) return;
    const exited = new Promise((r) => proc.once("exit", r));
    try {
      await this.call("robot_close");
    } catch {
      // nothing open, or the server is already going
    }
    proc.stdin.end();
    const timer = setTimeout(() => proc.kill("SIGTERM"), 5000);
    await exited;
    clearTimeout(timer);
  }
}

export async function launch(options = {}) {
  const fromEnv = (process.env.ROBOWRIGHT_COMMAND ?? "").split(" ").filter(Boolean);
  const command = options.command ?? (fromEnv.length ? fromEnv : ["robowright", "mcp"]);
  const proc = spawn(command[0], command.slice(1), {
    cwd: options.cwd,
    env: options.env ?? process.env,
    stdio: ["pipe", "pipe", options.stderr ?? "inherit"],
  });
  const conn = new Connection(proc);
  try {
    const init = await conn.request("initialize", {
      protocolVersion: PROTOCOL,
      capabilities: {},
      clientInfo: { name: "robowright-js", version: "0.1.0" },
    });
    conn.notify("notifications/initialized");
    return new Robowright(conn, init.serverInfo ?? {});
  } catch (e) {
    proc.kill("SIGTERM");
    throw e;
  }
}

export { TOOLS };
