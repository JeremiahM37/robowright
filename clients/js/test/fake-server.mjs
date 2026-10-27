// A stand-in for `robowright mcp`: answers the MCP handshake and tools/call from a script, so the client's
// own code is tested without Python. Each call is echoed back, so tests can see what was sent.
import { createInterface } from "node:readline";

const rl = createInterface({ input: process.stdin });
const send = (m) => process.stdout.write(JSON.stringify(m) + "\n");
process.stdout.write("not a protocol line\n");
rl.on("line", (line) => {
  const m = JSON.parse(line);
  if (m.method === "initialize") {
    send({ jsonrpc: "2.0", method: "notifications/message", params: { level: "info", data: "hello" } });
    send({ jsonrpc: "2.0", id: m.id, result: { protocolVersion: m.params.protocolVersion, capabilities: {}, serverInfo: { name: "fake", version: "9" } } });
  } else if (m.method === "tools/call") {
    const { name, arguments: args } = m.params;
    if (name === "robot_crash") process.exit(3);
    if (name === "robot_pick" && args.object === "nope") {
      send({ jsonrpc: "2.0", id: m.id, result: { isError: true, content: [{ type: "text", text: "KeyError: no object named 'nope'" }] } });
    } else if (name === "robot_expect") {
      const ok = (args.args.container === "bin") !== args.negate;
      send({ jsonrpc: "2.0", id: m.id, result: { content: [{ type: "text", text: `${ok ? "PASS" : "FAIL"}: ${JSON.stringify(args)}` }] } });
    } else {
      send({ jsonrpc: "2.0", id: m.id, result: { content: [{ type: "text", text: JSON.stringify({ name, args }) }] } });
    }
  }
});
