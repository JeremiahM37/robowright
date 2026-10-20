// Runs the extension's tests inside a real VS Code (downloaded once by @vscode/test-electron), on a small project of
// robot tests. ROBOWRIGHT_PYTHON is the Python with robowright installed (default: the repository's .venv).
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const cp = require("child_process");
const { downloadAndUnzipVSCode, resolveCliArgsFromVSCodeExecutablePath, runTests } = require("@vscode/test-electron");

async function main() {
  const repo = path.resolve(__dirname, "../../..");
  const python = process.env.ROBOWRIGHT_PYTHON || path.join(repo, ".venv", "bin", "python");
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), "rw-vscode-"));
  fs.mkdirSync(path.join(ws, "tests"));
  fs.mkdirSync(path.join(ws, ".vscode"));
  fs.cpSync(path.join(__dirname, "fixture"), path.join(ws, "tests"), { recursive: true });
  fs.writeFileSync(path.join(ws, ".vscode", "settings.json"), JSON.stringify({ "robowright.python": python, "robowright.pytestArgs": ["tests"] }));
  const version = process.env.VSCODE_VERSION || "1.95.3";
  const vscodeExecutablePath = await downloadAndUnzipVSCode(version);
  // the Debug profile needs the Python Debugger: installed into an extensions folder of its own, the only one
  // loaded (ROBOWRIGHT_TEST_NO_DEBUGPY=1 leaves it out, to test the message instead)
  const exts = path.join(__dirname, "..", ".vscode-test", "extensions-debugpy");
  const installed = fs.existsSync(exts) && fs.readdirSync(exts).some(d => d.startsWith("ms-python.debugpy"));
  if (!process.env.ROBOWRIGHT_TEST_NO_DEBUGPY && !installed) {
    const [cli, ...cliArgs] = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath);
    cp.execFileSync(cli, [...cliArgs, "--extensions-dir", exts, "--install-extension", "ms-python.debugpy"], { stdio: "inherit" });
  }
  const extArgs = process.env.ROBOWRIGHT_TEST_NO_DEBUGPY ? ["--disable-extensions"] : ["--extensions-dir", exts];
  try {
    await runTests({
      vscodeExecutablePath,
      extensionDevelopmentPath: path.resolve(__dirname, ".."),
      extensionTestsPath: path.join(__dirname, "suite.js"),
      launchArgs: [ws, ...extArgs, "--disable-workspace-trust"],
      extensionTestsEnv: { ROBOWRIGHT_TEST_WORKSPACE: ws },
    });
  } finally {
    fs.rmSync(ws, { recursive: true, force: true });
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
