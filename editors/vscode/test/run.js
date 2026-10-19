// Runs the extension's tests inside a real VS Code (downloaded once by @vscode/test-electron), on a small project of
// robot tests. ROBOWRIGHT_PYTHON is the Python with robowright installed (default: the repository's .venv).
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { runTests } = require("@vscode/test-electron");

async function main() {
  const repo = path.resolve(__dirname, "../../..");
  const python = process.env.ROBOWRIGHT_PYTHON || path.join(repo, ".venv", "bin", "python");
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), "rw-vscode-"));
  fs.mkdirSync(path.join(ws, "tests"));
  fs.mkdirSync(path.join(ws, ".vscode"));
  fs.cpSync(path.join(__dirname, "fixture"), path.join(ws, "tests"), { recursive: true });
  fs.writeFileSync(path.join(ws, ".vscode", "settings.json"), JSON.stringify({ "robowright.python": python, "robowright.pytestArgs": ["tests"] }));
  try {
    await runTests({
      version: process.env.VSCODE_VERSION || "1.95.3",
      extensionDevelopmentPath: path.resolve(__dirname, ".."),
      extensionTestsPath: path.join(__dirname, "suite.js"),
      launchArgs: [ws, "--disable-extensions", "--disable-workspace-trust"],
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
