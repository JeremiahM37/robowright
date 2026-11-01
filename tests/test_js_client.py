"""The JavaScript client (clients/js): its generated tool table matches the server, and its own tests pass."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

ROOT = Path(__file__).resolve().parent.parent


def test_the_js_client_is_generated_from_the_servers_current_tools():
    r = subprocess.run([sys.executable, "scripts/gen_js_client.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_js_clients_own_tests_pass_against_this_server():
    exe = Path(sys.executable).with_name("robowright")
    env = {"ROBOWRIGHT_COMMAND": f"{exe} mcp"} if exe.exists() else {}
    r = subprocess.run(
        ["node", "--test", "--test-reporter=tap", "test/client.test.js", "test/runner.test.js"],
        cwd=ROOT / "clients" / "js",
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **env},
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "# fail 0" in r.stdout


def test_the_js_client_is_found_in_a_checkout():
    from robowright import cli

    assert cli.js_client() == ROOT / "clients" / "js"  # in a checkout; robowright/js inside a wheel


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is not installed")
def test_robowright_init_js_then_npm_install_and_npm_test_pass(tmp_path):
    import json

    proj = tmp_path / "robot-tests"
    r = subprocess.run([sys.executable, "-m", "robowright.cli", "init", "--js", str(proj)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "npm install" in r.stdout and "npm test" in r.stdout
    package = json.loads((proj / "package.json").read_text())
    assert package["type"] == "module" and package["scripts"]["test"] == "node --test tests/"
    assert package["devDependencies"]["robowright"] == f"file:{ROOT / 'clients' / 'js'}"
    ci = (proj / ".github" / "workflows" / "robowright.yml").read_text()
    assert 'npm install --no-save "$(robowright js-client)"' in ci and "merge-reports results/*.json" in ci
    assert (proj / ".mcp.json").exists() and (proj / "robowright.toml").exists()
    assert subprocess.run(["npm", "install", "--silent"], cwd=proj, capture_output=True, timeout=300).returncode == 0
    exe = Path(sys.executable).with_name("robowright")
    env = {**os.environ, "PATH": f"{exe.parent}{os.pathsep}{os.environ['PATH']}"}
    env.pop("NODE_TEST_CONTEXT", None)
    t = subprocess.run(["node", "--test", "--test-reporter=tap", "tests/"], cwd=proj, capture_output=True, text=True, timeout=600, env=env)
    assert t.returncode == 0, t.stdout[-3000:] + t.stderr[-2000:]
    assert "# pass 2" in t.stdout and "# fail 0" in t.stdout
