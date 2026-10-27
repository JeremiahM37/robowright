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
        ["node", "--test", "--test-reporter=tap", "test/client.test.js"],
        cwd=ROOT / "clients" / "js",
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **env},
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "# fail 0" in r.stdout
