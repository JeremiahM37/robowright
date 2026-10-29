"""The Java client (clients/java): its generated tool classes match the server, and its own tests pass."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

ROOT = Path(__file__).resolve().parent.parent
JAVA = ROOT / "clients" / "java"


def test_the_java_client_is_generated_from_the_servers_current_tools():
    r = subprocess.run([sys.executable, "scripts/gen_java_client.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("javac") is None, reason="no JDK (javac) installed")
def test_the_java_clients_own_tests_pass_against_this_server(tmp_path):
    sources = sorted(str(p) for p in [*JAVA.glob("src/robowright/*.java"), *JAVA.glob("test/robowright/*.java")])
    r = subprocess.run(["javac", "-Xlint:all", "-Werror", "-d", str(tmp_path), *sources], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    exe = Path(sys.executable).with_name("robowright")
    env = {"ROBOWRIGHT_COMMAND": f"{exe} mcp"} if exe.exists() else {}
    r = subprocess.run(
        ["java", "-cp", str(tmp_path), "robowright.ClientTest"],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **env},
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "# fail 0" in r.stdout
