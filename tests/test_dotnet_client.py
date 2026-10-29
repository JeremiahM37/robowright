"""The .NET client (clients/dotnet): its generated tool classes match the server, and its own tests pass."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

ROOT = Path(__file__).resolve().parent.parent
DOTNET = ROOT / "clients" / "dotnet"


def test_the_dotnet_client_is_generated_from_the_servers_current_tools():
    r = subprocess.run([sys.executable, "scripts/gen_dotnet_client.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("dotnet") is None, reason="no .NET SDK (dotnet) installed")
def test_the_dotnet_clients_own_tests_pass_against_this_server(tmp_path):
    env = {**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"}
    r = subprocess.run(
        ["dotnet", "build", str(DOTNET / "Robowright.Tests"), "-c", "Release", "-o", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    exe = Path(sys.executable).with_name("robowright")
    if exe.exists():
        env["ROBOWRIGHT_COMMAND"] = f"{exe} mcp"
    r = subprocess.run([str(tmp_path / "Robowright.Tests")], capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "# fail 0" in r.stdout
