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


@pytest.mark.skipif(shutil.which("mvn") is None, reason="no Maven installed")
def test_the_java_client_builds_with_maven(tmp_path):
    """The build Maven Central gets the client from: clients/java/pom.xml compiles it with -Werror and packages the jar."""
    r = subprocess.run(
        ["mvn", "-q", "-B", "-f", str(JAVA / "pom.xml"), "package", f"-Dmaven.repo.local={tmp_path / 'repo'}"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert list((JAVA / "target").glob("robowright-*.jar"))


JUNIT = "junit-platform-console-standalone-1.11.4.jar"
JUNIT_SHA1 = "44650aaff63a4dbb63cb0bf0f3725a4eb9406da1"


def _junit_jar(tmp_path) -> Path | None:
    """JUnit's standalone launcher: from the cache, else from Maven Central (checked against its SHA-1), else None."""
    import hashlib
    import urllib.request

    cached = Path(os.environ.get("ROBOWRIGHT_JUNIT_JAR", Path.home() / ".cache" / "robowright" / "junit" / JUNIT))
    if cached.exists():
        return cached
    url = f"https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.11.4/{JUNIT}"
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
    except OSError:
        return None
    if hashlib.sha1(data).hexdigest() != JUNIT_SHA1:
        return None
    jar = tmp_path / JUNIT
    jar.write_bytes(data)
    return jar


def _run_junit(tmp_path, cls: str, env: dict) -> tuple[subprocess.CompletedProcess, str]:
    jar = _junit_jar(tmp_path)
    if jar is None:
        pytest.skip("JUnit's console launcher is not cached and could not be downloaded")
    out = tmp_path / "classes"
    sources = sorted(
        str(p)
        for p in [
            *JAVA.glob("src/robowright/*.java"),
            JAVA / "test" / "robowright" / "FakeServer.java",
            *JAVA.glob("junit/src/robowright/junit/*.java"),
            *JAVA.glob("junit/test/robowright/junit/*.java"),
        ]
    )
    r = subprocess.run(["javac", "-Xlint:all", "-Werror", "-cp", str(jar), "-d", str(out), *sources], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    work = tmp_path / "work"
    work.mkdir()
    cmd = ["java", "-jar", str(jar), "execute", "--class-path", str(out), "--select-class", cls, "--reports-dir", "reports"]
    r = subprocess.run(
        [*cmd, "--disable-banner", "--disable-ansi-colors"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **env},
    )
    xml = "\n".join(p.read_text() for p in (work / "reports").glob("*.xml"))
    return r, xml


@pytest.mark.skipif(shutil.which("javac") is None, reason="no JDK (javac) installed")
def test_robottest_runs_each_robot_names_a_failures_trace_and_retries(tmp_path):
    fake = f"java -cp {tmp_path / 'classes'} robowright.FakeServer"
    r, xml = _run_junit(tmp_path, "robowright.junit.SampleTest", {"ROBOWRIGHT_COMMAND": fake})
    assert r.returncode == 1, r.stdout[-3000:]  # one test fails, on purpose
    assert "[mujoco-panda]" in r.stdout and "[mujoco-ur5e]" in r.stdout
    assert "4 tests successful" not in r.stdout and "3 tests successful" in r.stdout and "1 tests failed" in r.stdout
    assert "trace: robowright-traces/SampleTest.failsWithItsTrace_so101.zip" in r.stdout
    assert "robowright show-trace robowright-traces/SampleTest.failsWithItsTrace_so101.zip" in r.stdout
    assert "failed 1 time, then passed on retry 1" in r.stdout + xml


@pytest.mark.skipif(shutil.which("javac") is None, reason="no JDK (javac) installed")
def test_robottest_against_the_real_server(tmp_path):
    exe = Path(sys.executable).with_name("robowright")
    if not exe.exists():
        pytest.skip("robowright is not installed beside this Python")
    r, xml = _run_junit(tmp_path, "robowright.junit.RealServerTest", {"ROBOWRIGHT_COMMAND": f"{exe} mcp"})
    assert r.returncode == 1, r.stdout[-3000:]
    assert "1 tests successful" in r.stdout and "1 tests failed" in r.stdout, r.stdout[-3000:]
    assert "bin spans" in r.stdout  # the matcher's own reason
    trace = tmp_path / "work" / "robowright-traces" / "RealServerTest.leavesTheCubeWhereItWas_so101.zip"
    assert trace.exists(), r.stdout[-3000:]
    text = subprocess.run([str(exe), "show-trace", "--text", str(trace)], capture_output=True, text=True)
    assert text.returncode == 0 and "to_be_inside" in text.stdout
