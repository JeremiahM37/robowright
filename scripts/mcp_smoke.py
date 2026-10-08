"""Drive `robowright mcp` over stdio the way an MCP client does, and run the test it writes.

python scripts/mcp_smoke.py OUT_DIR
"""

import asyncio
import base64
import subprocess
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters


def text(r):
    return "\n".join(getattr(c, "text", "") for c in r.content)


async def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    server = StdioServerParameters(command=sys.executable, args=["-m", "robowright.cli", "mcp"])
    async with Client(server, read_timeout_seconds=300) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert "robot_launch" in names and "robot_generate_test" in names, names
        assert "cube" in text(await c.call_tool("robot_launch", {"robot": "so101"}))
        bad = await c.call_tool("robot_pick", {"object": "banana"})
        assert bad.is_error and "scene has" in text(bad), text(bad)
        assert "holding cube" in text(await c.call_tool("robot_pick", {"object": "cube"}))
        await c.call_tool("robot_place", {"on": "bin"})
        check = text(await c.call_tool("robot_expect", {"subject": "cube", "matcher": "to_be_inside", "args": {"container": "bin"}}))
        assert check.startswith("PASS"), check
        shot = (await c.call_tool("robot_screenshot", {"camera": "top", "width": 320, "height": 240})).content[0]
        assert shot.mime_type == "image/png" and base64.b64decode(shot.data)[:4] == b"\x89PNG"
        await c.call_tool("robot_generate_test", {"test_name": "test_from_mcp", "path": str(out / "test_from_mcp.py")})
        trace = text(await c.call_tool("robot_save_trace", {"path": str(out / "session.zip")})).strip()
        read = text(await c.call_tool("robot_read_trace", {"path": trace}))
        assert "robot.pick" in read and "cube: at" in read, read
        await c.call_tool("robot_close", {})
        # The agent loop: run the tests, read why one failed.
        ok = text(await c.call_tool("robot_run_tests", {"target": str(out / "test_from_mcp.py"), "cwd": str(out)}))
        assert ok.startswith("exit code 0"), ok
        (out / "test_wrong.py").write_text(
            "from robowright import expect\n\n"
            "def test_wrong_spot(robot, scene):\n"
            "    robot.pick(scene['cube'])\n"
            "    robot.place(on=(0.25, -0.1, 0.0))\n"
            "    expect(scene['cube']).to_be_inside(scene['bin'], timeout=0.5)\n"
        )
        bad = text(await c.call_tool("robot_run_tests", {"target": str(out / "test_wrong.py"), "cwd": str(out)}))
        assert bad.startswith("exit code 1") and "expect(cube).to_be_inside" in bad and "at the failure" in bad, bad
    test = str(out / "test_from_mcp.py")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test], capture_output=True, text=True)
    print(run.stdout.strip().splitlines()[-1])
    return run.returncode


if __name__ == "__main__":
    sys.exit(asyncio.run(main(Path(sys.argv[1]))))
