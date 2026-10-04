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
        await c.call_tool("robot_close", {})
    test = str(out / "test_from_mcp.py")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test], capture_output=True, text=True)
    print(run.stdout.strip().splitlines()[-1])
    return run.returncode


if __name__ == "__main__":
    sys.exit(asyncio.run(main(Path(sys.argv[1]))))
