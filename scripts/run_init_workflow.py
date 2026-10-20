"""Run the CI workflow `robowright init` writes, step by step, the way GitHub Actions would: both shards of the test
job, then the report job's merge. CI runs this on every push (the init-workflow job), so the workflow handed to every
new project is known to work, not just to read well.

    python scripts/run_init_workflow.py PROJECT_DIR SOURCE_DIR

PROJECT_DIR is a project `robowright init` made; SOURCE_DIR is the checkout being tested, installed in place of the
workflow's own install line (which installs the published version).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


def main(project: str, source: str) -> int:
    project = Path(project).resolve()
    wf = yaml.safe_load((project / ".github/workflows/robowright.yml").read_text())
    test, report = wf["jobs"]["test"], wf["jobs"]["report"]
    shards = test["strategy"]["matrix"]["shard"]

    def run(cmd: str, **subst) -> None:
        for k, v in subst.items():
            cmd = cmd.replace(f"${{{{ matrix.{k} }}}}", str(v))
        if re.search(r"\$\{\{", cmd):
            raise SystemExit(f"a step uses a GitHub expression this script does not know: {cmd}")
        if cmd.startswith("sudo apt-get") and not os.environ.get("GITHUB_ACTIONS"):
            print(f"(skipped off a runner) $ {cmd}", flush=True)  # a system package install belongs on CI only
            return
        if cmd.startswith("pip install "):  # the published robowright: the checkout being tested instead
            cmd = re.sub(r'pip install ("robowright[^"]*"|robowright\S*)', f"pip install {source}", cmd)
        print(f"$ {cmd}", flush=True)
        subprocess.run(cmd, shell=True, cwd=project, check=True)

    for shard in shards:
        for step in test["steps"]:
            if "run" in step:
                run(step["run"], shard=shard)
    for step in report["steps"]:
        if "run" in step:
            run(step["run"])
    out = project / "report" / "report.html"
    if not out.exists():
        raise SystemExit("the report job wrote no report")
    print(f"ok: {len(shards)} shards ran and {out} was written")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
