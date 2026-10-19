"""The HTML report of a test run (``pytest --rw-report report.html``), as Playwright's is.

One page: every test with its outcome and duration, trials with their rates, and for each
failure the message, the trace as text and a link to the trace in the viewer.

A sharded run (``--rw-shard 2/4`` on four CI machines) writes a blob per shard instead (``pytest --rw-blob
shard-2.zip``): the results with every failure's trace and attachments inside it, so it can be carried off the machine
that ran it. ``robowright merge-reports shard-*.zip -o report.html`` makes one report of them, as Playwright's blob
reporter and ``merge-reports`` do.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import zipfile
from pathlib import Path

_TRACE = re.compile(r"trace: (\S+\.zip)")
_ORDER = ["failed", "xpassed", "flaky", "passed", "skipped", "xfailed"]


class Run:
    """The results of a session, gathered from pytest's reports (on the xdist controller too)."""

    def __init__(self):
        self.tests: dict[str, dict] = {}
        self.started = time.time()
        self.seconds: float | None = None  # a merged run's span; a live run's is measured when written

    def add(self, report) -> None:
        t = self.tests.setdefault(
            report.nodeid, {"outcome": "passed", "duration": 0.0, "message": "", "traces": [], "trials": "", "attachments": []}
        )
        t["duration"] += getattr(report, "duration", 0.0)
        retried = report.outcome == "rerun"  # an attempt that failed and was run again (--rw-retries)
        flaky = any(k == "robowright_flaky" for k, _ in report.user_properties)
        for k, v in report.user_properties:
            if k == "robowright_trace" and (report.failed or retried or flaky) and v not in t["traces"]:
                t["traces"].append(v)
            if k == "robowright_trials":
                t["trials"] = v
            if k == "robowright_flaky":
                t["outcome"] = "flaky"  # failed, then passed on a retry: as Playwright reports it
            if k == "robowright_attachment":
                a = json.loads(v)
                if a not in t["attachments"]:
                    t["attachments"].append(a)
        xfail = hasattr(report, "wasxfail")
        if retried:
            t["message"] = t["message"] or f"attempt failed and was retried:\n{report.longrepr}"
        elif report.failed:
            t["outcome"] = "failed" if report.when == "call" or t["outcome"] != "failed" else t["outcome"]
            t["message"] = t["message"] or str(report.longrepr)
            t["traces"] += [p for p in _TRACE.findall(str(report.longrepr)) if p not in t["traces"]]
        elif report.skipped:
            t["outcome"] = "xfailed" if xfail else "skipped"
            reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
            t["message"] = report.wasxfail if xfail else str(reason)
        elif report.when == "call" and xfail:
            t["outcome"] = "xpassed"

    def write(self, out: str | Path) -> Path:
        from .trace import Trace
        from .viewer import write_html

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pages = out.parent / (out.stem + "-traces")
        counts: dict[str, int] = {}
        rows = []
        for nodeid, t in self.tests.items():
            counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
            details = ""
            if t["message"] and t["outcome"] in ("failed", "xpassed", "flaky", "skipped", "xfailed"):
                details += f"<pre>{html.escape(t['message'][-6000:])}</pre>"
            for path in t["traces"]:
                p = Path(path)
                if not p.exists():
                    continue
                link = ""
                try:
                    pages.mkdir(exist_ok=True)
                    page = write_html(p, pages / (p.stem + ".html"))
                    link = f'<a href="{html.escape(str(page.relative_to(out.parent)))}">open the trace</a> · '
                    text = Trace(p).summary()
                except Exception as e:  # noqa: BLE001 - one unreadable trace leaves the report whole
                    text = f"(could not read the trace: {e})"
                details += f'<div class="trace">{link}<code>{html.escape(str(p))}</code><pre>{html.escape(text)}</pre></div>'
            for a in t["attachments"]:
                p = Path(a["path"])
                if not p.exists():
                    continue
                href = html.escape(Path(os.path.relpath(p.resolve(), out.parent.resolve())).as_posix())
                label, kind = html.escape(a["name"]), html.escape(a["type"])
                if a["type"].startswith("image/"):
                    details += f'<figure><img src="{href}" alt="{label}"><figcaption>{label}</figcaption></figure>'
                elif a["type"].startswith("video/"):
                    details += f'<figure><video src="{href}" controls preload="metadata"></video><figcaption>{label}</figcaption></figure>'
                else:
                    details += f'<div class="att"><a href="{href}">{label}</a> <span class="trials">{kind}</span></div>'
            trials = f'<span class="trials">{html.escape(t["trials"])}</span>' if t["trials"] else ""
            name = f"{html.escape(nodeid)}{trials}"
            body = f"<details><summary>{name}</summary>{details}</details>" if details else f"<span>{name}</span>"
            row = f'<tr class="{t["outcome"]}"><td class="o">{t["outcome"]}</td><td>{body}</td><td class="d">{t["duration"]:.1f}s</td></tr>'
            rows.append((_ORDER.index(t["outcome"]), row))
        order = _ORDER
        buttons = "".join(f'<button data-o="{o}">{o} {counts[o]}</button>' for o in order if counts.get(o))
        out.write_text(
            _PAGE.format(
                title="robowright report",
                when=time.strftime("%Y-%m-%d %H:%M", time.localtime(self.started)),
                total=len(self.tests),
                seconds=self.seconds if self.seconds is not None else time.time() - self.started,
                buttons=buttons,
                rows="\n".join(r for _, r in sorted(rows, key=lambda x: x[0])),
            )
        )
        return out

    def save_blob(self, path: str | Path) -> Path:
        """The results, with the trace and attachment files they name, in one zip (see the module)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tests, n = {}, 0
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for nodeid, t in self.tests.items():
                t = {**t, "traces": [], "attachments": []}
                for src in self.tests[nodeid]["traces"]:
                    if Path(src).exists():
                        n += 1
                        name = f"files/{n}/{Path(src).name}"
                        z.write(src, name)
                        t["traces"].append(name)
                for a in self.tests[nodeid]["attachments"]:
                    if Path(a["path"]).exists():
                        n += 1
                        name = f"files/{n}/{Path(a['path']).name}"
                        z.write(a["path"], name)
                        t["attachments"].append({**a, "path": name})
                tests[nodeid] = t
            run = {"format": 1, "started": self.started, "seconds": time.time() - self.started, "tests": tests}
            z.writestr("run.json", json.dumps(run))
        return path


def merge(blobs: list, out: str | Path) -> Path:
    """One HTML report at ``out`` from the blobs of a sharded run. Their files are unpacked next to it, in
    ``<report>-files/<blob>/``. A test in more than one blob (the same suite run twice) is listed once for each."""
    out = Path(out)
    files = out.parent / (out.stem + "-files")
    run, spans = Run(), []
    for blob in blobs:
        blob = Path(blob)
        dest = files / blob.stem
        with zipfile.ZipFile(blob) as z:
            data = json.loads(z.read("run.json"))
            if data.get("format") != 1:
                raise ValueError(f"{blob}: not a robowright report blob (pytest --rw-blob)")
            for name in z.namelist():
                if name.startswith("files/") and ".." not in Path(name).parts:
                    z.extract(name, dest)
        spans.append((data["started"], data["started"] + data["seconds"]))
        for nodeid, t in data["tests"].items():
            t["traces"] = [str(dest / p) for p in t["traces"]]
            t["attachments"] = [{**a, "path": str(dest / a["path"])} for a in t["attachments"]]
            key = nodeid if nodeid not in run.tests else f"{nodeid} [{blob.stem}]"
            run.tests[key] = t
    if not spans:
        raise ValueError("merge-reports needs at least one blob")
    run.started = min(a for a, _ in spans)
    run.seconds = max(b for _, b in spans) - run.started  # from the first shard's start to the last one's end
    return run.write(out)


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg:#fff;
  --fg:#1d1d1f;
  --mut:#6e6e73;
  --line:#e5e5ea;
  --fail:#c62828;
  --pass:#2e7d32;
  --skip:#8e8e93;
  --flaky:#a15c00;
  --code:#f5f5f7;
}}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111;
  --fg:#f2f2f2;
  --mut:#a1a1a6;
  --line:#2c2c2e;
  --fail:#ff6b6b;
  --pass:#6bd56b;
  --skip:#8e8e93;
  --flaky:#f0a64a;
  --code:#1c1c1e;
  }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .meta {{ color:var(--mut); margin-bottom:12px; }}
button {{
  font:inherit;
  margin:0 6px 12px 0;
  padding:4px 10px;
  border:1px solid var(--line);
  border-radius:6px;
  background:var(--code);
  color:var(--fg);
  cursor:pointer;
}}
button.on {{ border-color:var(--fg); }}
table {{ width:100%; border-collapse:collapse; }} td {{ border-top:1px solid var(--line); padding:6px 4px; vertical-align:top; }}
td.o {{ width:5.5em; font-weight:600; }} td.d {{ width:4.5em; text-align:right; color:var(--mut); }}
tr.failed td.o, tr.xpassed td.o {{
  color:var(--fail);
  }} tr.passed td.o {{ color:var(--pass);
  }} tr.skipped td.o, tr.xfailed td.o {{ color:var(--skip);
  }} tr.flaky td.o {{ color:var(--flaky);
}}
summary {{ cursor:pointer; overflow-wrap:anywhere; }} .trials {{ color:var(--mut); margin-left:8px; }}
pre {{ background:var(--code); padding:8px; border-radius:6px; overflow-x:auto; font-size:12px; white-space:pre-wrap; }}
.trace {{ margin-top:8px; }} .att {{ margin-top:6px; }}
figure {{ margin:8px 0; }} figure img {{ max-width:100%; height:auto; border-radius:6px; border:1px solid var(--line); }}
figcaption {{ color:var(--mut); font-size:12px; }}
table {{ table-layout:fixed; }}
td, code, summary {{ overflow-wrap:anywhere; }}
pre {{ max-width:100%; box-sizing:border-box; }}
@media (max-width: 600px) {{ td.o {{ width:4.5em; }} td.d {{ width:3.5em; }} }} a {{ color:inherit; }}
</style></head><body>
<h1>{title}</h1><div class="meta">{when} · {total} tests · {seconds:.0f} s</div>
<div><button data-o="" class="on">all {total}</button>{buttons}</div>
<table>{rows}</table>
<script>
for (const b of document.querySelectorAll("button")) b.onclick = () => {{
  for (const x of document.querySelectorAll("button")) x.classList.toggle("on", x === b);
  for (const r of document.querySelectorAll("tr")) r.hidden = b.dataset.o && r.className !== b.dataset.o;
}};
</script></body></html>
"""
