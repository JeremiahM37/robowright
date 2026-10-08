"""The HTML report of a test run (``pytest --rw-report report.html``), as Playwright's is.

One page: every test with its outcome and duration, trials with their rates, and for each
failure the message, the trace as text and a link to the trace in the viewer.
"""

from __future__ import annotations

import html
import re
import time
from pathlib import Path

_TRACE = re.compile(r"trace: (\S+\.zip)")
_ORDER = ["failed", "xpassed", "passed", "skipped", "xfailed"]


class Run:
    """The results of a session, gathered from pytest's reports (on the xdist controller too)."""

    def __init__(self):
        self.tests: dict[str, dict] = {}
        self.started = time.time()

    def add(self, report) -> None:
        t = self.tests.setdefault(report.nodeid, {"outcome": "passed", "duration": 0.0, "message": "", "traces": [], "trials": ""})
        t["duration"] += getattr(report, "duration", 0.0)
        for k, v in report.user_properties:
            if k == "robowright_trace" and report.failed and v not in t["traces"]:
                t["traces"].append(v)
            if k == "robowright_trials":
                t["trials"] = v
        xfail = hasattr(report, "wasxfail")
        if report.failed:
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
            if t["message"] and t["outcome"] in ("failed", "xpassed", "skipped", "xfailed"):
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
                seconds=time.time() - self.started,
                buttons=buttons,
                rows="\n".join(r for _, r in sorted(rows, key=lambda x: x[0])),
            )
        )
        return out


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
  --code:#f5f5f7;
}}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111;
  --fg:#f2f2f2;
  --mut:#a1a1a6;
  --line:#2c2c2e;
  --fail:#ff6b6b;
  --pass:#6bd56b;
  --skip:#8e8e93;
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
}}
summary {{ cursor:pointer; overflow-wrap:anywhere; }} .trials {{ color:var(--mut); margin-left:8px; }}
pre {{ background:var(--code); padding:8px; border-radius:6px; overflow-x:auto; font-size:12px; white-space:pre-wrap; }}
.trace {{ margin-top:8px; }}
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
