"""robowright show-report, Playwright's show-report: open the report a run wrote, or serve it."""

import subprocess
import sys
import time
import urllib.request

from robowright.cli import main


def test_it_opens_the_report_a_run_wrote(tmp_path, capsys):
    (tmp_path / "report.html").write_text("<!doctype html><title>robowright report</title>")
    assert main(["show-report", str(tmp_path), "--no-open"]) == 0  # a folder finds its report.html
    assert capsys.readouterr().out.strip().endswith("report.html")


def test_no_report_says_how_to_make_one(tmp_path, capsys):
    assert main(["show-report", str(tmp_path / "nope.html"), "--no-open"]) == 2
    assert "pytest --rw-report" in capsys.readouterr().err


def test_it_serves_the_report_with_its_traces(tmp_path):
    (tmp_path / "report.html").write_text('<a href="report-traces/t.html">open the trace</a>')
    (tmp_path / "report-traces").mkdir()
    (tmp_path / "report-traces" / "t.html").write_text("the trace")
    p = subprocess.Popen(
        [sys.executable, "-m", "robowright.cli", "show-report", str(tmp_path / "report.html"), "--port", "0", "--no-open"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = p.stdout.readline()  # serving http://127.0.0.1:PORT/report.html (Ctrl-C to stop)
        url = line.split()[1]
        deadline = time.monotonic() + 10
        while True:
            try:
                page = urllib.request.urlopen(url, timeout=2).read().decode()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        assert "open the trace" in page
        trace = urllib.request.urlopen(url.replace("report.html", "report-traces/t.html"), timeout=2).read().decode()
        assert trace == "the trace"
    finally:
        p.terminate()
        p.wait(timeout=10)
