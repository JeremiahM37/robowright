"""Build a trace of a failing test and render it to HTML for the verify UI check."""

import sys
from pathlib import Path

import robowright as rw
from robowright import expect
from robowright.viewer import write_html

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/robowright-verify")
out.mkdir(parents=True, exist_ok=True)
w = rw.launch(name="verify::test_place_in_wrong_spot", settings=rw.Settings(trace="on", trace_dir=str(out)))
w.robot.reset_to()
w.robot.pick(w.scene["cube"])
w.robot.place(on=(0.25, -0.1, 0.0))
try:
    expect(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=0.5)
except rw.ExpectationError:
    pass
path = w.close()
print(write_html(path, out / "trace.html"))
