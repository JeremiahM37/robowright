"""rw.step, Playwright's test.step: named parts of a test, in the trace and in the error."""

import pytest

import robowright as rw
from robowright import expect


def test_steps_are_in_the_trace_with_what_ran_inside_them(tmp_path):
    w = rw.launch(settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name="steps")
    with rw.step("put the cube in the bin"):
        with rw.step("look"):
            expect(w.scene["cube"]).to_be_at_rest()
        w.step(10)
    path = w.close(trace_path=tmp_path / "steps.zip")
    t = rw.Trace(path)
    steps = [e for e in t.events if e["type"] == "step"]
    assert [(e["name"], e.get("depth", 0), e["status"]) for e in steps] == [("put the cube in the bin", 0, "ok"), ("look", 1, "ok")]
    outer = steps[0]
    assert outer["end_t"] > outer["t"]  # it lasted while the world stepped inside it
    inner_expect = next(e for e in t.events if e["type"] == "expect")
    assert inner_expect["depth"] == 2
    text = t.summary()
    assert "step 'put the cube in the bin'" in text
    assert "    step 'look'" in text  # indented under the step it is in


def test_an_error_inside_a_step_names_the_step(tmp_path):
    w = rw.launch(settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name="steps-fail")
    with pytest.raises(rw.ExpectationError) as info:
        with rw.step("put the cube in the bin"), rw.step("check it landed"):
            expect(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=0.1)
    assert "in step 'put the cube in the bin > check it landed'" in "\n".join(getattr(info.value, "__notes__", []))
    path = w.close(failed=True, trace_path=tmp_path / "fail.zip")
    steps = [e for e in rw.Trace(path).events if e["type"] == "step"]
    assert [e["status"] for e in steps] == ["failed", "failed"]
    assert "ExpectationError" in steps[1]["detail"]


def test_a_trace_without_steps_reads_as_before(tmp_path):
    w = rw.launch(settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name="plain")
    expect(w.scene["cube"]).to_be_at_rest()
    t = rw.Trace(w.close(trace_path=tmp_path / "plain.zip"))
    assert all("depth" not in e for e in t.events)


def test_expect_world_to_satisfy_polls_any_value_of_the_world():
    w = rw.launch()
    start = w.time
    expect(w).to_satisfy(lambda world: world.time >= start + 0.5, "half a second has passed", timeout=2)
    assert w.time >= start + 0.5
    with pytest.raises(rw.ExpectationError, match="never"):
        expect(w).to_satisfy(lambda world: False, "never", timeout=0.1)
    w.close()
