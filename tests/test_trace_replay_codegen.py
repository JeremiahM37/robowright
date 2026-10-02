import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

import robowright as rw
from robowright import Trace, condition, expect
from robowright.codegen import generate
from robowright.policies import ScriptedPickPlace
from robowright.replay import replay


def _run(tmp_path, name="run", backend="mujoco", fail=False, faults=True):
    w = rw.launch(backend=backend, seed=3, name=name, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    w.robot.reset_to()
    if faults:
        w.faults.jitter("cube", 0.015, yaw_std=0.3)
        w.faults.action_delay(2)
        w.faults.joint_noise(0.005)
    w.robot.pick(w.scene["cube"])
    if faults:
        w.faults.push("cube", (0.0, 0.5, 0.0), duration=0.05)
        w.faults.weak_joint("wrist_roll", 0.8)
    w.robot.place(on=w.scene["bin"] if not fail else (0.26, -0.1, 0.0))
    try:
        expect(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=0.5)
    except rw.ExpectationError:
        pass
    return w.close(), w


def test_trace_contents(tmp_path):
    path, w = _run(tmp_path)
    tr = Trace(path)
    assert len(tr) == w.step_count + 1
    kinds = {e["type"] for e in tr.events}
    assert {"edit", "fault", "action", "expect"} <= kinds
    assert tr.arrays["qpos"].shape == (len(tr), 6)
    assert tr.arrays["obj_pos"].shape == (len(tr), 2, 3)
    assert any(n.startswith("frames/front/") for n in tr.frame_names)
    assert tr.state0 is not None
    assert tr.meta["status"] == "passed"


def test_mujoco_replay_is_bit_identical_with_faults(tmp_path):
    path, _ = _run(tmp_path)
    r = replay(path)
    assert r.identical, r.summary()


def test_pybullet_replay_is_deterministic(tmp_path):
    pytest.importorskip("pybullet")
    path, _ = _run(tmp_path, backend="pybullet")
    r = replay(path)
    assert r.max_qpos_error < 1e-9 and r.max_object_error < 1e-9, r.summary()


def test_replay_detects_divergence(tmp_path):
    path, _ = _run(tmp_path, faults=False)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.namelist():
            data = src.read(item)
            if item == "steps.npz":
                import io

                arr = dict(np.load(io.BytesIO(data)))
                arr["ctrl"][60:, 0] += 0.2
                buf = io.BytesIO()
                np.savez(buf, **arr)
                data = buf.getvalue()
            dst.writestr(item, data)
    r = replay(tampered)
    assert r.first_divergent_step is not None and 55 <= r.first_divergent_step <= 70


def test_codegen_reproduces_a_passing_run_exactly(tmp_path):
    path, w = _run(tmp_path, name="orig")
    code = generate(path, test_name="test_regen")
    compile(code, "regen.py", "exec")
    ns = {}
    exec(code, ns)
    rw_settings = rw.Settings(trace="on", trace_dir=str(tmp_path / "regen"))
    original_launch = rw.launch
    try:
        rw.launch = lambda *a, **k: original_launch(*a, **{**k, "settings": rw_settings})
        ns["rw"] = rw
        ns["test_regen"]()
    finally:
        rw.launch = original_launch
    a = Trace(path).arrays
    b = Trace(tmp_path / "regen" / "test_regen.zip").arrays
    assert a["qpos"].shape == b["qpos"].shape
    assert np.array_equal(a["obj_pos"][-1], b["obj_pos"][-1])


def test_codegen_of_failure_stops_at_failure(tmp_path):
    path, _ = _run(tmp_path, name="tests/test_x.py::test_bad[mujoco]", fail=True, faults=False)
    code = generate(path)
    assert "def test_bad_mujoco_regression():" in code
    assert code.rstrip().endswith("timeout=0.5)")
    assert "It failed at" in code


def test_codegen_recreates_policies_and_conditions(tmp_path):
    w = rw.launch(seed=1, name="pol", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    w.robot.reset_to()
    w.robot.run_policy(
        ScriptedPickPlace(chunk=8), until=condition(w.scene["cube"], "to_be_inside", w.scene["bin"]), timeout=12, privileged=True
    )
    code = generate(w.close())
    assert "from robowright.policies import ScriptedPickPlace" in code
    assert "policy = ScriptedPickPlace(object='cube', target='bin', chunk=8" in code
    assert "until=rw.condition(scene['cube'], 'to_be_inside', scene['bin'])" in code
    compile(code, "x.py", "exec")


def test_cli_round_trip(tmp_path):
    path, _ = _run(tmp_path, faults=False)
    cli = [sys.executable, "-m", "robowright.cli"]
    out = subprocess.run([*cli, "replay", str(path)], capture_output=True, text=True, check=True)
    assert "bit-identical" in out.stdout
    html = tmp_path / "t.html"
    subprocess.run([*cli, "show-trace", str(path), "-o", str(html), "--no-open"], check=True, capture_output=True)
    text = html.read_text()
    assert "data:image/jpeg;base64," in text and text.count("</script>") == 1
    data = json.loads(text.split("const D = ", 1)[1].split(";\nconst M", 1)[0])
    assert data["meta"]["name"] == "run"
    gen = tmp_path / "test_gen.py"
    subprocess.run([*cli, "codegen", str(path), "-o", str(gen)], check=True, capture_output=True)
    assert gen.read_text().startswith('"""Regression test generated')


def test_tracing_survives_missing_gl(tmp_path, monkeypatch):
    w = rw.launch(name="nogl", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))

    def broken(*a, **k):
        raise RuntimeError("no EGL")

    monkeypatch.setattr(w.backend, "render", broken)
    w.robot.reset_to()
    with pytest.warns(UserWarning, match="camera frames disabled"):
        w.robot.arm.home()
    tr = Trace(w.close())
    assert tr.frame_names == [] and len(tr) > 1
