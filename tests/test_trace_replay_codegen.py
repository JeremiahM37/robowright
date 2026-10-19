import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import Trace, condition, expect
from robowright.codegen import generate
from robowright.policies import ScriptedPickPlace
from robowright.replay import replay
from robowright.scene import default_scene


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
    assert tr.frame_names == []  # no rendering while the test runs: the viewer draws frames from the state
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
    w = rw.launch(name="nogl", settings=rw.Settings(trace="on", trace_dir=str(tmp_path), trace_cameras=["front"]))

    def broken(*a, **k):
        raise RuntimeError("no EGL")

    monkeypatch.setattr(w.backend, "render", broken)
    w.robot.reset_to()
    with pytest.warns(UserWarning, match="camera frames disabled"):
        w.robot.arm.home()
    tr = Trace(w.close())
    assert tr.frame_names == [] and len(tr) > 1


@pytest.mark.parametrize("robot", ["panda", "go2"])
def test_replay_and_codegen_beyond_the_so101(tmp_path, robot):
    """A 7-DoF arm and a floating-base robot: replay is exact and codegen reproduces the run step for step."""
    from robowright import expect

    w = rw.launch(robot=robot, seed=3, name="orig", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    w.robot.reset_to()
    if robot == "go2":
        w.faults.push("robot", force=(0, 30.0, 0), duration=0.1)
        w.wait(0.5)
        w.robot.crouch(0.3)
        w.robot.stand()
        expect(w.robot.base).to_be_upright(tol_deg=15)
    else:
        w.faults.action_delay(steps=2)
        w.robot.pick(w.scene["cube"])
        w.robot.place(on=w.scene["bin"])
    path = w.close()
    assert replay(path).first_divergent_step is None
    code = generate(path, test_name="test_regen")
    ns = {}
    exec(code, ns)
    original_launch = rw.launch
    try:
        rw.launch = lambda *a, **k: original_launch(*a, **{**k, "settings": rw.Settings(trace="on", trace_dir=str(tmp_path / "regen"))})
        ns["rw"] = rw
        ns["test_regen"]()
    finally:
        rw.launch = original_launch
    a, b = Trace(path).arrays, Trace(tmp_path / "regen" / "test_regen.zip").arrays
    assert np.array_equal(a["qpos"], b["qpos"])


def test_codegen_keeps_settings_and_expect_timeouts(tmp_path):
    """A regenerated test moves at the original speeds and waits as long in each check."""
    from robowright.errors import ExpectationError

    s = rw.Settings(trace="on", trace_dir=str(tmp_path), max_joint_speed=1.0)
    with rw.launch(robot="so101", name="slow", settings=s) as w:
        w.robot.reset_to()
        w.robot.arm.move_to((0.22, 0.0, 0.08))
        with pytest.raises(ExpectationError):
            expect(w.scene["cube"], timeout=0.3).to_be_inside(w.scene["bin"])
    code = generate(w.trace_path)
    assert "settings=rw.Settings(max_joint_speed=1.0" in code  # (and fidelity="adjusted", when it ran so)
    assert "timeout=0.3" in code
    assert "robot.reset_to()" in code


def test_frames_captured_during_the_run_on_request(tmp_path):
    w = rw.launch(name="live", settings=rw.Settings(trace="on", trace_dir=str(tmp_path), trace_cameras=["front"]))
    w.robot.reset_to()
    w.robot.arm.home()
    tr = Trace(w.close())
    assert any(n.startswith("frames/front/") for n in tr.frame_names)


def test_viewer_draws_frames_from_the_recorded_state(tmp_path):
    from robowright.viewer import build_html

    path, _ = _run(tmp_path)
    try:
        html = build_html(path)
    except Exception as e:  # pragma: no cover - no GL on this machine
        pytest.skip(f"rendering unavailable: {e}")
    assert html.count("data:image/jpeg;base64,") > 10


def test_viewer_without_gl_keeps_the_telemetry(tmp_path, monkeypatch):
    from robowright.viewer import build_html

    path, _ = _run(tmp_path, faults=False)

    def broken(*a, **k):
        raise RuntimeError("no EGL")

    # The module the viewer's own import gets (sys.modules): a pytester run earlier in the same
    # process can leave the package attribute robowright.render pointing at a stale copy.
    import importlib

    monkeypatch.setattr(importlib.import_module("robowright.render"), "jpeg_frames", broken)
    with pytest.warns(UserWarning, match="without camera frames"):
        html = build_html(path)
    assert '"frames":{}' in html and '"qpos"' in html


@pytest.mark.parametrize("robot", ["so101", "go2"])
def test_render_a_trace_to_video(tmp_path, robot):
    """Any trace redraws from its state alone: arms from joints and object poses, legged robots also from the base pose."""
    from robowright.render import Renderer, render_video

    w = rw.launch(robot=robot, name=robot, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    w.robot.reset_to()
    if robot == "go2":
        w.faults.push("robot", force=(0, 40.0, 0), duration=0.1)
        w.wait(0.4)
    else:
        w.robot.arm.move_to((0.22, 0.05, 0.08))
    tr = Trace(w.close())
    if robot == "go2":
        assert tr.arrays["base"].shape == (len(tr), 7)
    try:
        r = Renderer(tr)
        first, last = r.frame(0, "front", (160, 120)), r.frame(len(tr) - 1, "front", (160, 120))
        r.close()
    except Exception as e:  # pragma: no cover - no GL on this machine
        pytest.skip(f"rendering unavailable: {e}")
    assert first.shape == (120, 160, 3) and np.abs(first.astype(int) - last).mean() > 0.1  # it moved
    gif = render_video(tr, tmp_path / "run.gif", size=(160, 120), fps=10)
    assert gif.read_bytes()[:6] == b"GIF89a"


def _toy_arm(tmp_path):
    """A five-joint arm with two slide fingers and one gripper actuator, from its model file alone
    (the same made-up robot test_detect.py picks and places with)."""
    link = 'type="capsule" size="0.02" mass="0.3"'
    xml = f"""
<mujoco model="toy">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default><joint damping="2" armature="0.05" range="-3 3"/><position kp="300" kv="20"/></default>
  <worldbody>
    <body name="base"><geom type="cylinder" size="0.05 0.02" pos="0 0 0.02"/>
      <body name="l1" pos="0 0 0.04"><joint name="j1" axis="0 0 1"/><geom {link} fromto="0 0 0 0 0 0.1"/>
        <body name="l2" pos="0 0 0.1"><joint name="j2" axis="0 1 0"/><geom {link} fromto="0 0 0 0.2 0 0"/>
          <body name="l3" pos="0.2 0 0"><joint name="j3" axis="0 1 0"/><geom {link} fromto="0 0 0 0.18 0 0"/>
            <body name="l4" pos="0.18 0 0"><joint name="j4" axis="0 1 0"/><geom {link} fromto="0 0 0 0.04 0 0"/>
              <body name="hand" pos="0.04 0 0"><joint name="j5" axis="1 0 0"/>
                <geom type="box" size="0.01 0.04 0.015" mass="0.2"/>
                <body name="left_finger" pos="0.035 0.01 0"><joint name="fl" type="slide" axis="0 1 0" range="0 0.03"/>
                  <geom type="box" size="0.025 0.004 0.01" pos="0 0 0" mass="0.02" friction="1.5"/></body>
                <body name="right_finger" pos="0.035 -0.01 0"><joint name="fr" type="slide" axis="0 -1 0" range="0 0.03"/>
                  <geom type="box" size="0.025 0.004 0.01" pos="0 0 0" mass="0.02" friction="1.5"/></body>
              </body></body></body></body></body></body>
  </worldbody>
  <equality><joint joint1="fr" joint2="fl"/></equality>
  <actuator>
    <position joint="j1" ctrlrange="-3 3"/><position joint="j2" ctrlrange="-3 3"/><position joint="j3" ctrlrange="-3 3"/>
    <position joint="j4" ctrlrange="-3 3"/><position joint="j5" ctrlrange="-3 3"/>
    <position name="grip" joint="fl" kp="200" ctrlrange="0 0.03"/>
  </actuator>
</mujoco>"""
    path = tmp_path / "toy_arm.xml"
    path.write_text(xml)
    return path


def _run_file_robot(tmp_path):
    """A pick and place with a robot loaded from a file under a name only this test uses (a backend kept
    from an earlier test with the same robot name would be reused). Returns the trace and the robot's name."""
    import warnings

    from robowright import robots

    name = f"toy_{tmp_path.name}"
    path = _toy_arm(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MuJoCo's notes on attaching a gripper
        robots.load(path, name=name, arm_joints=("j1", "j2", "j3", "j4", "j5"), timestep=0.002)
    with rw.launch(scene=default_scene(name), settings=rw.Settings(trace="on", trace_dir=str(tmp_path / "traces"), trace_cameras=[])) as w:
        w.robot.pick(w.scene["cube"])
        w.robot.place(on=w.scene["bin"])
        expect(w.scene["cube"]).to_be_inside(w.scene["bin"])
    return w.trace_path, name


def test_a_robot_from_a_model_file_is_recorded_with_how_it_was_loaded(tmp_path):
    path, name = _run_file_robot(tmp_path)
    meta = Trace(path).meta
    assert meta["robot_file"] == {
        "file": str(tmp_path / "toy_arm.xml"),
        "name": name,
        "overrides": {"arm_joints": ["j1", "j2", "j3", "j4", "j5"], "timestep": 0.002},
    }
    assert meta["backend_options"] == {}


def test_codegen_loads_the_file_robot_and_the_generated_test_runs_and_replays_in_a_fresh_process(tmp_path):
    path, name = _run_file_robot(tmp_path)
    code = generate(path, test_name="test_toy_arm")
    load_line = f"ROBOT = rw.robots.load({str(tmp_path / 'toy_arm.xml')!r}, name={name!r}, "
    assert load_line + "arm_joints=('j1', 'j2', 'j3', 'j4', 'j5'), timestep=0.002)" in code
    assert code.index("ROBOT = ") < code.index("SCENE = ")
    test = tmp_path / "test_toy_arm_regression.py"
    test.write_text(code)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH"), str(Path(rw.__file__).parents[1])]))}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "-q", "-p", "no:cacheprovider"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    replay_code = (
        "import sys\nfrom robowright.replay import replay\n"
        f"r = replay({str(path)!r})\nprint(r.summary())\nsys.exit(0 if r.identical else 1)\n"
    )
    rep = subprocess.run([sys.executable, "-c", replay_code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert rep.returncode == 0, rep.stdout + rep.stderr[-2000:]


def test_a_trace_whose_model_file_is_gone_says_which_file(tmp_path, monkeypatch):
    from robowright import robots

    path, name = _run_file_robot(tmp_path)
    monkeypatch.delitem(robots.REGISTRY, name, raising=False)  # as in a fresh process
    (tmp_path / "toy_arm.xml").unlink()
    with pytest.raises(FileNotFoundError, match="toy_arm.xml"):  # the path, so the user can put it back
        Trace(path).scene()


def test_backend_options_are_recorded_and_passed_to_the_launch_call(tmp_path):
    w = rw.launch(backend="mujoco", seed=1, name="opts", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    w.backend_options = {"frame": "base_link", "use_sim_time": True, "arm_controller": "arm_controller_with_a_long_name"}
    path = w.close(trace_path=tmp_path / "opts.zip")
    assert Trace(path).meta["backend_options"]["frame"] == "base_link"
    code = generate(path, test_name="test_opts")
    assert "    with rw.launch(\n        SCENE,\n" in code  # too long for one line
    assert "        frame='base_link'," in code and "        use_sim_time=True," in code
    assert "frame=" not in generate(path, test_name="test_opts", backend="drake")  # another engine gets none of them


def test_the_text_of_a_trace_lists_its_faults(tmp_path):
    """A trace with a fault once could not be printed at all (the fault records were joined as strings)."""
    w = rw.launch(robot="so101", settings=rw.Settings(trace="on"))
    w.faults.push("cube", (0.0, 1.5, 0.0), at=w.time + 0.05)
    w.step(10)
    path = w.close(failed=True, trace_path=tmp_path / "pushed.zip")
    text = Trace(path).summary()
    assert "faults: Push(object=cube, force=" in text and "duration=0.1" in text


def _failing_pick(tmp_path):
    w = rw.launch(backend="mujoco", seed=0, name="lip", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    w.robot.reset_to()
    w.robot.pick(w.scene["cube"], lift=0.05)
    with pytest.raises(rw.ExpectationError):
        expect(w.scene["cube"]).to_be_above(0.12, timeout=0.5)
    return w.close(failed=True)


def _line_of(text: str) -> int:
    return next(i for i, line in enumerate(Path(__file__).read_text().splitlines(), 1) if text in line)


def test_each_event_keeps_the_line_of_the_test_that_made_it(tmp_path):
    tr = Trace(_failing_pick(tmp_path))
    here = str(Path(__file__).resolve())
    at = {e["name"]: e.get("loc") for e in tr.events}
    assert at["robot.pick"] == [here, _line_of('w.robot.pick(w.scene["cube"], lift=0.05)')]
    assert at["expect(cube).to_be_above"] == [here, _line_of('expect(w.scene["cube"]).to_be_above(0.12, timeout=0.5)')]
    assert tr.meta["sources"][here] == Path(__file__).read_text()  # the viewer's Source tab shows it
    assert list(tr.meta["sources"]) == [here]  # robowright's own files and pytest's are not kept


def test_the_text_of_a_failed_trace_says_where_the_test_failed(tmp_path):
    text = Trace(_failing_pick(tmp_path)).summary()
    line = _line_of('expect(w.scene["cube"]).to_be_above(0.12, timeout=0.5)')
    assert f'test_trace_replay_codegen.py:{line}: expect(w.scene["cube"]).to_be_above(0.12, timeout=0.5)' in text


def test_the_viewer_shows_the_source_at_the_selected_action(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    from robowright.viewer import build_html

    page_file = tmp_path / "view.html"
    page_file.write_text(build_html(_failing_pick(tmp_path)))
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(page_file.as_uri())
        page.click('button[data-tab="source"]')
        page.click("#events li.failed")
        assert "to_be_above(0.12, timeout=0.5)" in page.inner_text(".src .at.failed")
        page.click("#events li:has-text('robot.pick')")  # the tab stays on Source, at the new action's line
        assert "w.robot.pick(" in page.inner_text(".src .at")
        browser.close()
    assert not errors
