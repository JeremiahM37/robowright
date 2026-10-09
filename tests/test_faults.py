import numpy as np

import robowright as rw


def test_joint_noise_only_affects_readings(quiet_world):
    w = quiet_world()
    w.faults.joint_noise(0.05)
    true = w.robot.true_qpos()
    seen = w.robot.qpos()
    assert np.abs(seen - true).max() > 0.001
    assert np.allclose(w.backend.qpos(), true)


def test_action_delay_lags_commands(quiet_world):
    w = quiet_world()
    w.faults.action_delay(steps=5)
    before = w.backend.ctrl()[0]
    w.robot._target[0] = 0.5
    w.step(3)
    assert w.backend.ctrl()[0] == before
    w.step(5)
    assert abs(w.backend.ctrl()[0] - 0.5) < 1e-6


def test_push_moves_object(quiet_world):
    w = quiet_world()
    p0 = w.scene["cube"].position.copy()
    w.faults.push("cube", (2.0, 0, 0), duration=0.1)
    w.wait(0.5)
    assert w.scene["cube"].position[0] - p0[0] > 0.01


def test_jitter_is_seeded():
    def start(seed):
        w = rw.launch(seed=seed, settings=rw.Settings(trace="off"))
        p = w.faults.jitter("cube", 0.02, yaw_std=0.3)
        w.backend.close()
        return p

    assert np.allclose(start(4), start(4))
    assert not np.allclose(start(4), start(5))


def test_camera_dropout(quiet_world):
    w = quiet_world()
    w.faults.camera_dropout(p=1.0)
    img = w.robot.observe(cameras=("front",))["images"]["front"]
    assert img.max() == 0


def _launch(seed, *objects):
    return rw.launch(rw.tabletop(*objects) if objects else None, seed=seed, settings=rw.Settings(trace="off"))


def test_randomize_scene_moves_free_objects_within_bounds_and_leaves_bins():
    def start(seed):
        w = _launch(seed)
        cube0, bin0 = w.scene["cube"].position.copy(), w.scene["bin"].position.copy()
        w.faults.randomize_scene()
        w.wait(0.2)
        out = w.scene["cube"].position.copy(), cube0, w.scene["bin"].position.copy(), bin0
        w.backend.close()
        return out

    seen = []
    for seed in range(8):
        cube, cube0, bin_, bin0 = start(seed)
        assert np.all(np.abs(cube[:2] - cube0[:2]) <= rw.faults.SCENE_XY + 1e-6)
        assert abs(cube[2] - cube0[2]) < 1e-3  # still resting on the table
        assert np.allclose(bin_, bin0)
        seen.append(cube[:2])
    assert len({tuple(np.round(p, 5)) for p in seen}) == 8  # a different scene for every seed
    assert np.allclose(start(3)[0], start(3)[0])  # and the same one for the same seed


def test_randomize_scene_leaves_stacked_objects_alone():
    w = _launch(
        0,
        rw.ObjectSpec("base", "box", (0.02, 0.02, 0.02), (0.22, -0.06, None)),
        rw.ObjectSpec("top", "box", (0.0125, 0.0125, 0.0125), (0.22, -0.06, 0.0525), color="green"),
        rw.ObjectSpec("loose", "box", (0.0125, 0.0125, 0.0125), (0.15, 0.0, None), color="yellow"),
    )
    assert set(w.faults.randomize_scene()) == {"loose"}
    w.backend.close()


def test_jitter_after_randomize_replaces_it():
    """A jitter's spread is the one asked for, not the default randomization's added to it."""
    w = _launch(2)
    w.faults.randomize_scene()
    p = w.faults.jitter("cube", xy_std=0.0)
    assert np.allclose(p[:2], (0.22, -0.06))
    w.backend.close()


def test_used_randomness():
    w = _launch(0)
    assert not w.faults.used_randomness()
    w.faults.push("cube", (1.0, 0, 0))  # deterministic
    assert not w.faults.used_randomness()
    w.faults.jitter("cube", 0.01)
    assert w.faults.used_randomness()
    w.backend.close()
