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
