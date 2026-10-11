"""The pieces a locomotion policy needs: a joint PD loop on torque motors, the control rate, and the
base's state in the base's own frame. No download: the robot is the built-in Go2, given a PD loop
with its own gains (the way a maker publishes them)."""

import mujoco
import numpy as np
import pytest

import robowright as rw
from robowright import robots

KP = np.array([40.0, 60.0, 30.0] * 4)  # per joint, in the Go2's joint order
KD = np.array([1.0, 2.0, 0.5] * 4)


@pytest.fixture(scope="module")
def go2_pd():
    """The Go2 with ``KP``/``KD`` as its published PD loop."""
    joints = robots.get("go2").arm_joints
    return robots.load(robots.get("go2").mjcf(), name="go2_pd_test", pd_gains=tuple(zip(KP, KD)), arm_joints=joints, base_body="base")


def _world(robot_name):
    return rw.launch(robot=robot_name, backend="mujoco", seed=0)


def test_pd_gains_become_the_servo_gains(go2_pd):
    w = _world(go2_pd.name)
    m = w.backend.model
    for i in range(len(go2_pd.arm_joints)):
        a = w.backend._act[i]
        assert m.actuator_gainprm[a, 0] == pytest.approx(KP[i])
        assert m.actuator_biasprm[a, 1] == pytest.approx(-KP[i])
        assert m.actuator_biasprm[a, 2] == pytest.approx(-KD[i])


def test_the_servo_is_the_published_pd_law(go2_pd):
    """Torque = kp (target - q) - kd qdot, at every physics step, clipped to the motor's range."""
    w = _world(go2_pd.name)
    b = w.backend
    rng = np.random.default_rng(1)
    b.data.qpos[b._qadr] += rng.uniform(-0.3, 0.3, len(b._qadr))
    b.data.qvel[b._dadr] = rng.uniform(-2.0, 2.0, len(b._dadr))
    target = rng.uniform(-0.5, 0.5, len(b._qadr))
    b.set_ctrl(target)
    mujoco.mj_forward(b.model, b.data)
    tau = KP * (target - b.qpos()) - KD * b.qvel()
    lim = b.model.actuator_forcelimited[b._act].astype(bool)
    lo, hi = b.model.actuator_forcerange[b._act].T
    expected = np.clip(tau, np.where(lim, lo, -np.inf), np.where(lim, hi, np.inf))
    assert np.allclose(b.data.actuator_force[b._act], expected, atol=1e-9)


def test_targets_are_held_for_each_control_step(go2_pd):
    """A published loop takes a new target per control step and holds it: no ramp across the step."""
    w = _world(go2_pd.name)
    b = w.backend
    b.set_ctrl(np.zeros(len(b._qadr)))
    b.set_ctrl(np.full(len(b._qadr), 0.2))
    assert not b._ramp.moving
    ramped = _world("go2")
    ramped.backend.set_ctrl(np.zeros(len(ramped.backend._qadr)))
    ramped.backend.set_ctrl(np.full(len(ramped.backend._qadr), 0.2))
    assert ramped.backend._ramp.moving  # the default servo path is unchanged


def test_policy_rate_is_the_control_step(go2_pd):
    """Physics at 0.002 s and control at 50 Hz: a policy call every 10 physics steps."""
    w = _world(go2_pd.name)
    assert w.backend.model.opt.timestep == pytest.approx(0.002)
    t0 = w.backend.time
    w.step(1)
    assert w.backend.time - t0 == pytest.approx(0.02)
    assert w.dt == pytest.approx(0.02)


def test_observation_is_in_the_base_frame(go2_pd):
    w = _world(go2_pd.name)
    r = w.robot
    yaw, roll = 0.7, 0.2
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    qr = np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
    quat = _mul(q, qr)
    w.backend.set_base_pose(tuple(r.model.stand_height * np.array([0, 0, 1.0])), tuple(quat))
    w.backend.data.qvel[w.backend._base_v + 3 : w.backend._base_v + 6] = [0.1, -0.2, 0.3]  # body-frame spin
    obs = r.observe()
    R = _rotation(quat)
    assert np.allclose(obs["imu_quat"], quat)
    assert np.allclose(obs["base_ang_vel_body"], [0.1, -0.2, 0.3], atol=1e-9)
    assert np.allclose(obs["projected_gravity"], R.T @ np.array([0.0, 0.0, -1.0]), atol=1e-9)
    assert np.allclose(obs["base_ang_vel"], R @ np.array([0.1, -0.2, 0.3]), atol=1e-9)


def test_imu_orientation_is_current(go2_pd):
    """MuJoCo's body pose lags the joints by a step; the IMU view must not."""
    w = _world(go2_pd.name)
    w.backend.set_ctrl(np.zeros(len(w.backend._qadr)) + 0.3)
    w.backend.step()
    assert np.allclose(w.backend.current_base_quat(), w.backend.data.qpos[w.backend._base_q + 3 : w.backend._base_q + 7])


def _mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _rotation(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
