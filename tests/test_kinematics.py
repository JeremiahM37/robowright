import mujoco
import numpy as np
import pytest

from robowright import assets
from robowright.kinematics import Chain
from robowright.robot import DOWN, TCP_OFFSET


@pytest.fixture(scope="module")
def chain():
    return Chain(assets.so101_urdf(), "base_link", "gripper_link", TCP_OFFSET)


def test_urdf_fk_matches_mujoco_model(chain):
    m = mujoco.MjModel.from_xml_path(str(assets.so101_mjcf()))
    d = mujoco.MjData(m)
    bare = Chain(assets.so101_urdf(), "base_link", "gripper_link")
    rng = np.random.default_rng(0)
    gid = m.body("gripper").id
    for _ in range(100):
        q = rng.uniform(bare.lower, bare.upper)
        d.qpos[:5] = q
        mujoco.mj_kinematics(m, d)
        T = bare.fk(q)
        assert np.linalg.norm(T[:3, 3] - d.xpos[gid]) < 1e-5
        assert np.abs(T[:3, :3] - d.xmat[gid].reshape(3, 3)).max() < 1e-4


def test_ik_round_trip(chain):
    rng = np.random.default_rng(1)
    seed = np.array([0, -0.5, 0.5, 1.4, 0])
    hits = 0
    for _ in range(40):
        target = np.array([rng.uniform(0.15, 0.27), rng.uniform(-0.1, 0.1), rng.uniform(0.02, 0.07)])
        q, err = chain.ik(target, seed, DOWN, yaw=None)
        if err < 1e-3:
            hits += 1
            T = chain.fk(q)
            assert np.linalg.norm(T[:3, 3] - target) < 1e-3
            assert T[2, 2] > 0.99  # tool z axis up = fingers point down
            assert np.all(q >= chain.lower - 1e-9) and np.all(q <= chain.upper + 1e-9)
    assert hits >= 38


def test_unreachable_reports_error(chain):
    _, err = chain.ik((0.6, 0, 0.3), np.zeros(5), DOWN)
    assert err > 0.1


def test_chain_rejects_unknown_link():
    with pytest.raises(ValueError):
        Chain(assets.so101_urdf(), "base_link", "no_such_link")
