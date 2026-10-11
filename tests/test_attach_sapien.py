"""The SAPIEN adapter's logic, with a scene written for the test: no ManiSkill, no SAPIEN, no GPU.

The fake scene has the shape the adapter reads (actors and articulations by name, batched poses, a PhysX
contact list), so these checks cover the adapter's own arithmetic: batch handling, names and handles, the
impulse scaling of a force, the clock and the contact forces. The published environment is tested in
examples/environments/test_maniskill.py.
"""

import numpy as np
import pytest

import robowright as rw
from robowright.backends.attached_sapien import SapienEngine
from robowright.errors import CapabilityError


class _Tensorish:
    """Stands in for a torch tensor: the adapter calls detach(), cpu() and numpy() on one."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def clone(self):
        return _Tensorish(self.a.copy())

    def __setitem__(self, idx, value):
        self.a[idx] = value

    def __array__(self, dtype=None):
        return self.a if dtype is None else self.a.astype(dtype)


class _Actor:
    def __init__(self, name, pos, body="dynamic"):
        self.name = name
        self.px_body_type = body
        self.raw = np.array([[*pos, 1.0, 0.0, 0.0, 0.0]], dtype=float)
        self.lin = np.zeros((1, 3))
        self.ang = np.zeros((1, 3))
        self.forces = []

    @property
    def pose(self):
        raw = self.raw

        class _P:
            raw_pose = _Tensorish(raw)

        return _P

    @property
    def linear_velocity(self):
        return _Tensorish(self.lin)

    @property
    def angular_velocity(self):
        return _Tensorish(self.ang)

    def apply_force(self, force):
        self.forces.append(np.asarray(force, dtype=float))


class _Entity:
    def __init__(self, name):
        self.name = name


class _Body:
    def __init__(self, name):
        self.entity = _Entity(name)


class _Point:
    def __init__(self, impulse, normal):
        self.impulse = np.asarray(impulse, dtype=float)
        self.normal = np.asarray(normal, dtype=float)


class _Contact:
    def __init__(self, a, b, points):
        self.bodies = [_Body(a), _Body(b)]
        self.points = points


class _Px:
    def __init__(self):
        self.contacts = []

    def get_contacts(self):
        return self.contacts


class _Scene:
    def __init__(self, actors, articulations=None):
        self.actors = {a.name: a for a in actors}
        self.articulations = articulations or {}
        self.px = _Px()


class _Joint:
    def __init__(self, name):
        self.name = name


class _Link:
    def __init__(self, name, pos):
        self.name = name
        self._actor = _Actor(name, pos, body="dynamic")

    @property
    def pose(self):
        return self._actor.pose

    @property
    def linear_velocity(self):
        return self._actor.linear_velocity

    @property
    def angular_velocity(self):
        return self._actor.angular_velocity


class _Art:
    def __init__(self, links, joints, q):
        self.links = links
        self.active_joints = [_Joint(j) for j in joints]
        self._q = np.asarray(q, dtype=float)[None]

    @property
    def qpos(self):
        return _Tensorish(self._q)

    @property
    def qvel(self):
        return _Tensorish(np.zeros_like(self._q))

    def set_qpos(self, q):
        self._q = np.asarray(q.a if hasattr(q, "a") else q, dtype=float).copy()

    def set_qvel(self, v):
        pass


class _Core:
    """What the adapter reads from ``env.unwrapped``: rates, the batch size and the scene."""

    def __init__(self, scene, num_envs=1, sim_freq=100, control_freq=20):
        self.scene = scene
        self.num_envs = num_envs
        self.sim_freq = sim_freq
        self.control_freq = control_freq


def _scene():
    cube = _Actor("cube", [0.0, 0.0, 0.02])
    table = _Actor("table", [0.0, 0.0, -0.1], body="static")
    arm = _Art([_Link("hand", [0.4, 0.0, 0.5])], ["j1", "j2"], [0.1, -0.2])
    return _Scene([cube, table], {"arm": arm}), cube, arm


def _engine(core):
    return SapienEngine(type("Env", (), {"unwrapped": core})())


def test_batched_poses_are_read_as_one_environment():
    scene, cube, _ = _scene()
    eng = _engine(_Core(scene))
    pos, quat = eng.pose(eng.resolve("cube"))
    assert pos.shape == (3,) and quat.shape == (4,)
    np.testing.assert_allclose(pos, [0.0, 0.0, 0.02])
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])  # ManiSkill's wxyz, as robowright's


def test_a_batch_of_more_than_one_environment_is_refused_with_the_reason():
    scene, _, _ = _scene()
    with pytest.raises(CapabilityError, match="runs 2 in parallel"):
        _engine(_Core(scene, num_envs=2))


def test_names_resolve_to_actors_and_articulation_links_and_free_bodies_are_the_dynamic_actors():
    scene, _, _ = _scene()
    eng = _engine(_Core(scene))
    assert eng.resolve("cube") == ("actor", "cube")
    assert eng.resolve("hand") == ("link", "arm", "hand")
    assert eng.resolve("missing") is None
    assert eng.label(("link", "arm", "hand")) == "hand"
    assert eng.free_bodies == ["cube"]  # the static table is not free
    assert eng.joint_names == ["j1", "j2"]
    assert set(eng.names("body")) == {"cube", "table", "hand"}


def test_a_joint_is_read_by_name_from_the_articulations_qpos_order():
    scene, _, arm = _scene()
    eng = _engine(_Core(scene))
    assert eng.joint_value("j2") == pytest.approx(-0.2)
    eng.set_joint("j2", 0.5)
    assert eng.joint_value("j2") == pytest.approx(0.5)
    np.testing.assert_allclose(eng.qpos(), [0.1, 0.5])
    assert eng.joint_qadr("j2") == 1


def test_a_force_is_delivered_once_scaled_by_the_substeps_of_one_control_step():
    scene, cube, _ = _scene()
    eng = _engine(_Core(scene, sim_freq=100, control_freq=20))
    eng.apply_force("cube", [1.0, 0.0, -2.0])
    (applied,) = cube.forces
    np.testing.assert_allclose(applied, [5.0, 0.0, -10.0])  # 100 / 20 physics substeps per control step


def test_the_clock_counts_control_steps_and_restarts_at_reset():
    scene, _, _ = _scene()
    core = _Core(scene, control_freq=20)
    eng = _engine(core)
    assert eng.control_dt == pytest.approx(0.05)
    for _ in range(4):
        eng.note_step()
    assert eng.time == pytest.approx(0.2)
    eng.rebind(type("Env", (), {"unwrapped": core})())  # what a reset does
    assert eng.time == 0.0


def test_a_static_body_cannot_be_moved_and_a_free_one_is_refused_only_by_name():
    scene, _, _ = _scene()
    eng = _engine(_Core(scene))
    with pytest.raises(ValueError, match="not a free body"):
        eng.set_object_pose("table", [0.0, 0.0, 0.0])


def test_contact_forces_are_normal_impulses_over_the_physics_step_and_named_by_entity():
    scene, _, _ = _scene()
    scene.px.contacts = [
        _Contact("cube", "table", [_Point([0.0, 0.0, 0.01], [0.0, 0.0, 1.0]), _Point([0.0, 0.0, 0.01], [0.0, 0.0, 1.0])]),
        _Contact("hand", "cube", [_Point([0.0, 0.0, 0.002], [0.0, 0.0, 1.0])]),
        _Contact("cube", "cube", [_Point([0.0, 0.0, 9.0], [0.0, 0.0, 1.0])]),  # a body against itself is not a contact
    ]
    eng = _engine(_Core(scene, sim_freq=100))
    pairs = eng.contact_pairs()
    assert [(a, b) for a, b, _ in pairs] == [("cube", "table"), ("hand", "cube")]
    assert pairs[0][2] == pytest.approx(0.02 / 0.01)
    assert pairs[1][2] == pytest.approx(0.002 / 0.01)
    assert eng.contacts_of(eng.resolve("cube")) == ["hand", "table"]
    assert eng.contacts_of(eng.resolve("table")) == ["cube"]


def test_robowright_attaches_to_an_object_with_a_sapien_scene_and_refuses_one_without():
    scene, _, _ = _scene()
    core = _Core(scene)
    env = type("Env", (), {"unwrapped": core, "step": lambda self, *a: None, "reset": lambda self, *a, **k: None})()
    world = rw.attach(env, engine="sapien", name="unit_sapien")
    assert world.backend.engine == "sapien"
    assert world.body("cube").position.shape == (3,)
    world.close(failed=False)
    with pytest.raises(ValueError, match="cannot find a SAPIEN scene"):
        rw.attach(type("Bare", (), {"unwrapped": object()})(), engine="sapien", name="unit_sapien_bare")


def test_an_unknown_engine_name_is_refused_and_the_message_lists_sapien():
    with pytest.raises(ValueError, match="'mujoco', 'pybullet' or 'sapien'"):
        rw.attach(engine="nope", model=None, data=None, name="unit_bad_engine")


def test_a_contact_names_a_body_as_the_environment_does_not_as_its_scene_entity():
    """ManiSkill names an actor's SAPIEN entity ``scene-0_<name>``; a contact against the actor must still match its name."""
    scene, _, _ = _scene()
    scene.px.contacts = [_Contact("scene-0_cube", "scene-0_table", [_Point([0.0, 0.0, 0.01], [0.0, 0.0, 1.0])])]
    eng = _engine(_Core(scene, sim_freq=100))
    assert eng.contacts_of(eng.resolve("cube")) == ["table"]
    assert [(a, b) for a, b, _ in eng.contact_pairs()] == [("cube", "table")]
