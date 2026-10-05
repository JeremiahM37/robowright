"""The robots robowright knows how to drive.

Every arm here runs the same tests unchanged: the task scene stays put and
each robot is mounted where its top-down workspace covers it. Models come
from MuJoCo Menagerie (fetched on first use, see :mod:`.menagerie`) except
the SO-101, which ships with robowright so the default needs no download.
"""

from __future__ import annotations

from .. import assets
from . import menagerie
from .model import PREFIX, Attachment, Derived, Kinematics, RobotModel, body_labels

__all__ = ["PREFIX", "Attachment", "Derived", "Kinematics", "RobotModel", "body_labels", "get", "load", "names", "register", "REGISTRY"]

REGISTRY: dict[str, RobotModel] = {}


def register(model: RobotModel) -> RobotModel:
    REGISTRY[model.name] = model
    return model


def get(name: str | RobotModel) -> RobotModel:
    """A robot by name, or from a model file (``path/to/robot.xml``), detected on first use."""
    if isinstance(name, RobotModel):
        return name
    try:
        return REGISTRY[name]
    except KeyError:
        pass
    if is_file(name):
        from .detect import load

        return load(name)
    raise ValueError(f"unknown robot {name!r}; available: {', '.join(sorted(REGISTRY))}, or the path of a robot's MJCF file")


def load(path, name: str | None = None, **overrides) -> RobotModel:
    """Any robot from its model file (MJCF): what it is worked out from the model itself, and
    anything given as a keyword (a :class:`RobotModel` field) overrides that. See :mod:`.detect`."""
    from .detect import load as _load

    return _load(path, name, **overrides)


def is_file(name: str) -> bool:
    """Whether ``name`` names a model file rather than a registered robot."""
    return name.lower().endswith((".xml", ".mjcf", ".urdf")) or "/" in name or "\\" in name


def names(family: str | None = None) -> list[str]:
    """The built-in robots (and any registered in code); robots loaded from a file are not listed."""
    return [n for n, m in REGISTRY.items() if (family is None or m.family == family) and "file" not in m.extra]


def _m(rel):
    return lambda: menagerie.path(rel)


ROBOTIQ_2F85 = Attachment(_m("robotiq_2f85/2f85.xml"), "attachment_site")
_2F85 = dict(
    attach=ROBOTIQ_2F85,
    hand="gripper/base",
    left_finger=("gripper/left_follower",),
    right_finger=("gripper/right_follower",),
    gripper_actuator="gripper/fingers_actuator",
    gripper_open=0.0,
    gripper_closed=255.0,
)

register(
    RobotModel(
        "so101",
        "SO-101",
        assets.so101_mjcf,
        ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
        hand="gripper",
        left_finger=("gripper",),
        right_finger=("moving_jaw_so101_v1",),
        gripper_actuator="gripper",
        gripper_open=1.2,
        gripper_closed=-0.17,
        home=(0.2, 0.0, 0.08),
        seed=(0.0, -1.0, 1.0, 1.2, 0.0),
        # One jaw is fixed, so the grasp centre is not midway between the closed
        # jaws; this is the hand-tuned point between the jaw pads.
        tcp=(0.0054, 0.0, -0.09),
        tcp_inset=0.013,
        timestep=0.005,
        maker="TheRobotStudio / Hugging Face",
        tags=("lerobot", "low-cost", "5dof"),
        source="MuJoCo Menagerie (bundled)",
    )
)
register(
    RobotModel(
        "panda",
        "Franka Emika Panda",
        _m("franka_emika_panda/panda.xml"),
        tuple(f"joint{i}" for i in range(1, 8)),
        hand="hand",
        left_finger=("left_finger",),
        right_finger=("right_finger",),
        gripper_actuator="actuator8",
        gripper_open=255.0,
        gripper_closed=0.0,
        base_pos=(-0.3, 0.0, 0.0),
        home=(0.2, 0.0, 0.2),
        maker="Franka Robotics",
        license="Apache-2.0",
        tags=("research", "7dof"),
        grip_force=70.0,  # Franka Hand continuous grasping force (140 N peak); the model squeezes 0.6 N
        grip_force_source="https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf",
    )
)
register(
    RobotModel(
        "ur5e",
        "Universal Robots UR5e + Robotiq 2F-85",
        _m("universal_robots_ur5e/ur5e.xml"),
        ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        base_pos=(-0.3, 0.0, 0.0),
        home=(0.2, 0.0, 0.2),
        maker="Universal Robots",
        license="BSD (arm, gripper)",
        tags=("industrial", "cobot", "6dof"),
        **_2F85,
    )
)
register(
    RobotModel(
        "ur10e",
        "Universal Robots UR10e + Robotiq 2F-85",
        _m("universal_robots_ur10e/ur10e.xml"),
        ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        base_pos=(-0.45, 0.0, 0.0),
        home=(0.2, 0.0, 0.2),
        maker="Universal Robots",
        license="BSD (arm, gripper)",
        tags=("industrial", "cobot", "6dof"),
        **_2F85,
    )
)
register(
    RobotModel(
        "gen3",
        "Kinova Gen3 + Robotiq 2F-85",
        _m("kinova_gen3/gen3.xml"),
        tuple(f"joint_{i}" for i in range(1, 8)),
        base_pos=(-0.3, 0.0, 0.0),
        home=(0.2, 0.0, 0.2),
        maker="Kinova",
        license="BSD (arm, gripper)",
        tags=("research", "7dof"),
        **{**_2F85, "attach": Attachment(_m("robotiq_2f85/2f85.xml"), "pinch_site")},
    )
)
register(
    RobotModel(
        "iiwa14",
        "KUKA LBR iiwa 14 + Robotiq 2F-85",
        _m("kuka_iiwa_14/iiwa14.xml"),
        tuple(f"joint{i}" for i in range(1, 8)),
        base_pos=(-0.4, 0.0, 0.0),
        home=(0.2, 0.0, 0.2),
        maker="KUKA",
        license="BSD-3-Clause (arm), BSD (gripper)",
        tags=("industrial", "7dof"),
        **_2F85,
    )
)
register(
    RobotModel(
        "xarm7",
        "UFACTORY xArm 7",
        _m("ufactory_xarm7/xarm7.xml"),
        tuple(f"joint{i}" for i in range(1, 8)),
        hand="xarm_gripper_base_link",
        left_finger=("left_finger",),
        right_finger=("right_finger",),
        gripper_actuator="gripper",
        gripper_open=0.0,
        gripper_closed=255.0,
        base_pos=(-0.25, 0.0, 0.0),
        home=(0.2, 0.0, 0.2),
        maker="UFACTORY",
        license="BSD",
        tags=("cobot", "7dof"),
        grip_force=30.0,  # xArm Gripper (G1) maximum; the model squeezes ~140 N
        grip_force_source="https://docs.xarm.ufactory.cc/8.technical_specifications.html",
    )
)
register(
    RobotModel(
        "vx300s",
        "Trossen ViperX 300 S (ALOHA)",
        _m("trossen_vx300s/vx300s.xml"),
        ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        hand="gripper_link",
        left_finger=("left_finger_link",),
        right_finger=("right_finger_link",),
        gripper_actuator="gripper",
        gripper_open=0.057,
        gripper_closed=0.021,
        base_pos=(-0.1, 0.0, 0.0),
        home=(0.2, 0.0, 0.15),
        maker="Trossen Robotics",
        license="BSD",
        tags=("aloha", "6dof"),
    )
)
register(
    RobotModel(
        "wx250s",
        "Trossen WidowX 250 S (Bridge)",
        _m("trossen_wx250s/wx250s.xml"),
        ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        hand="wx250s/gripper_link",
        left_finger=("wx250s/left_finger_link",),
        right_finger=("wx250s/right_finger_link",),
        gripper_actuator="gripper",
        gripper_open=0.037,
        gripper_closed=0.015,
        base_pos=(-0.05, 0.0, 0.0),
        home=(0.2, 0.0, 0.15),
        maker="Trossen Robotics",
        license="BSD",
        tags=("bridge", "open-x", "6dof"),
    )
)
register(
    RobotModel(
        "piper",
        "AgileX PiPER",
        _m("agilex_piper/piper.xml"),
        tuple(f"joint{i}" for i in range(1, 7)),
        hand="link6",
        left_finger=("link7",),
        right_finger=("link8",),
        gripper_actuator="gripper",
        gripper_open=0.035,
        gripper_closed=0.0,
        base_pos=(0.0, 0.0, 0.0),
        home=(0.2, 0.0, 0.12),
        maker="AgileX Robotics",
        license="MIT",
        tags=("low-cost", "6dof"),
        grip_force=40.0,  # PiPER gripper rated force (50 N peak); the model squeezes 0.35 N
        grip_force_source="https://static.generation-robots.com/media/agilex-piper-user-manual.pdf",
    )
)
register(
    RobotModel(
        "yam",
        "I2RT YAM",
        _m("i2rt_yam/yam.xml"),
        tuple(f"joint{i}" for i in range(1, 7)),
        hand="link_6",
        left_finger=("link_left_finger",),
        right_finger=("link_right_finger",),
        gripper_actuator="gripper",
        gripper_open=0.041,
        gripper_closed=0.0,
        base_pos=(-0.05, 0.0, 0.0),
        home=(0.2, 0.0, 0.15),
        maker="I2RT",
        license="MIT",
        tags=("low-cost", "6dof"),
    )
)
register(
    RobotModel(
        "arx_l5",
        "ARX L5",
        _m("arx_l5/arx_l5.xml"),
        tuple(f"joint{i}" for i in range(1, 7)),
        hand="link6",
        left_finger=("link7",),
        right_finger=("link8",),
        gripper_actuator="gripper",
        gripper_open=0.044,
        gripper_closed=0.0,
        base_pos=(-0.05, 0.0, 0.0),
        home=(0.2, 0.0, 0.15),
        maker="ARX",
        license="BSD-3-Clause",
        tags=("low-cost", "6dof"),
    )
)
register(
    RobotModel(
        "sawyer",
        "Rethink Sawyer + Robotiq 2F-85",
        _m("rethink_robotics_sawyer/sawyer.xml"),
        tuple(f"right_j{i}" for i in range(7)),
        base_pos=(-0.35, 0.0, 0.0),
        home=(0.2, 0.0, 0.2),
        maker="Rethink Robotics",
        license="Apache-2.0 (arm), BSD (gripper)",
        tags=("research", "7dof"),
        **_2F85,
    )
)


# Legged robots -----------------------------------------------------------------
def _legs(*prefixes, names=("hip_joint", "thigh_joint", "calf_joint")):
    return tuple(f"{p}_{n}" for p in prefixes for n in names)


_LEGGED = dict(family="legged", base_pos=(0.0, 0.0, 0.0))
register(
    RobotModel(
        "go2",
        "Unitree Go2",
        _m("unitree_go2/go2.xml"),
        _legs("FL", "FR", "RL", "RR"),
        base_body="base",
        servo=(60.0, 3.0),
        maker="Unitree Robotics",
        license="BSD",
        tags=("quadruped",),
        **_LEGGED,
    )
)
register(
    RobotModel(
        "go1",
        "Unitree Go1",
        _m("unitree_go1/go1.xml"),
        _legs("FR", "FL", "RR", "RL"),
        base_body="trunk",
        maker="Unitree Robotics",
        license="BSD-3-Clause",
        tags=("quadruped",),
        **_LEGGED,
    )
)
register(
    RobotModel(
        "a1",
        "Unitree A1",
        _m("unitree_a1/a1.xml"),
        _legs("FR", "FL", "RR", "RL"),
        base_body="trunk",
        maker="Unitree Robotics",
        license="BSD-3-Clause",
        tags=("quadruped",),
        **_LEGGED,
    )
)
register(
    RobotModel(
        "spot",
        "Boston Dynamics Spot",
        _m("boston_dynamics_spot/spot.xml"),
        _legs("fl", "fr", "hl", "hr", names=("hx", "hy", "kn")),
        base_body="body",
        maker="Boston Dynamics",
        license="BSD-3-Clause",
        tags=("quadruped",),
        **_LEGGED,
    )
)
register(
    RobotModel(
        "anymal_c",
        "ANYbotics ANYmal C",
        _m("anybotics_anymal_c/anymal_c.xml"),
        _legs("LF", "RF", "LH", "RH", names=("HAA", "HFE", "KFE")),
        base_body="base",
        stand=(0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8),
        maker="ANYbotics",
        license="BSD",
        tags=("quadruped",),
        **_LEGGED,
    )
)
_G1_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
_G1_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
register(
    RobotModel(
        "g1",
        "Unitree G1",
        _m("unitree_g1/g1.xml"),
        tuple(f"{side}_{j}_joint" for side in ("left", "right") for j in _G1_LEG)
        + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        + tuple(f"{side}_{j}_joint" for side in ("left", "right") for j in _G1_ARM),
        base_body="pelvis",
        # squat: hips and ankles pitch against the knees so the torso stays over the feet
        crouch=(-0.8, 0, 0, 1.6, -0.8, 0) * 2 + (0,) * 17,
        maker="Unitree Robotics",
        license="BSD",
        tags=("humanoid",),
        **_LEGGED,
    )
)
