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

__all__ = ["PREFIX", "Attachment", "Derived", "Kinematics", "RobotModel", "body_labels", "get", "names", "register", "REGISTRY"]

REGISTRY: dict[str, RobotModel] = {}


def register(model: RobotModel) -> RobotModel:
    REGISTRY[model.name] = model
    return model


def get(name: str | RobotModel) -> RobotModel:
    if isinstance(name, RobotModel):
        return name
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; available: {', '.join(sorted(REGISTRY))}") from None


def names(family: str | None = None) -> list[str]:
    return [n for n, m in REGISTRY.items() if family is None or m.family == family]


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
        tags=("research", "7dof"),
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
        tags=("cobot", "7dof"),
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
        tags=("low-cost", "6dof"),
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
        tags=("research", "7dof"),
        **_2F85,
    )
)
