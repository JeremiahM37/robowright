"""The contract: what every engine does for every robot, as pytest tests.

``test_arm`` holds the arm contract (kinematics, servos, the gripper, contacts, state, picking
and placing), ``test_legged`` the one for quadrupeds and humanoids. They ship with robowright so
that a robot or an engine from outside it can be held to the same tests::

    robowright check --robot my_arm --backend mujoco,my_engine

which runs ``pytest --pyargs robowright.contract`` with those options (see ``robowright.cli``).
"""
