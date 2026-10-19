# rw.attach

Attach to a simulation someone else built and steps: a MuJoCo, PyBullet, Genesis, Drake, SAPIEN, Webots, Isaac Lab
or Isaac Sim world, a gymnasium environment, or a vector of them. Their code keeps stepping it; robowright observes,
checks invariants on every step, applies faults, and records the run.

::: robowright.attached.attach

::: robowright.attached.AttachedWorld
    options:
      members: [bodies, joints, body, joint, step, run_until, model, data]

::: robowright.attached.AttachedSubject

::: robowright.attached.AttachedJoint

::: robowright.attached.AttachedScene

::: robowright.attached_remote.attachable
