"""The fixed obstacles a Gazebo world holds (its table, its floor, its walls), read as robowright's
collision model needs them.

A ROS 2 robot stands in a world robowright did not build: the table it stands on, the floor, what
the stack's launch put there. :func:`from_scene` turns Gazebo's scene (``gz.msgs.Scene``, read by
:meth:`robowright.gazebo.Gazebo.scene`) into fixed obstacles, with the poses Gazebo has, so the
planner can keep the arm off them. Only geometry Gazebo gives as a primitive is used: boxes,
cylinders, spheres and planes. A mesh has no size in the message, so it is skipped and named in
the notes rather than guessed at. Where a model has a bounding box and its meshes, the box stands in
for them (named after the model, and noted).

The models that are the robot itself (any model with one of the robot's joints) and the scene's
own objects are not obstacles: the planner already has them, and the objects are where the
simulation says they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    name: str  # the Gazebo model it belongs to (and its link, where there are several)
    kind: str  # box, cylinder, sphere or plane
    size: tuple  # box: half-extents; cylinder: (radius, half-length); sphere: (radius,); plane: (half-x, half-y)
    pos: tuple  # world position (m)
    quat: tuple  # world orientation (w, x, y, z)


def quat_mul(a, b) -> np.ndarray:
    """The quaternion a * b, both as (w, x, y, z)."""
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


def quat_rot(q, v) -> np.ndarray:
    """The vector v turned by the quaternion q (w, x, y, z)."""
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R @ np.asarray(v, float)


def _compose(parent: tuple, child: tuple) -> tuple:
    """A pose given relative to a parent, as a pose in the parent's parent's frame."""
    (pp, pq), (cp, cq) = parent, child
    return tuple(np.asarray(pp, float) + quat_rot(pq, cp)), tuple(quat_mul(pq, cq))


def from_scene(models: list[dict], robot: set[str] = frozenset(), exclude: set[str] = frozenset()) -> tuple[list[Obstacle], list[str]]:
    """Fixed obstacles from Gazebo's models, and notes on what was skipped.

    ``models`` is the scene as plain data (see :func:`robowright.gazebo.scene_dict`): each model has
    ``name``, ``pose`` (position, quaternion w x y z; relative to its parent), ``joints`` (names),
    ``links`` (each with ``name``, ``pose`` relative to the model, and ``collisions`` with ``name``,
    ``pose`` relative to the link and ``geometry``), ``models`` (nested) and optionally
    ``bounding_box`` (the model's corners (min, max) in its frame). ``robot``: the robot's link names: a model
    with one of them is the robot and is left out; ``exclude``: names of models left out (the scene's objects).
    A link's geometry is its collisions where it has any, else its visuals (what Gazebo draws).
    """
    out: list[Obstacle] = []
    notes: list[str] = []

    def walk(model: dict, parent: tuple) -> None:
        pose = _compose(parent, model.get("pose", ((0, 0, 0), (1, 0, 0, 0))))
        name = model["name"]
        if name in exclude or robot & {lk["name"] for lk in model.get("links", ())}:
            return
        if "bounding_box" in model and any(
            "mesh" in (c.get("geometry") or {}) for lk in model.get("links", ()) for c in lk.get("collisions", ())
        ):
            # A mesh has no size of its own in the message: the model's box stands in for it, where it has one.
            (x0, y0, z0), (x1, y1, z1) = model["bounding_box"]
            half = ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
            center = pose[0] + quat_rot(pose[1], ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
            out.append(Obstacle(f"{name} (bounding box)", "box", half, tuple(center), pose[1]))
            notes.append(f"{name}: its meshes are taken as the model's bounding box (the scene message gives no other size)")
            mesh_as_box = True
        else:
            mesh_as_box = False
        for link in model.get("links", ()):
            lpose = _compose(pose, link.get("pose", ((0, 0, 0), (1, 0, 0, 0))))
            for col in link.get("collisions") or link.get("visuals", ()):
                cpose = _compose(lpose, col.get("pose", ((0, 0, 0), (1, 0, 0, 0))))
                label = f"{name}/{link['name']}" if len(model.get("links", ())) > 1 else name
                geom = col.get("geometry") or {}
                if "box" in geom:
                    out.append(Obstacle(label, "box", tuple(geom["box"]), *cpose))
                elif "cylinder" in geom:
                    out.append(Obstacle(label, "cylinder", tuple(geom["cylinder"]), *cpose))
                elif "sphere" in geom:
                    out.append(Obstacle(label, "sphere", (geom["sphere"],), *cpose))
                elif "plane" in geom:
                    out.append(Obstacle(label, "plane", tuple(geom["plane"]), *cpose))
                elif "mesh" in geom and mesh_as_box:
                    pass
                elif "mesh" in geom:
                    notes.append(f"{label}: a mesh ({geom['mesh']}) has no size in the scene message: not an obstacle")
                else:
                    notes.append(f"{label}: a geometry robowright does not model ({sorted(geom) or 'none'}): not an obstacle")
        for sub in model.get("models", ()):
            walk(sub, pose)

    for m in models:
        walk(m, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    return out, notes
