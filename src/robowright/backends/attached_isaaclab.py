"""An Isaac Lab environment that robowright did not build: one its own code made (``gym.make("Isaac-...")``) and steps.

Isaac Lab runs many copies of a task at once on the GPU (``num_envs``); robowright watches one of them, ``env_index``
(0 by default). Everything is read from the environment's scene (``env.unwrapped.scene``):

* Names. Each articulation's links are bodies named ``<articulation>/<link>`` (``robot/panda_hand``); a link name
  only one articulation has also answers alone (``panda_hand``). Rigid objects are bodies by their scene name
  (``object``), and are the free bodies a test can move. Joints are ``<articulation>/<joint>``, and a joint name
  only one articulation has also answers alone.
* Where things are: in the watched copy's own frame, that is relative to its origin in the grid Isaac Lab lays the
  copies out on, so a test reads the same numbers whichever copy it watches. Quaternions are (w, x, y, z), as
  Isaac Lab's are.
* Contacts come from the scene's contact sensors (``ContactSensor``); a scene without one has none to give, and
  says so.
* Forces. Isaac Lab keeps an external force until it is changed, so a push is set before each step it covers and
  cleared after the last one.
* The clock is the simulation's (``sim.current_time``), and a control step is the environment's ``step_dt``.
* Rendering and saved state are not read here: a run is recorded without a state to redraw or replay it from.
"""

from __future__ import annotations

import numpy as np

from ..errors import CapabilityError
from .attached_engine import AttachedEngine
from .base import FORCES, GROUND_TRUTH


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _unique_aliases(qualified: list[str]) -> dict[str, str]:
    """``link -> art/link`` for the link names only one articulation has."""
    seen: dict[str, list[str]] = {}
    for q in qualified:
        seen.setdefault(q.split("/", 1)[1], []).append(q)
    return {short: qs[0] for short, qs in seen.items() if len(qs) == 1}


class IsaacLabEngine(AttachedEngine):
    """Handles are ``("link", articulation, link index)`` or ``("object", name)``."""

    engine = "isaaclab"

    def __init__(self, env, *, env_index: int = 0, seed: int = 0, dt: float | None = None):
        self.env = env
        self.u = getattr(env, "unwrapped", env)
        self.scene = self.u.scene
        n = int(self.u.num_envs)
        if not 0 <= env_index < n:
            raise ValueError(f"env_index {env_index} is not one of this environment's {n} copies (0-{n - 1})")
        self.index = int(env_index)
        super().__init__(seed, dt=dt if dt is not None else float(self.u.step_dt))
        self._contact_sensors = [s for s in self.scene.sensors.values() if type(s).__name__ == "ContactSensor"]
        self.capabilities = frozenset({GROUND_TRUTH, FORCES} | ({"contacts"} if self._contact_sensors else set()))
        self._forced: set = set()
        self._forced_now: set = set()
        self._scan()

    def _scan(self) -> None:
        self._links: dict[str, tuple[str, int]] = {}
        for art_name, art in self.scene.articulations.items():
            for i, link in enumerate(art.body_names):
                self._links[f"{art_name}/{link}"] = (art_name, i)
        self._link_alias = _unique_aliases(list(self._links))
        self._objects = list(self.scene.rigid_objects.keys())
        self._joints: dict[str, tuple[str, int]] = {}
        for art_name, art in self.scene.articulations.items():
            for j, joint in enumerate(art.joint_names):
                self._joints[f"{art_name}/{joint}"] = (art_name, j)
        self._joint_alias = _unique_aliases(list(self._joints))
        self.joint_names = list(self._joints)
        self.free_bodies = list(self._objects)

    def _origin(self) -> np.ndarray:
        return _np(self.scene.env_origins[self.index])

    # --- naming -----------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        if kind == "body":
            return [*self._links, *self._objects]
        if kind in ("geom", "site"):
            return []  # Isaac Lab names links and objects; a test names those
        raise ValueError(f"unknown kind {kind!r}")

    def resolve(self, name: str):
        if name in self._objects:
            return ("object", name)
        full = name if name in self._links else self._link_alias.get(name)
        if full is not None:
            art, i = self._links[full]
            return ("link", art, i)
        if name in self.scene.articulations:  # an articulation by its own name: its root link
            return ("link", name, 0)
        return None

    def label(self, handle) -> str:
        if handle[0] == "object":
            return handle[1]
        return f"{handle[1]}/{self.scene.articulations[handle[1]].body_names[handle[2]]}"

    # --- ground truth -----------------------------------------------------------------
    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        i = self.index
        if handle[0] == "object":
            d = self.scene.rigid_objects[handle[1]].data
            pos, quat = d.root_pos_w[i], d.root_quat_w[i]
        else:
            d = self.scene.articulations[handle[1]].data
            pos, quat = d.body_pos_w[i, handle[2]], d.body_quat_w[i, handle[2]]
        return _np(pos).reshape(3) - self._origin(), _np(quat).reshape(4)

    def velocity(self, handle) -> np.ndarray:
        i = self.index
        if handle[0] == "object":
            d = self.scene.rigid_objects[handle[1]].data
            lin, ang = d.root_lin_vel_w[i], d.root_ang_vel_w[i]
        else:
            d = self.scene.articulations[handle[1]].data
            lin, ang = d.body_lin_vel_w[i, handle[2]], d.body_ang_vel_w[i, handle[2]]
        return np.concatenate([_np(lin).reshape(3), _np(ang).reshape(3)])

    # --- contacts ---------------------------------------------------------------------
    def _no_contacts(self):
        raise CapabilityError(
            "this Isaac Lab scene has no contact sensor (ContactSensor), so it reports no contacts: add one to the scene "
            "to test what touches what"
        )

    def contact_pairs(self) -> list[tuple]:
        if not self._contact_sensors:
            self._no_contacts()
        out = []
        for s in self._contact_sensors:
            forces = _np(s.data.net_forces_w[self.index])  # (bodies, 3)
            names = list(getattr(s, "body_names", []))
            for b, f in enumerate(forces):
                mag = float(np.linalg.norm(f))
                if mag > 1e-6 and b < len(names):
                    out.append((names[b], "?", mag))  # a net force says a body is touched, not by what
        return out

    def contacts_of(self, handle) -> list[str]:
        me = self.label(handle).split("/")[-1]
        return sorted({other for a, other, _ in self.contact_pairs() if a == me})

    # --- joints and the free bodies -----------------------------------------------------
    def named_joints(self) -> list[str]:
        return list(self._joints)

    def _joint(self, name: str):
        full = name if name in self._joints else self._joint_alias.get(name)
        if full is None:
            raise KeyError(f"no joint named {name!r}; joints: {', '.join(self._joints) or 'none'}")
        return self._joints[full]

    def joint_qadr(self, name: str) -> int:
        return self._joint(name)[1]

    def joint_value(self, name: str) -> float:
        art, j = self._joint(name)
        return float(_np(self.scene.articulations[art].data.joint_pos[self.index, j]))

    def set_joint(self, name: str, value: float) -> None:
        import torch

        art_name, j = self._joint(name)
        art = self.scene.articulations[art_name]
        pos, vel = art.data.joint_pos[self.index : self.index + 1].clone(), art.data.joint_vel[self.index : self.index + 1].clone()
        pos[0, j], vel[0, j] = float(value), 0.0
        art.write_joint_state_to_sim(pos, vel, env_ids=torch.tensor([self.index], device=pos.device))

    def qpos(self) -> np.ndarray:
        return np.array([self.joint_value(n) for n in self.joint_names], dtype=float)

    def qvel(self) -> np.ndarray:
        out = []
        for n in self.joint_names:
            art, j = self._joint(n)
            out.append(float(_np(self.scene.articulations[art].data.joint_vel[self.index, j])))
        return np.array(out, dtype=float)

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        import torch

        if name not in self._objects:
            raise ValueError(f"{name!r} is not a free body, so its pose cannot be set (free bodies: {', '.join(self._objects)})")
        obj = self.scene.rigid_objects[name]
        dev = obj.data.root_pos_w.device
        q = obj.data.root_quat_w[self.index] if quat is None else torch.tensor(np.asarray(quat, float), device=dev, dtype=torch.float32)
        p = torch.tensor(np.asarray(pos, float) + self._origin(), device=dev, dtype=torch.float32)
        ids = torch.tensor([self.index], device=dev)
        obj.write_root_pose_to_sim(torch.cat([p, q]).reshape(1, 7), env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=ids)

    def apply_force(self, name: str, force) -> None:
        """A world-frame force (N) on the body, for the step being taken (cleared after it unless set again)."""
        handle = self.resolve(name)
        if handle is None:
            raise KeyError(f"no body named {name!r} in this simulation")
        self._set_force(handle, np.asarray(force, dtype=float))
        self._forced_now.add(handle)

    def _set_force(self, handle, force: np.ndarray) -> None:
        import torch

        if handle[0] == "object":
            asset, body = self.scene.rigid_objects[handle[1]], 0
        else:
            asset, body = self.scene.articulations[handle[1]], handle[2]
        dev = asset.data.root_pos_w.device
        f = torch.tensor(force, device=dev, dtype=torch.float32).reshape(1, 1, 3)
        asset.set_external_force_and_torque(f, torch.zeros_like(f), body_ids=[body], env_ids=torch.tensor([self.index], device=dev))

    def note_step(self) -> None:
        # Isaac Lab keeps an external force until it is changed: one that was not set again for this step ends now
        for handle in self._forced - self._forced_now:
            self._set_force(handle, np.zeros(3))
        self._forced, self._forced_now = self._forced_now, set()

    # --- clock ------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.u.sim.current_time)

    def trace_meta(self) -> dict:
        return {"engine": self.engine, "inputs_recorded": False, "env_index": self.index, "num_envs": int(self.u.num_envs)}


def is_isaaclab(env) -> bool:
    """An Isaac Lab environment: a subclass of one of Isaac Lab's (ManagerBasedEnv, DirectRLEnv, ...). A task's own class
    lives in its package (isaaclab_tasks.direct...), not in isaaclab, so the class it derives from is what tells."""
    u = getattr(env, "unwrapped", env)
    return hasattr(u, "scene") and any(c.__module__.startswith("isaaclab.envs") for c in type(u).__mro__)


def isaaclab_engine_of(env, *, env_index: int = 0, seed: int = 0, dt: float | None = None) -> IsaacLabEngine | None:
    """The Isaac Lab adapter for an environment Isaac Lab made, or None."""
    if not is_isaaclab(env):
        return None
    return IsaacLabEngine(env, env_index=env_index, seed=seed, dt=dt)
