"""Learned policies: run a trained model as a robowright policy, whatever trained it.

A robowright policy reads ``obs`` (joint readings, camera images, the task, and with
``privileged=True`` object poses) and returns joint targets. A trained model reads and returns
tensors in the conventions it was trained with: its own observation keys, image layout, joint
order and units, gripper range, normalisation, batch dimension and action chunks.
:class:`LearnedPolicy` translates between the two, so any model runs in ``run_policy``, on
every robot and engine, and is tested like a scripted policy::

    policy = LearnedPolicy("checkpoints/pick.onnx", images=["front"], units="deg", gripper=(0, 100))
    rollout = robot.run_policy(policy, task="put the cube in the bin", until=done)

The model can be

* a Python object: a ``torch.nn.Module``, anything with ``select_action(batch)`` (the
  convention LeRobot and others follow), or any callable ``model(batch)``;
* a file: ONNX (``.onnx``, run with onnxruntime), ``torch.export`` (``.pt2``) or TorchScript
  (``.pt``, ``.ts``);
* ``package.module:name``: an object to import (a class is instantiated);
* ``loader:reference``: a model that a ``robowright.policies`` plugin loads. This is how a
  framework's own checkpoints (a hub id, a run directory) plug in without robowright
  depending on the framework;
* the name of a policy in the project's ``robowright.toml``, whose ``[policies.NAME]`` table
  holds the model and these settings.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

_FILES = (".onnx", ".pt2", ".pt", ".ts", ".torchscript")


class LearnedPolicy:
    """A trained model as a policy for ``robot.run_policy``.

    Observations, in the batch handed to the model:

    * ``state``: what goes into the state vector, in order. ``"qpos"`` is the joints (in the
      model's order, units and gripper range below); ``"tcp"`` is the tool position;
      ``"objects.NAME"`` is an object's position and ``"objects.NAME.quat"`` its (w, x, y, z)
      orientation (both make the policy privileged: object poses come from the engine, or on
      hardware from ``world.perception``). Passed as ``state_key``; ``None`` passes no state.
    * ``images``: the cameras the model sees, as ``{camera: key}`` or a list of cameras (keyed
      ``"observation.images.<camera>"``), rendered at ``image_size`` (default: the world's
      ``Settings.image_size``) and laid out as ``image_format``: ``"chw"`` (float, 0..1, the
      PyTorch convention) or ``"hwc"`` (uint8, as rendered).
    * the task string, as ``task_key``, when ``run_policy`` is given one.
    * ``batch``: whether a leading batch dimension is added (and taken off the action).

    Joints, in both directions:

    * ``joints``: the order the model numbers them in, by robowright's names (the robot's
      ``arm_joints``, and ``"gripper"``). Default: robowright's order.
    * ``units``: ``"rad"`` or ``"deg"`` for revolute joints (prismatic ones stay in metres).
    * ``gripper``: the model's values for closed and open, which robowright's opening (0 closed,
      1 open) maps to linearly; ``None`` if the model leaves the gripper alone.
    * ``normalize``: ``{"state": stats, "action": stats}`` with ``{"mean", "std"}`` or
      ``{"min", "max"}`` (scaled to -1..1), as the model was trained; or a JSON file holding it.

    The action is the model's output (``output_key`` of a dict, else the output itself):
    one action, or a chunk of them, of which the first ``steps`` run (default: all).

    ``args`` calls the model with those batch entries as positional arguments instead of the
    batch, for exports whose forward takes tensors (``args=["observation.state"]``).
    """

    def __init__(
        self,
        model,
        *,
        state=("qpos",),
        state_key: str | None = "observation.state",
        images=None,
        image_size: tuple | None = None,
        image_format: str = "chw",
        task_key: str | None = "task",
        batch: bool = True,
        joints=None,
        units: str = "rad",
        gripper=(0.0, 1.0),
        normalize=None,
        output_key: str = "action",
        steps: int | None = None,
        args=None,
        device: str | None = None,
    ):
        self.ref = model if isinstance(model, str) else None
        settings = {}
        if self.ref is not None:
            from . import plugins

            project = plugins.project_policies().get(self.ref)
            if project is not None:
                model, settings = project
        given = {k: v for k, v in locals().items() if k in _DEFAULTS and v != _DEFAULTS[k]}
        opts = {**_DEFAULTS, **settings, **given}  # the project's settings, under what is passed here
        self._given = given
        self.model = load_model(model, device=opts["device"]) if isinstance(model, str) else model
        self.state = tuple(opts["state"] or ())
        self.state_key = opts["state_key"]
        imgs = opts["images"] or {}
        self.images = dict(imgs) if isinstance(imgs, dict) else {c: f"observation.images.{c}" for c in imgs}
        self.image_size = tuple(opts["image_size"]) if opts["image_size"] else None
        if opts["image_format"] not in ("chw", "hwc"):
            raise ValueError(f'image_format is "chw" or "hwc", not {opts["image_format"]!r}')
        self.image_format = opts["image_format"]
        self.task_key, self.batch, self.output_key, self.steps = opts["task_key"], opts["batch"], opts["output_key"], opts["steps"]
        if opts["units"] not in ("rad", "deg"):
            raise ValueError(f'units is "rad" or "deg", not {opts["units"]!r}')
        self.units = opts["units"]
        self.joints = list(opts["joints"]) if opts["joints"] else None
        self.gripper = tuple(float(g) for g in opts["gripper"]) if opts["gripper"] is not None else None
        self.normalize = _stats(opts["normalize"])
        self.args = list(opts["args"]) if opts["args"] else None
        self.device = opts["device"]
        self._torch = _is_torch(self.model)
        self._call = getattr(self.model, "select_action", None) or self.model
        self._bound = None
        for f in self.state:
            if f not in ("qpos", "tcp") and not (f.startswith("objects.") and f.count(".") in (1, 2)):
                raise ValueError(f'state feature {f!r}: use "qpos", "tcp", "objects.NAME" or "objects.NAME.quat"')

    # --- what run_policy asks of a policy ---------------------------------------------------
    @property
    def cameras(self) -> tuple:
        return tuple(self.images)

    @property
    def privileged(self) -> bool:
        return any(f == "tcp" or f.startswith("objects.") for f in self.state)

    def reset(self):
        if hasattr(self.model, "reset"):
            self.model.reset()

    def to_config(self):
        """Settings that recreate this policy in a generated test (only for a model named by
        reference: an object handed in cannot be written into a test)."""
        if self.ref is None:
            return None
        return {"model": self.ref, **{k: list(v) if isinstance(v, tuple) else v for k, v in self._given.items()}}

    def __call__(self, obs) -> np.ndarray:
        self._bind(obs["robot"])
        out = self._infer(self.inputs(obs))
        a = np.asarray(out, float)
        if self.batch and a.ndim >= 2 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 1:
            a = a[None]
        if a.ndim != 2:
            raise ValueError(f"the model returned an action of shape {np.shape(out)}; expected (dof,) or a chunk (steps, dof)")
        if self.steps:
            a = a[: self.steps]
        return np.array([self.to_robot(row) for row in a])

    # --- the translation, both ways ---------------------------------------------------------
    def _bind(self, robot: str):
        if self._bound == robot:
            return
        from . import robots
        from .robot import _kinematics

        m = robots.get(robot)
        names = [*m.arm_joints, *(["gripper"] if m.has_gripper else [])]
        order = self.joints or [n for n in names if n != "gripper" or self.gripper is not None]
        unknown = [j for j in order if j not in names]
        if unknown:
            raise ValueError(f"joints {unknown} are not {m.name}'s; its joints are {names}")
        if "gripper" in order and self.gripper is None:
            raise ValueError('joints names "gripper" but gripper=None says the model leaves it alone')
        self._idx = np.array([names.index(j) for j in order])
        revolute = np.append(_kinematics(m.name).revolute, False)  # the gripper is an opening, not an angle
        self._deg = (revolute[self._idx] if self.units == "deg" else np.zeros(len(order), bool)).astype(bool)
        self._grip = np.array([j == "gripper" for j in order])
        self._n = len(names)
        self._bound = robot

    def to_model(self, q) -> np.ndarray:
        """Robowright joint values (arm, then the gripper opening) in the model's order and units."""
        v = np.asarray(q, float)[self._idx].copy()
        v[self._deg] = np.degrees(v[self._deg])
        if self.gripper is not None:
            lo, hi = self.gripper
            v[self._grip] = lo + v[self._grip] * (hi - lo)
        return v

    def to_robot(self, a) -> np.ndarray:
        """A model action as robowright joint targets. Joints the model does not drive (the
        gripper, with ``gripper=None``) are left out at the end, so they hold their targets."""
        a = np.asarray(a, float)
        if a.shape[-1] != len(self._idx):
            raise ValueError(f"the model's action has {a.shape[-1]} values; with joints {self._order()} it needs {len(self._idx)}")
        a = _unnormalize(a, self.normalize.get("action"))
        v = a.copy()
        v[self._deg] = np.radians(v[self._deg])
        if self.gripper is not None:
            lo, hi = self.gripper
            v[self._grip] = (v[self._grip] - lo) / (hi - lo)
        out = np.full(self._n, np.nan)
        out[self._idx] = v
        missing = np.isnan(out)
        if missing.any() and not missing[-int(missing.sum()) :].all():
            raise ValueError(f"the model drives {self._order()}; robowright needs every arm joint")
        return out[~missing]

    def _order(self) -> list[str]:
        from . import robots

        m = robots.get(self._bound)
        names = [*m.arm_joints, "gripper"]
        return [names[i] for i in self._idx]

    def inputs(self, obs) -> dict:
        """The batch handed to the model for ``obs``."""
        b = {}
        if self.state_key is not None and self.state:
            parts = []
            for f in self.state:
                if f == "qpos":
                    parts.append(self.to_model(obs["qpos"]))
                elif f == "tcp":
                    parts.append(np.asarray(self._need(obs, "tcp"), float))
                else:
                    _, name, *quat = f.split(".")
                    poses = self._need(obs, "objects")
                    if name not in poses:
                        raise KeyError(f"state feature {f!r}: no pose for {name!r} (objects: {sorted(poses)})")
                    parts.append(np.asarray(poses[name][1 if quat else 0], float))
            s = _normalize(np.concatenate(parts), self.normalize.get("state"))
            b[self.state_key] = s.astype(np.float32)
        for cam, key in self.images.items():
            img = self._need(obs, "images")[cam]
            if self.image_format == "chw":
                img = np.transpose(np.asarray(img, np.float32) / 255.0, (2, 0, 1))
            b[key] = np.ascontiguousarray(img)
        if self.task_key is not None and obs.get("task") is not None:
            b[self.task_key] = obs["task"]
        if self.batch:
            b = {k: [v] if isinstance(v, str) else v[None] for k, v in b.items()}
        return b

    def _need(self, obs, key):
        if key not in obs:
            how = "privileged=True" if key in ("objects", "tcp") else f"cameras={list(self.images)}"
            raise KeyError(f"the policy needs obs[{key!r}]: run it with {how}")
        return obs[key]

    def _infer(self, batch):
        if self._torch:
            import torch

            dev = self.device or _device(self.model)
            batch = {k: torch.as_tensor(v, device=dev) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
            with torch.inference_mode():
                out = self._call(*[batch[k] for k in self.args]) if self.args else self._call(batch)
        else:
            out = self._call(*[batch[k] for k in self.args]) if self.args else self._call(batch)
        if isinstance(out, dict):
            if self.output_key not in out:
                raise KeyError(f"the model returned {sorted(out)}; set output_key to the action's")
            out = out[self.output_key]
        if isinstance(out, (list, tuple)) and len(out) and not np.isscalar(out[0]) and np.ndim(out[0]) > 0:
            out = out[0]  # several outputs: the action is the first
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return out


_DEFAULTS = {
    "state": ("qpos",),
    "state_key": "observation.state",
    "images": None,
    "image_size": None,
    "image_format": "chw",
    "task_key": "task",
    "batch": True,
    "joints": None,
    "units": "rad",
    "gripper": (0.0, 1.0),
    "normalize": None,
    "output_key": "action",
    "steps": None,
    "args": None,
    "device": None,
}


def _stats(normalize) -> dict:
    if normalize is None:
        return {}
    if isinstance(normalize, (str, Path)):
        normalize = json.loads(Path(normalize).read_text())
    out = {}
    for key, s in normalize.items():
        if key not in ("state", "action"):
            raise ValueError(f'normalize has "state" and "action" statistics, not {key!r}')
        s = {k: np.asarray(v, float) for k, v in s.items()}
        if not ({"mean", "std"} <= set(s) or {"min", "max"} <= set(s)):
            raise ValueError(f'normalize[{key!r}] needs "mean" and "std", or "min" and "max"')
        out[key] = s
    return out


def _normalize(x, s):
    if not s:
        return x
    if "mean" in s:
        return (x - s["mean"]) / np.maximum(s["std"], 1e-8)
    return 2 * (x - s["min"]) / np.maximum(s["max"] - s["min"], 1e-8) - 1


def _unnormalize(x, s):
    if not s:
        return x
    if "mean" in s:
        return x * s["std"] + s["mean"]
    return (x + 1) / 2 * (s["max"] - s["min"]) + s["min"]


def _is_torch(model) -> bool:
    mod = type(model).__module__
    if not mod.startswith("torch") and not hasattr(model, "parameters"):
        return False
    try:
        import torch
    except ImportError:
        return False
    return isinstance(model, torch.nn.Module)


def _device(model):
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return "cpu"


# --- loading a model from a reference ----------------------------------------------------------
def load_model(ref: str, device: str | None = None):
    """The model ``ref`` names: a file, ``module:name``, or ``loader:reference`` (a plugin)."""
    from . import plugins

    prefix, sep, rest = ref.partition(":")
    if sep and len(prefix) > 1 and prefix in plugins.policy_loader_names():
        return plugins.load_policy_model(prefix, rest, device=device)
    path = Path(ref).expanduser()
    if path.suffix.lower() in _FILES or path.is_file():
        if not path.is_file():
            raise FileNotFoundError(f"no model file {ref!r}")
        if path.suffix.lower() == ".onnx":
            return OnnxModel(path, device)
        import torch

        if path.suffix.lower() == ".pt2":
            model = torch.export.load(str(path)).module()
            return model.to(device) if device else model
        model = torch.jit.load(str(path), map_location=device or "cpu")
        model.eval()
        return model
    if sep and len(prefix) > 1:
        try:
            obj = importlib.import_module(prefix)
        except ImportError as e:
            loaders = plugins.policy_loader_names()
            hint = f"; policy loaders installed: {', '.join(loaders)}" if loaders else ""
            raise ValueError(f"model {ref!r}: no module {prefix!r} and no policy loader of that name{hint}") from e
        for part in rest.split("."):
            obj = getattr(obj, part)
        return obj() if isinstance(obj, type) else obj
    raise ValueError(f"model {ref!r}: not a file, a module:name, a policy loader's loader:reference, or a policy in robowright.toml")


class OnnxModel:
    """An ONNX model run with onnxruntime: called with the batch, fed the inputs it declares."""

    def __init__(self, path, device: str | None = None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("running ONNX models needs onnxruntime: pip install 'robowright[onnx]'") from e
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device and device.startswith("cuda") else ["CPUExecutionProvider"]
        self.path = Path(path)
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]

    def __call__(self, batch=None, *args):
        if args or not isinstance(batch, dict):  # positional, as LearnedPolicy(args=...) passes them
            batch = dict(zip(self.inputs, (batch, *args)))
        missing = [n for n in self.inputs if n not in batch]
        if missing:
            raise KeyError(f"{self.path.name} takes {sorted(self.inputs)}; the batch has no {missing} (got {sorted(batch)})")
        feed = {n: np.asarray(batch[n], _ONNX_TYPES.get(i.type, np.float32)) for n, i in self.inputs.items()}
        out = self.session.run(None, feed)
        return out[0] if len(out) == 1 else dict(zip(self.outputs, out))


_ONNX_TYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(int64)": np.int64, "tensor(uint8)": np.uint8}
