"""Read a robot's URDF: the format ROS, and most robot makers, describe their robots in.

MuJoCo reads URDF itself (links, joints, limits, inertias, collision shapes, ``<mimic>``
joints as equality constraints), so this module only does what its reader cannot:

* **mesh paths.** ``package://name/...`` is resolved the way ROS would, from
  ``ROS_PACKAGE_PATH`` and then from the directories around the file (a package is a
  directory of that name, or one whose ``package.xml`` gives that name); ``file://`` and
  relative paths are taken as they are.
* **mesh formats** MuJoCo cannot load (COLLADA ``.dae`` above all) are converted to OBJ with
  ``trimesh``. Without it, visual meshes are left out and collision meshes are an error.
* **actuators.** A URDF names no motors, so every joint that is not a ``<mimic>`` follower
  gets a position servo: stiff enough to reach the joint's ``effort`` limit within
  ``_HINGE_ERR`` of its target (``_SLIDE_ERR`` for slides), critically damped, with the
  motor inertia (armature) a geared joint needs for that stiffness to integrate stably.

The result is an MJCF file in robowright's cache, which :mod:`.detect` then reads like any
other; every decision is in the notes ``robowright robots --inspect`` prints.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .model import _collision_geoms, reset_data

VERSION = 1  # bump when the output changes, to invalidate cached conversions
_LOADS = {".stl", ".obj", ".msh"}  # mesh formats MuJoCo reads
_HINGE_ERR = 0.02  # rad from target at full effort
_SLIDE_ERR = 0.002  # m
_PAYLOAD = 1.5  # kg at the end of the arm, for joints whose effort the file leaves out
_ACCEL = 5.0  # rad/s^2 (m/s^2 for slides) a joint whose effort is guessed can give its load
_PLACEHOLDER = 50.0  # an effort limit this many times what a joint needs is a placeholder (1000 N m on a hobby servo)
_OMEGA_DT = 0.3  # servo natural frequency times the timestep: stable with margin under implicitfast
_HINGE, _SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)


class URDFError(ValueError):
    """The URDF could not be read; the message says why."""


def to_mjcf(path: Path, cache: Path) -> tuple[Path, list[str]]:
    """An MJCF file for the URDF at ``path`` (written once under ``cache``), and notes on how."""
    raw = path.read_bytes()
    out = cache / f"{path.stem}-{hashlib.sha1(raw + str(path).encode() + str(VERSION).encode()).hexdigest()[:12]}.xml"
    notes: list[str] = []
    tree = _resolved(path, raw, cache / "meshes", notes)
    spec = _load(tree, cache)
    notes.append("collision: the robot's links collide with the world but not with each other (as PyBullet treats a URDF)")
    _add_servos(spec, notes)
    xml = spec.to_xml()
    if not out.exists() or out.read_text() != xml:
        _write(out, xml)
    notes.insert(0, f"model: read from URDF {path.name} (converted to {out})")
    return out, notes


def _write(out: Path, text: str) -> None:
    """Replace ``out`` atomically: several test workers convert the same file at once."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=out.suffix)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, out)


# ---------------------------------------------------------------------------------------------
def _resolved(path: Path, raw: bytes, meshes: Path, notes) -> ET.ElementTree:
    """The URDF with every mesh path absolute and loadable, and MuJoCo told to keep what robowright needs."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise URDFError(f"{path.name} is not valid XML ({e}); a .xacro file must be expanded first: xacro FILE > robot.urdf") from e
    if root.tag != "robot":
        raise URDFError(f"{path.name} is not a URDF (its root element is <{root.tag}>, not <robot>)")
    if any(el.tag.startswith("{http://www.ros.org/wiki/xacro}") or el.tag.startswith("xacro:") for el in root.iter()):
        raise URDFError(f"{path.name} is a xacro template; expand it first: xacro {path.name} > robot.urdf")
    # A link drawn but given no collision shape (the SO-100's gripper and jaw) could not touch
    # anything; its visual meshes are what it is shaped like. Not a massless one: that is a
    # frame drawn as a marker (an end-effector point), not a part.
    copied = []
    for link in root.iter("link"):
        mass = float((link.find("inertial/mass") if link.find("inertial/mass") is not None else ET.Element("m")).get("value", 0))
        if mass > 0 and link.find("visual/geometry") is not None and link.find("collision") is None:
            for v in link.findall("visual"):
                c = ET.SubElement(link, "collision")
                c.extend(copy.deepcopy(e) for e in v if e.tag in ("origin", "geometry"))
            copied.append(link.get("name"))
    if copied:
        notes.append(f"collision: links {copied} have none in the file, so their visual shapes are used")
    dropped, converted = [], 0
    for kind in ("visual", "collision"):
        for link in root.iter("link"):
            for el in list(link.findall(kind)):
                mesh = el.find("geometry/mesh")
                if mesh is None:
                    continue
                file = _find(mesh.get("filename", ""), path)
                if file.suffix.lower() not in _LOADS:
                    obj = _convert(file, meshes)
                    if obj is None:
                        if kind == "collision":
                            raise URDFError(
                                f"collision mesh {file.name} is {file.suffix} and MuJoCo reads only STL, OBJ and MSH;"
                                " pip install trimesh pycollada to convert it"
                            )
                        link.remove(el)
                        dropped.append(file.name)
                        continue
                    file, converted = obj, converted + 1
                mesh.set("filename", str(file))
    if converted:
        notes.append(f"meshes: {converted} converted to OBJ (MuJoCo does not read their format)")
    if dropped:
        notes.append(f"meshes: visual meshes {sorted(set(dropped))} left out (pip install trimesh pycollada to convert them)")
    for el in root.findall("mujoco") + root.findall("gazebo") + root.findall("transmission"):
        root.remove(el)
    ext = ET.SubElement(root, "mujoco")
    # Keep every link (a fixed "tool0" or "flange" is where a gripper goes), its visuals, and
    # the paths just resolved; inertias the file leaves physically impossible are evened out.
    ET.SubElement(ext, "compiler", strippath="false", fusestatic="false", discardvisual="false", balanceinertia="true")
    return ET.ElementTree(root)


def _find(name: str, urdf: Path) -> Path:
    """The file a URDF mesh ``filename`` names."""
    if name.startswith("file://"):
        p = Path(name[len("file://") :])
        if p.exists():
            return p
    elif name.startswith("package://"):
        pkg, _, rest = name[len("package://") :].partition("/")
        for d in _package_dirs(pkg, urdf):
            if (d / rest).exists():
                return (d / rest).resolve()
    elif Path(name).is_absolute():
        if Path(name).exists():
            return Path(name)
    else:
        # Relative to the file, or (a common slip) to the package root above a urdf/ directory.
        for up in [urdf.parent, *urdf.parent.parents][:3]:
            if (up / name).exists():
                return (up / name).resolve()
    raise URDFError(f"mesh {name!r} not found (looked in ROS_PACKAGE_PATH and the directories around {urdf.name})")


def _package_dirs(pkg: str, urdf: Path):
    """Where ROS package ``pkg`` may be: ROS_PACKAGE_PATH, then each directory above the file."""
    for base in filter(None, os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep)):
        b = Path(base)
        yield b / pkg
        if b.name == pkg:
            yield b
    for up in [urdf.parent, *urdf.parent.parents][:8]:
        if up.name == pkg or _package_name(up) == pkg:
            yield up
        yield up / pkg
        for xml in up.glob("*/package.xml"):  # a package whose directory has another name
            if _package_name(xml.parent) == pkg:
                yield xml.parent
    # Not a ROS package at all ("package://meshes/..." meaning "the meshes next to me"): a
    # directory of that name near the file.
    for up in [urdf.parent, *urdf.parent.parents][:3]:
        yield up


def _package_name(d: Path) -> str | None:
    try:
        m = re.search(r"<name>\s*([^<\s]+)\s*</name>", (d / "package.xml").read_text(errors="replace"))
    except OSError:
        return None
    return m.group(1) if m else None


def _convert(file: Path, out: Path) -> Path | None:
    """``file`` as OBJ (cached), or None without trimesh (or if it cannot read the file)."""
    try:
        import trimesh
    except ImportError:
        return None
    obj = out / f"{file.stem}-{hashlib.sha1(str(file).encode()).hexdigest()[:10]}.obj"
    if obj.exists():
        return obj
    try:
        mesh = trimesh.load(str(file), force="mesh")
    except Exception:  # noqa: BLE001 - trimesh raises many kinds; any of them means "cannot convert"
        return None
    if mesh.is_empty:
        return None
    out.mkdir(parents=True, exist_ok=True)
    _write(obj, mesh.export(file_type="obj", include_texture=False))
    return obj


_CONCAVE = 1.3  # a mesh whose convex hull is this much bigger than it is split into convex parts
_BULGE = 0.005  # m: a gripper mesh whose hull reaches this much further into the grasp is split
_GEOM_ATTRS = ("contype", "conaffinity", "condim", "friction", "solref", "solimp", "margin", "gap", "group", "priority", "rgba", "density")
# CoACD's default concavity, with a lighter search: 12 s rather than 24 s on the SO-101's jaw
# (28k faces). A coarser split (threshold 0.08, 6 s) left the xArm's finger pockets lumpy
# enough to squeeze a cube out sideways; this one holds it, as the default does.
_COACD = dict(threshold=0.05, max_convex_hull=16, resolution=1000, preprocess_resolution=30, mcts_nodes=10, mcts_iterations=60, seed=0)


def _convex_parts(file: Path, out: Path) -> list[Path] | None:
    """``file`` as convex pieces: ``[file]`` if it is (nearly) convex already, and None if it
    is not but cannot be split (no trimesh or CoACD). Pieces are cached next to converted meshes."""
    try:
        import trimesh
    except ImportError:
        return [file]  # cannot tell; MuJoCo takes the hull
    key = hashlib.sha1(file.read_bytes() + repr(sorted(_COACD.items())).encode()).hexdigest()[:12]
    done = out / f"{file.stem}-{key}.parts"
    if done.exists():
        return [Path(p) for p in done.read_text().split("\n") if p]
    out.mkdir(parents=True, exist_ok=True)
    with _locked(out / f"{file.stem}-{key}.lock"):  # test workers wait for one split, not each do it
        if done.exists():
            return [Path(p) for p in done.read_text().split("\n") if p]
        return _split(file, key, done, out, trimesh)


def _split(file: Path, key: str, done: Path, out: Path, trimesh) -> list[Path] | None:
    mesh = trimesh.load(str(file), force="mesh")
    # Volume means something only for a closed surface; an open one is taken as it is.
    if mesh.is_empty or not mesh.is_watertight or mesh.volume <= 0 or mesh.convex_hull.volume < _CONCAVE * mesh.volume:
        parts = [file]
    else:
        try:
            import coacd
        except ImportError:
            return None
        coacd.set_log_level("error")
        pieces = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **_COACD)
        parts = []
        for k, (v, f) in enumerate(pieces):
            p = out / f"{file.stem}-{key}-{k}.obj"
            _write(p, trimesh.Trimesh(v, f).export(file_type="obj"))
            parts.append(p)
    _write(done, "\n".join(str(p) for p in parts))
    return parts


@contextlib.contextmanager
def _locked(path: Path):
    try:
        import fcntl
    except ImportError:  # Windows: no lock, at worst the work is done twice
        yield
        return
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------------------------
def split_jaws(mjcf: Path, hand: str, fingers: tuple[str, str], cache: Path) -> tuple[Path, list[str]]:
    """The model with each gripper mesh whose convex hull fills the grasp split into convex parts.

    Every engine collides a mesh as its convex hull. That is right for an arm's links, and for
    a finger whose inner face is a pocket for a rubber pad (the xArm's: the hull's flat face is
    where the pad would be). It is wrong for a jaw shaped like an L (the SO-100's and SO-101's
    fixed jaw), whose hull fills the gripper's mouth: the object is pushed out instead of
    held. So a mesh on the hand or a finger is split (CoACD) only where its hull reaches more
    than _BULGE further towards the grasp than the mesh, within the last 4.5 cm of the fingers.
    """
    try:
        import trimesh
    except ImportError:
        return mjcf, []
    spec = mujoco.MjSpec.from_file(str(mjcf))
    for i, g in enumerate(spec.geoms):
        if not g.name:
            g.name = f"robowright_geom{i}"
    m = spec.compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    h = m.body(hand).id
    below = lambda b0: {b for b in range(m.nbody) if b == b0 or b0 in _chain(m, b)}  # noqa: E731
    sides = [below(m.body(f).id) if f != hand else {h} for f in fingers]

    def mesh_of(g):
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        f = m.mesh_face[m.mesh_faceadr[k] : m.mesh_faceadr[k] + m.mesh_facenum[k]]
        return trimesh.Trimesh(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f, process=False)

    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    hand_side = below(h)
    meshes = [g for g in _collision_geoms(m, hand_side) if m.geom_type[g] == mesh]
    if not meshes:
        return mjcf, []
    pts = {g: mesh_of(g) for g in meshes}
    side_pts = [np.vstack([pts[g].vertices for g in meshes if m.geom_bodyid[g] in side] or [np.zeros((0, 3))]) for side in sides]
    if any(len(p) == 0 for p in side_pts):
        return mjcf, []
    axis = np.vstack(side_pts).mean(axis=0) - d.xpos[h]
    axis /= np.linalg.norm(axis)
    along = lambda p: (p - d.xpos[h]) @ axis  # noqa: E731
    tip = max(along(p).max() for p in side_pts)
    centre = np.mean([p[along(p) > along(p).max() - 0.02].mean(axis=0) for p in side_pts], axis=0)
    bulging = {}
    for g in meshes:
        mesh = pts[g]
        u = centre - mesh.vertices.mean(axis=0)
        u -= axis * (u @ axis)
        if np.linalg.norm(u) < 1e-4:
            continue
        u /= np.linalg.norm(u)
        on = trimesh.sample.sample_surface(mesh, 20000, seed=0)[0]
        hull = trimesh.sample.sample_surface(mesh.convex_hull, 20000, seed=0)[0]
        worst = 0.0
        for lo in np.arange(tip - 0.045, tip, 0.005):
            a, b = (p[(along(p) >= lo) & (along(p) < lo + 0.005)] for p in (on, hull))
            if len(a) and len(b):
                worst = max(worst, float((b @ u).max() - (a @ u).max()))
        if worst > _BULGE:
            bulging[m.geom(g).name] = worst
    if not bulging:
        return mjcf, []
    notes, unsplit = [], []
    for name, worst in bulging.items():
        g = spec.geom(name)
        mesh = spec.mesh(g.meshname)
        file = Path(mesh.file) if Path(mesh.file).is_absolute() else Path(spec.meshdir or mjcf.parent) / mesh.file
        parts = _convex_parts(file, cache / "meshes")
        if not parts or len(parts) < 2:
            unsplit.append(file.name)
            continue
        body = g.parent
        for k, part in enumerate(parts):
            pm = spec.add_mesh(name=f"{mesh.name}_part{k}", file=str(part))
            pm.scale = mesh.scale
            c = body.add_geom(name=f"{name}_part{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=pm.name, pos=g.pos, quat=g.quat)
            for attr in _GEOM_ATTRS:
                setattr(c, attr, getattr(g, attr))
        spec.delete(g)
        notes.append(f"{file.name} ({worst * 1000:.0f} mm, {len(parts)} parts)")
    out = []
    if notes:
        out.append(f"collision: gripper meshes whose hulls would fill the grasp are split into convex parts (CoACD): {notes}")
    if unsplit:
        out.append(f"warning: the hulls of {unsplit} fill part of the grasp, and splitting them needs pip install trimesh coacd")
    if not notes:
        return mjcf, out
    path = mjcf.with_name(f"{mjcf.stem}-jaws.xml")
    _write(path, spec.to_xml())
    return path, out


def _chain(m, b) -> list[int]:
    out = []
    while b > 0:
        b = int(m.body_parentid[b])
        out.append(b)
    return out


def _load(tree: ET.ElementTree, cache: Path) -> mujoco.MjSpec:
    """MuJoCo's own reading of the URDF."""
    tmp = cache / f".{os.getpid()}-{id(tree)}.urdf"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tree.write(tmp, encoding="unicode")
    try:
        spec = mujoco.MjSpec.from_file(str(tmp))
        spec.compile()
    except Exception as e:  # noqa: BLE001 - MuJoCo raises ValueError/Exception with its parser's message
        raise URDFError(f"MuJoCo could not read the URDF: {str(e).strip()}") from e
    finally:
        tmp.unlink(missing_ok=True)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Friction as MuJoCo's own gripper models set it up for holding things: an elliptic cone,
    # with friction kept from slipping (impratio), and torsional friction at the contacts.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.condim = 4
    # A URDF's collision shapes are drawn for motion planning, where links that overlap at
    # their joints, or brush past each other in normal motion, do no harm. As PyBullet does
    # for URDFs by default, the robot's links do not collide with each other, only with the world.
    m = spec.compile()
    bodies = sorted({m.body(int(b)).name for b in m.geom_bodyid if b > 0})
    for i, a in enumerate(bodies):
        for b in bodies[i + 1 :]:
            spec.add_exclude(bodyname1=a, bodyname2=b)
    return spec


def _add_servos(spec: mujoco.MjSpec, notes) -> None:
    """A position servo on every joint that is not a ``<mimic>`` follower."""
    m = spec.compile()
    followers = {int(m.eq_obj1id[e]) for e in range(m.neq) if m.eq_type[e] == int(mujoco.mjtEq.mjEQ_JOINT)}
    joints = [j for j in range(m.njnt) if m.jnt_type[j] in (_HINGE, _SLIDE) and j not in followers]
    if not joints:
        raise URDFError("the URDF has no movable joints")
    d = mujoco.MjData(m)
    reset_data(m, d)
    M = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, M)
    except TypeError:
        mujoco.mj_fullM(m, M, d.qM)
    dt = float(m.opt.timestep)
    load = _gravity_load(m)
    guessed, added, placeholder = [], [], []
    for j in joints:
        js = spec.joint(m.joint(j).name)
        dof = int(m.jnt_dofadr[j])
        slide = m.jnt_type[j] == _SLIDE
        effort = float(m.jnt_actfrcrange[j, 1]) if m.jnt_actfrclimited[j] else 0.0
        need = max(2.0 * float(load[dof]), 1.0 if not slide else 10.0)
        if effort > _PLACEHOLDER * need:
            placeholder.append(f"{m.joint(j).name} ({effort:g})")
            effort = 0.0
        if effort <= 0:
            # No limit in the file: twice what holding up and accelerating the arm takes.
            effort = need
            guessed.append(m.joint(j).name)
        kp = effort / (_SLIDE_ERR if slide else _HINGE_ERR)
        inertia = float(M[dof, dof])
        # A servo of stiffness kp on inertia I rings at sqrt(kp / I); integrating that stably
        # takes sqrt(kp / I) * dt below _OMEGA_DT. Geared motors carry reflected inertia that
        # URDFs leave out, and adding it back is what makes a stiff servo stable.
        need = kp * (dt / _OMEGA_DT) ** 2
        if need > inertia:
            js.armature = float(js.armature) + need - inertia
            added.append(m.joint(j).name)
            inertia = need
        kv = 2.0 * np.sqrt(kp * inertia)
        lo, hi = (float(v) for v in m.jnt_range[j]) if m.jnt_limited[j] else (0.0, 0.0)
        a = spec.add_actuator(name=m.joint(j).name, target=m.joint(j).name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp, kv=kv)
        if m.jnt_limited[j]:
            a.ctrlrange = [lo, hi]
            a.ctrllimited = 1
        a.forcerange = [-effort, effort]
        a.forcelimited = 1
    notes.append(
        f"actuators: none in a URDF, so a position servo on each of {len(joints)} joints, reaching its effort limit"
        f" {_HINGE_ERR} rad ({_SLIDE_ERR * 1000:g} mm for slides) from its target"
        + (f"; {len(followers)} <mimic> joint(s) follow theirs" if followers else "")
    )
    if placeholder:
        notes.append(f"effort: {placeholder} in the file read as placeholders (over {_PLACEHOLDER:g} times what the joints need)")
    if guessed:
        notes.append(f"effort: for {guessed}, assumed twice what holding and accelerating the arm (and {_PAYLOAD:g} kg at its end) takes")
    if added:
        notes.append(f"armature: motor inertia added to {added} so their servos integrate stably")


def _gravity_load(m: mujoco.MjModel) -> np.ndarray:
    """Per degree of freedom, the largest torque (or force) for gravity and acceleration over poses spread across the
    joints' ranges, with _PAYLOAD kg (a gripper and what it holds) shared among the chain's ends."""
    leaves = [b for b in range(1, m.nbody) if not any(m.body_parentid[c] == b for c in range(1, m.nbody))]
    # The ends of the longest chain: the tool end, not a camera mount or a cover on the base.
    depth = {b: len(_chain(m, b)) for b in leaves}
    leaves = [b for b in leaves if depth[b] == max(depth.values())]
    mass = m.body_mass.copy()
    m.body_mass[leaves] += _PAYLOAD / len(leaves)
    try:
        return _worst_bias(m)
    finally:
        m.body_mass[:] = mass


def _worst_bias(m: mujoco.MjModel) -> np.ndarray:
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    out = np.zeros(m.nv)
    M = np.zeros((m.nv, m.nv))
    for i in range(64):
        reset_data(m, d)
        if i:
            for j in range(m.njnt):
                if m.jnt_type[j] in (_HINGE, _SLIDE):
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
                    d.qpos[m.jnt_qposadr[j]] = rng.uniform(max(lo, -np.pi), min(hi, np.pi))
        mujoco.mj_forward(m, d)
        try:
            mujoco.mj_fullM(m, d, M)
        except TypeError:
            mujoco.mj_fullM(m, M, d.qM)
        # Holding against gravity, and accelerating the load (_ACCEL) on top: a vertical axis
        # carries no gravity at all, but still has an arm to swing.
        out = np.maximum(out, np.abs(d.qfrc_bias) + _ACCEL * np.diag(M))
    return out
