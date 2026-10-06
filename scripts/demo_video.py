"""robowright's demo video: real runs, recorded as traces and drawn afterwards.

    python scripts/demo_video.py                 # record what is missing, then render demo.mp4
    python scripts/demo_video.py --only record   # just record (e.g. the Isaac Sim run, on a GPU machine)

Every clip is an ordinary robowright run: the same calls a test makes, traced (state only) and
redrawn here with MuJoCo's renderer, a moving camera and captions. Runs on other engines are
drawn the same way, so the picture shows what each engine simulated, not how it renders.
Traces are cached in ~/.cache/robowright-demo/traces; delete one to record it again.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

import robowright as rw  # noqa: E402
from robowright.backends.base import create  # noqa: E402
from robowright.scene import ObjectSpec, tabletop  # noqa: E402
from robowright.trace import Trace  # noqa: E402

W, H, FPS = 1920, 1080, 30
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "robowright-demo"
TRACES = CACHE / "traces"
DESCRIPTIONS = Path.home() / ".cache" / "robot_descriptions"
INK, MUTED, ACCENT, BG = (245, 247, 250), (176, 186, 200), (255, 122, 69), (16, 20, 28)


# --- recording --------------------------------------------------------------------------------
def _pick_and_place(w):
    r = w.robot
    r.arm.home()
    r.pick(w.scene["cube"])
    r.place(on=w.scene["bin"])
    r.arm.home()


def _side_grasp(w):
    r = w.robot
    r.arm.home()
    r.pick(w.scene["can"], approach="side")
    r.place(on=w.scene["bin"])
    r.arm.home()


def _shove(w):
    r = w.robot
    w.wait(0.6)
    r.crouch(0.4)
    r.stand()
    w.wait(0.3)
    w.faults.push("robot", force=(0.0, 0.3 * r.total_mass * 9.81, 0.0), duration=0.1)
    w.wait(2.0)


def _fumble(w):
    """A run that fails: a shove knocks the cube out of the grip, and the check after the place says so."""
    r = w.robot
    r.arm.home()
    r.pick(w.scene["cube"])
    w.faults.push("cube", force=(0.0, 2.0, 0.0), duration=0.1)
    try:
        r.place(on=w.scene["bin"])
        rw.expect(w.scene["cube"]).to_be_inside(w.scene["bin"])
    except rw.RobowrightError as e:
        return f"{type(e).__name__}: {e}"
    return None


SIDE_SCENE = tabletop(
    ObjectSpec("can", "cylinder", (0.02, 0.05), (0.22, -0.06, None), color="red"),
    ObjectSpec("bin", "bin", (0.05, 0.05, 0.035), (0.2, 0.12, 0.0), color="blue", mass=0.0),
)

ARMS = ["so101", "panda", "ur5e", "xarm7", "piper", "vx300s"]
FILES = {
    "Fanuc M-710iC  ·  URDF": DESCRIPTIONS / "fanuc_m710ic_description/urdf/m710ic70.urdf",
    "UR5e  ·  xacro": f"{DESCRIPTIONS}/ur_description/urdf/ur.urdf.xacro?ur_type=ur5e&name=ur5e",
    "PAL TIAGo  ·  MJCF, mobile base": DESCRIPTIONS / "mujoco_menagerie/pal_tiago/tiago_position.xml",
}
ENGINES = ["mujoco", "pybullet", "drake", "genesis", "isaac"]


def clips() -> dict[str, tuple]:
    """name -> (robot, backend, scene or None, run)"""
    out = {f"arm-{r}": (r, "mujoco", None, _pick_and_place) for r in ARMS}
    out |= {f"file-{i}": (str(p), "mujoco", None, _pick_and_place) for i, p in enumerate(FILES.values())}
    out |= {f"engine-{e}": ("panda", e, None, _pick_and_place) for e in ENGINES}
    out["side"] = ("panda", "mujoco", SIDE_SCENE, _side_grasp)
    out |= {f"legged-{r}": (r, "mujoco", None, _shove) for r in ("go2", "spot", "g1")}
    out["fail"] = ("wx250s", "mujoco", None, _fumble)
    return out


def record(only_backend: str | None = None) -> None:
    TRACES.mkdir(parents=True, exist_ok=True)
    for name, (robot, backend, scene, run) in clips().items():
        path = TRACES / f"{name}.zip"
        if path.exists() or (only_backend and backend != only_backend):
            continue
        from robowright.backends.base import available

        if backend not in available():
            print(f"skip {name}: no {backend} here")
            continue
        print(f"record {name}")
        spec = dataclasses.replace(scene, robot=robot) if scene is not None else None
        w = rw.launch(scene=spec, robot=robot, backend=backend, name=name, settings=rw.Settings(trace="on", trace_dir=str(TRACES)))
        try:
            msg = run(w)
            if msg:
                (TRACES / f"{name}.txt").write_text(msg)
        finally:
            w.close(trace_path=path)


# --- drawing ----------------------------------------------------------------------------------
class Stage:
    """A recorded run, posed at any time and drawn from a named or a free camera."""

    def __init__(self, name: str, width: int, height: int):
        self.trace = tr = Trace(TRACES / f"{name}.zip")
        self.scene = tr.scene()
        self.b = create("mujoco", self.scene)
        m = self.b.model
        m.vis.global_.offwidth, m.vis.global_.offheight = max(width, 640), max(height, 480)
        m.vis.quality.offsamples = 8
        self.r = mujoco.Renderer(m, height, width)
        self.t = tr.arrays["t"]
        self.duration = float(self.t[-1])
        self._movable = [(i, n) for i, n in enumerate(tr.meta["object_names"]) if not self.scene.object(n).static]

    def pose(self, t: float) -> None:
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        a, b = self.trace.arrays, self.b
        if "base" in a:
            b.set_base_pose(a["base"][i][:3], a["base"][i][3:7])
        b.set_joint_positions(a["qpos"][i])
        for k, name in self._movable:
            b.set_object_pose(name, a["obj_pos"][i][k], a["obj_quat"][i][k])

    def draw(self, t: float, camera=None) -> Image.Image:
        """``camera``: None for the scene's own, or (lookat, distance, azimuth, elevation)."""
        self.pose(t)
        if camera is None:
            self.r.update_scene(self.b.data, camera=self.scene.cameras[0].name)
        else:
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = camera
            self.r.update_scene(self.b.data, camera=cam)
        return Image.fromarray(self.r.render())


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    for path in (
        CACHE / f"Inter-{weight}.ttf",
        Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if weight in ('Bold', 'SemiBold') else ''}.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def caption(img: Image.Image, title: str, sub: str, t: float, dur: float) -> Image.Image:
    """A title and a line under it, lower left, fading in and out with the clip."""
    a = ease(t / 0.5) * ease((dur - t) / 0.4)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    grad = Image.linear_gradient("L").resize((img.width, 360)).point(lambda v: int(v * 0.75 * a))
    layer.paste((8, 10, 16, 255), (0, img.height - 360), grad)
    ft, fs = font("SemiBold", 62), font("Regular", 32)
    x, y = 96, img.height - 190
    d.rectangle((x, y - 26, x + 64, y - 20), fill=(*ACCENT, int(255 * a)))
    d.text((x, y), title, font=ft, fill=(*INK, int(255 * a)))
    d.text((x, y + 82), sub, font=fs, fill=(*MUTED, int(255 * a)))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def label(img: Image.Image, text: str, xy=(24, 24)) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    f = font("Medium", 26)
    w = d.textlength(text, font=f)
    x, y = xy
    d.rounded_rectangle((x, y, x + w + 32, y + 46), radius=23, fill=(12, 16, 24, 190))
    d.text((x + 16, y + 8), text, font=f, fill=INK)


def grid(stages, names, t, speed, cols, rows, cams=None) -> Image.Image:
    tw, th = W // cols, H // rows
    out = Image.new("RGB", (W, H), BG)
    for k, (s, n) in enumerate(zip(stages, names)):
        img = s.draw(min(t * speed, s.duration), cams[k] if cams else None)
        label(img, n)
        out.paste(img, ((k % cols) * tw, (k // cols) * th))
    d = ImageDraw.Draw(out)
    for c in range(1, cols):
        d.line(((c * tw, 0), (c * tw, H)), fill=BG, width=4)
    for r in range(1, rows):
        d.line(((0, r * th), (W, r * th)), fill=BG, width=4)
    return out


# --- the cut ----------------------------------------------------------------------------------
def segments():
    """(seconds, frame(t) -> Image) for each part of the video, in order."""
    titles = {
        "so101": "SO-101",
        "panda": "Franka Panda",
        "ur5e": "UR5e",
        "xarm7": "xArm 7",
        "piper": "AgileX PiPER",
        "vx300s": "ViperX 300",
    }

    hero = Stage("side", W, H)

    def intro(t):
        img = hero.draw(2.0 + t * 0.6, ((-0.05, 0.0, 0.3), 1.9, 140 + 6 * t, -16))
        img = img.filter(ImageFilter.GaussianBlur(3)).point(lambda v: int(v * 0.45))
        d = ImageDraw.Draw(img)
        a = ease(t / 0.8)
        f1, f2 = font("Bold", 150), font("Regular", 44)
        text = "robowright"
        d.text(((W - d.textlength(text, font=f1)) / 2, 380), text, font=f1, fill=tuple(int(c * a) for c in INK))
        sub = "Playwright, for robots: tests that run on any robot, in any simulator"
        d.text(((W - d.textlength(sub, font=f2)) / 2, 580), sub, font=f2, fill=tuple(int(c * a) for c in MUTED))
        return img

    yield 4.0, intro

    arms = [Stage(f"arm-{r}", W // 3, H // 2) for r in ARMS]
    yield (
        9.0,
        lambda t: caption(
            grid(arms, [titles[r] for r in ARMS], t, 2.2, 3, 2),
            "One test, every robot",
            "robot.pick(cube) · robot.place(on=bin) · expect(cube).to_be_inside(bin)",
            t,
            9.0,
        ),
    )

    files = [Stage(f"file-{i}", W // 3, H) for i in range(len(FILES))]
    # Framed per robot: a 2 m industrial arm, a cobot, a mobile manipulator.
    fcams = [((-0.2, 0.0, 0.7), 3.6, 125, -14), ((0.0, 0.03, 0.25), 1.6, 130, -16), ((-0.1, 0.0, 0.55), 2.6, 125, -14)]
    yield (
        8.0,
        lambda t: caption(
            grid(files, list(FILES), t, 2.0, 3, 1, [(c[0], c[1], c[2] + 2 * t, c[3]) for c in fcams]),
            "Any robot, from its model file",
            "pytest --rw-robot arm.urdf  ·  joints, fingers, motors and mounting worked out from the model",
            t,
            8.0,
        ),
    )

    engines = [Stage(f"engine-{e}", W // 3, H // 2) for e in ENGINES if (TRACES / f"engine-{e}.zip").exists()]
    names = {"mujoco": "MuJoCo", "pybullet": "PyBullet", "drake": "Drake", "genesis": "Genesis", "isaac": "Isaac Sim"}
    engine_names = [names[e] for e in ENGINES if (TRACES / f"engine-{e}.zip").exists()]

    def engines_frame(t):
        img = grid(engines, engine_names, t, 2.2, 3, 2)
        if len(engines) < 6:  # the spare tile says what the grid shows
            d = ImageDraw.Draw(img)
            x0, y0 = (len(engines) % 3) * (W // 3), (len(engines) // 3) * (H // 2)
            d.rectangle((x0, y0, x0 + W // 3, y0 + H // 2), fill=BG)
            for k, line in enumerate(["the same test file,", "the same assertions,", f"{len(engines)} physics engines"]):
                f = font("SemiBold" if k == 2 else "Regular", 44)
                d.text((x0 + 60, y0 + 170 + 64 * k), line, font=f, fill=ACCENT if k == 2 else INK)
        return caption(img, "Same test, every engine", "pytest --rw-backend mujoco,pybullet,drake,genesis,isaac", t, 9.0)

    yield 9.0, engines_frame

    side = Stage("side", W, H)
    yield (
        10.0,
        lambda t: caption(
            side.draw(min(t * 1.6, side.duration), ((0.05, 0.03, 0.2), 1.45 - 0.02 * t, 150 + 4 * t, -16)),
            "Side grasps, planned",
            'robot.pick(can, approach="side")  ·  a collision-free path round the table and the bin',
            t,
            10.0,
        ),
    )

    legged = [Stage(f"legged-{r}", W // 3, H) for r in ("go2", "spot", "g1")]
    yield (
        7.0,
        lambda t: caption(
            grid(legged, ["Unitree Go2", "Boston Dynamics Spot", "Unitree G1"], t, 1.0, 3, 1),
            "Legged robots too",
            "crouch, stand, take a shove  ·  expect(robot.base).always.to_be_upright()",
            t,
            7.0,
        ),
    )

    fail = Stage("fail", W, H)
    message = (TRACES / "fail.txt").read_text() if (TRACES / "fail.txt").exists() else ""

    def failure(t):
        img = fail.draw(min(t * 1.2, fail.duration), ((0.2, -0.04, 0.05), 0.55, 120 + 3 * t, -20))
        if t > 4.0 and message:
            a = ease((t - 4.0) / 0.5)
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            d.rounded_rectangle((96, 120, W - 96, 380), radius=18, fill=(20, 8, 10, int(225 * a)))
            kind, _, rest = message.partition(": ")
            d.text((136, 150), kind, font=font("SemiBold", 40), fill=(255, 110, 110, int(255 * a)))
            words, lines, line = rest.split(), [], ""
            for word in words:
                if len(line) + len(word) > 92:
                    lines.append(line)
                    line = ""
                line += word + " "
            lines.append(line)
            for k, ln in enumerate(lines[:3]):
                d.text((136, 210 + 46 * k), ln, font=ImageFont.truetype(MONO, 30), fill=(*INK, int(255 * a)))
            img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        return caption(img, "When it fails, it says why", "every run traced · replayed bit for bit · turned into a regression test", t, 9.0)

    yield 9.0, failure

    def outro(t):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        a = ease(t / 0.6)
        f1 = font("Bold", 110)
        d.text(((W - d.textlength("robowright", font=f1)) / 2, 300), "robowright", font=f1, fill=tuple(int(c * a) for c in INK))
        lines = [
            "pytest --rw-robot your_arm.urdf --rw-backend mujoco,drake",
            "MJCF · URDF · xacro    MuJoCo · PyBullet · Drake · Genesis · Isaac Sim",
        ]
        for k, ln in enumerate(lines):
            f = ImageFont.truetype(MONO, 40)
            d.text(
                ((W - d.textlength(ln, font=f)) / 2, 500 + 70 * k),
                ln,
                font=f,
                fill=tuple(int(c * a) for c in (ACCENT if k == 0 else MUTED)),
            )
        return img

    yield 4.0, outro


def render(out: Path) -> None:
    import imageio.v2 as imageio

    parts = list(segments())
    fade = 0.4
    with imageio.get_writer(
        out,
        fps=FPS,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"],
    ) as w:
        prev_last = None
        for k, (dur, frame) in enumerate(parts):
            n = int(round(dur * FPS))
            for i in range(n):
                img = frame(i / FPS)
                if prev_last is not None and i < fade * FPS:  # cross-fade from the last part
                    img = Image.blend(prev_last, img, ease(i / (fade * FPS)))
                w.append_data(np.asarray(img))
            prev_last = frame((n - 1) / FPS)
            print(f"part {k + 1}/{len(parts)} done")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="demo.mp4")
    ap.add_argument("--only", choices=["record", "render"])
    ap.add_argument("--backend", help="record only this engine's clips")
    a = ap.parse_args(argv)
    if a.only != "render":
        record(a.backend)
    if a.only != "record":
        render(Path(a.out))


if __name__ == "__main__":
    main()
