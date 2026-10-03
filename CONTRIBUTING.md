# Contributing

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev,drake]"
pytest tests examples -n 4 --rw-backend mujoco,pybullet            # the core, on the SO-101
pytest tests/test_conformance.py tests/test_legged.py --rw-robot all,legged --rw-backend drake
ruff check . && ruff format --check .
python bench/run.py --quick      # framework benchmarks; regenerates BENCHMARKS.md
python bench/matrix.py           # robot x engine matrix; regenerates MATRIX.md
```

Engines are heavy: a Genesis worker takes ~4.5 GB, so keep `-n` small for it.

**Adding a robot:** register a `RobotModel` in `src/robowright/robots/__init__.py`
(MJCF path, arm joints, hand, finger bodies, gripper actuator, base position), then run
the contract on every engine: `pytest tests/test_conformance.py --rw-robot <name>
--rw-backend mujoco,pybullet,drake,genesis`. Don't tune a backend for one robot; if an
engine genuinely disagrees, add the case to `conftest.py` with the measured reason.

**Adding an engine:** implement `backends/base.py`'s `Backend` (physics only) and make
`tests/test_conformance.py` and `tests/test_legged.py` pass for every robot.

Ground rules:

- A test means the same thing on every backend. Kinematics, waiting, assertions and
  tracing live in the core; a backend only does physics (see `backends/base.py`).
- Matchers report what they saw, not just that they failed.
- Anything that changes the world during a test (edits, faults) must be recorded in the
  trace, or replay and codegen silently drift. `tests/test_trace_replay_codegen.py`
  checks bit-identical replay and codegen round trips; keep them green.
- Benchmark numbers in docs come from `bench/run.py` and `bench/matrix.py`, never by hand.
