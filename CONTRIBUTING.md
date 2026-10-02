# Contributing

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
pytest tests examples -n auto --rw-backend mujoco,pybullet
ruff check . && ruff format --check .
python bench/run.py --quick      # benchmarks; regenerates BENCHMARKS.md
```

Ground rules:

- A test means the same thing on every backend. Kinematics, waiting, assertions and
  tracing live in the core; a backend only does physics (see `backends/base.py`).
- Matchers report what they saw, not just that they failed.
- Anything that changes the world during a test (edits, faults) must be recorded in the
  trace, or replay and codegen silently drift. `tests/test_trace_replay_codegen.py`
  checks bit-identical replay and codegen round trips; keep them green.
- Benchmarks numbers in docs come from `bench/run.py`, never by hand.
