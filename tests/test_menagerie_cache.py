"""The Menagerie cache is shared by pytest-xdist workers, and one worker's checkout must not be read half-done.

A checkout writes a robot's XML before its meshes (git fetches each mesh as it checks out, under
``--filter=blob:none``), so a worker that found the XML while another was still checking out compiled
the model with meshes missing: CI failed on ANYmal C with ``Error opening file 'assets/top_shell.obj'``.
"""

from __future__ import annotations

import fcntl
import threading
import time

from robowright.robots import menagerie


def test_a_reader_waits_for_a_checkout_in_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # no fallback checkout under ~/.cache/robot_descriptions
    monkeypatch.delenv("ROBOWRIGHT_MENAGERIE", raising=False)
    cache = menagerie.cache_dir()
    (cache / ".git").mkdir(parents=True)
    robot = cache / "some_robot"
    robot.mkdir()

    lock = open(cache.parent / "menagerie.lock", "a")  # noqa: SIM115 - held across the thread below
    fcntl.flock(lock, fcntl.LOCK_EX)  # a checkout is in progress
    (robot / "scene.xml").write_text("<mujoco/>")  # its XML is already there, its meshes are not
    got = []
    reader = threading.Thread(target=lambda: got.append(menagerie.path("some_robot/scene.xml")))
    reader.start()
    time.sleep(0.5)
    assert not got, "the reader returned the model while its checkout was still writing it"

    (robot / "mesh.obj").write_text("")  # the checkout finishes
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    reader.join(timeout=5)
    assert got == [robot / "scene.xml"]
