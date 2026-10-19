"""rw_attach, Playwright's testInfo.attach: files a test adds to its entry in the HTML report."""

TEST = """
def test_with_attachments(world, rw_attach, tmp_path):
    rw_attach("notes.txt", "the cube went left")
    log = tmp_path / "stack.log"
    log.write_text("[INFO] controller up")
    rw_attach("stack.log", path=log)
    rw_attach.camera(world, name="start.png")
    world.step(5)
    assert False, "fails, so the report opens its details"
"""


def test_attachments_are_in_the_report(pytester):
    pytester.makepyfile(test_att=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-report", "out/report.html")
    r.assert_outcomes(failed=1)
    att = pytester.path / "robowright-traces" / "attachments" / "test_att.py__test_with_attachments[mujoco]"
    assert (att / "notes.txt").read_text() == "the cube went left"
    assert (att / "stack.log").read_text() == "[INFO] controller up"
    assert (att / "start.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    page = (pytester.path / "out" / "report.html").read_text()
    # The report keeps its own copy beside it (report-traces/attachments), so it is shared as one folder.
    src = page.split('<img src="', 1)[1].split('"', 1)[0]
    assert src.startswith("report-traces/attachments/") and src.endswith("start.png")
    assert (pytester.path / "out" / src).read_bytes() == (att / "start.png").read_bytes()
    assert ">notes.txt</a>" in page and ">stack.log</a>" in page
