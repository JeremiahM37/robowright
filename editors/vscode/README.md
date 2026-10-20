# robowright for VS Code

Robot tests in the editor, as Playwright's extension has them for web tests:

- **Test Explorer.** robowright's tests appear under *robowright*, by file. Run all of them, a file or one test.
  Each result shows as the test finishes. Tests are found again when a test file, `conftest.py` or the settings change.
- **Failures on their line.** A failed expectation shows where it failed, with the message robowright wrote (what was
  expected, what the world held instead) and a **Show trace** link.
- **Traces in a tab.** *Show trace* opens the trace viewer in an editor tab: the action timeline, the camera drawn
  from the recorded state, joints and contacts at every step. Every run records one, for passing tests too. Any trace
  `.zip` opens from the Explorer's context menu, or with *robowright: Show trace*.
- **Live view.** The *Run with live view* profile opens the camera in your browser as soon as a test builds its world.
  Tests then run at real-time pace.
- **Debug.** The *Debug* profile (the bug icon on a test) runs it under the Python Debugger, so a breakpoint in the
  test, a fixture or robowright itself stops there with the world's state in the Variables view. One process (any
  `-n` is dropped) and no `--rw-timeout` while stopped. Needs the Python Debugger extension (`ms-python.debugpy`).
- **Watch.** The eye button on a run profile (Playwright's watch mode): the chosen tests run again each time a Python
  file is saved, until you stop it.
- **Record a new test.** *robowright: Record a new test* (the record button on the Testing view) opens a page where
  pick, place, the gripper, waits and checks run on the robot as you choose them and are written as a test; *Save*
  writes `tests/test_recorded.py`, which opens here and joins the Test Explorer (Playwright's *Record new*).
- **Pick locator.** *robowright: Pick locator* opens the robot's camera view: click an object, then *Use* on one of
  the locators that find it (`scene["cube"]`, `scene.get(color="red")`, ...), and it is typed at your cursor.
- **UI mode.** *robowright: Open the UI* runs `robowright test --ui` in a terminal.

## Settings

| setting | what it does |
| --- | --- |
| `robowright.python` | the Python with robowright installed (default: the Python extension's interpreter, else `python3`) |
| `robowright.pytestArgs` | arguments every run gets, as on the command line: test paths, `--rw-robot panda,ur5e`, `--rw-backend mujoco,drake` |

## Install

The extension is not on the Marketplace yet. Build it and install the file:

```bash
cd editors/vscode
npx @vscode/vsce package          # robowright-0.1.0.vsix
code --install-extension robowright-0.1.0.vsix
```

## How it works

Each list and run is a fresh `python -m pytest` with robowright's events plugin (`-p robowright.ui_events`), the one
`robowright test --ui` uses: it writes each test's start, outcome, message and trace as JSON lines, and the extension
reads them as they come. Runs are pinned to the rootdir that listing found, so a test run alone reports under the
same id.

## Tests

`npm test` runs the extension inside a real VS Code (downloaded by `@vscode/test-electron`) on a small project of robot
tests. It needs a display: `xvfb-run -a npm test` on a server. It installs the Python Debugger into
`.vscode-test/extensions-debugpy` for the Debug profile's test (`ROBOWRIGHT_TEST_NO_DEBUGPY=1` runs without it).
`ROBOWRIGHT_PYTHON` names the Python with robowright (default: the repository's `.venv`).
