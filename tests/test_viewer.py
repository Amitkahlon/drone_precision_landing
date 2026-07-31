"""The mjpython handover that lets viewer commands work on macOS.

os.execv is always stubbed here: a real handover would replace the pytest
process.
"""

import pytest

from drone_landing.simulation import viewer as viewer_module
from drone_landing.simulation.viewer import (
    ensure_viewer_thread,
    relaunch_arguments,
    viewer_needs_mjpython,
)


@pytest.fixture
def captured_exec(monkeypatch):
    """Record what ensure_viewer_thread would have handed to execv.

    Also stubs __main__ to a plain object so these cases are not affected by
    whether pytest itself was started as a console script or via -m.
    """
    calls = []
    monkeypatch.setattr(viewer_module.os, "execv", lambda path, args: calls.append((path, args)))
    monkeypatch.setattr(viewer_module.shutil, "which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setitem(viewer_module.sys.modules, "__main__", object())
    monkeypatch.delenv(viewer_module._RELAUNCH_ENV, raising=False)
    return calls


def test_no_handover_needed_off_macos(monkeypatch):
    monkeypatch.setattr(viewer_module.sys, "platform", "linux")

    assert viewer_needs_mjpython() is False


def test_handover_needed_on_macos_without_the_launcher(monkeypatch):
    monkeypatch.setattr(viewer_module.sys, "platform", "darwin")
    monkeypatch.setattr(viewer_module.mujoco.viewer, "_MJPYTHON", None, raising=False)

    assert viewer_needs_mjpython() is True


def test_no_handover_needed_once_running_under_the_launcher(monkeypatch):
    monkeypatch.setattr(viewer_module.sys, "platform", "darwin")
    monkeypatch.setattr(viewer_module.mujoco.viewer, "_MJPYTHON", object(), raising=False)

    assert viewer_needs_mjpython() is False


def test_ensure_is_a_noop_when_the_viewer_already_works(monkeypatch, captured_exec):
    monkeypatch.setattr(viewer_module, "viewer_needs_mjpython", lambda: False)

    ensure_viewer_thread()

    assert captured_exec == []


def test_ensure_hands_over_to_mjpython(monkeypatch, captured_exec):
    monkeypatch.setattr(viewer_module, "viewer_needs_mjpython", lambda: True)
    monkeypatch.setattr(viewer_module.sys, "argv", ["/venv/bin/drone-fly", "random"])

    ensure_viewer_thread()

    assert captured_exec == [
        ("/fake/bin/mjpython", ["/fake/bin/mjpython", "/venv/bin/drone-fly", "random"]),
    ]


def test_ensure_marks_the_child_so_it_cannot_loop(monkeypatch, captured_exec):
    monkeypatch.setattr(viewer_module, "viewer_needs_mjpython", lambda: True)
    monkeypatch.setattr(viewer_module.sys, "argv", ["drone-fly"])

    ensure_viewer_thread()

    assert viewer_module.os.environ[viewer_module._RELAUNCH_ENV] == "1"


def test_ensure_refuses_to_relaunch_twice(monkeypatch, captured_exec):
    monkeypatch.setattr(viewer_module, "viewer_needs_mjpython", lambda: True)
    monkeypatch.setenv(viewer_module._RELAUNCH_ENV, "1")

    with pytest.raises(RuntimeError, match="re-ran under mjpython"):
        ensure_viewer_thread()

    assert captured_exec == []


def test_ensure_reports_a_missing_launcher(monkeypatch, captured_exec):
    monkeypatch.setattr(viewer_module, "viewer_needs_mjpython", lambda: True)
    monkeypatch.setattr(viewer_module.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="needs mjpython"):
        ensure_viewer_thread()

    assert captured_exec == []


def test_relaunch_arguments_use_the_script_path_for_scripts(monkeypatch):
    monkeypatch.setattr(viewer_module.sys, "argv", ["demos/yaw_rotation.py", "--fast"])
    monkeypatch.setitem(viewer_module.sys.modules, "__main__", object())

    assert relaunch_arguments() == ["demos/yaw_rotation.py", "--fast"]


def test_relaunch_arguments_prefer_the_module_form(monkeypatch):
    class MainWithSpec:
        __spec__ = type("Spec", (), {"name": "drone_landing.cli.fly"})()

    monkeypatch.setattr(viewer_module.sys, "argv", ["ignored", "random", "--seed", "1"])
    monkeypatch.setitem(viewer_module.sys.modules, "__main__", MainWithSpec())

    assert relaunch_arguments() == ["-m", "drone_landing.cli.fly", "random", "--seed", "1"]
