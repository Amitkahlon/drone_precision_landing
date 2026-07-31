"""Shared plumbing for running a simulation inside MuJoCo's passive viewer."""

import os
import shutil
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager

import mujoco
import mujoco.viewer
import numpy as np

from ..drone import Drone

# Height (m) above the drone at which the state label is drawn.
_LABEL_OFFSET_M = 0.4

# Set on the re-executed process so a failed handover cannot loop forever.
_RELAUNCH_ENV = "DRONE_LANDING_UNDER_MJPYTHON"

StepFn = Callable[[mujoco.viewer.Handle], None]
ContinueFn = Callable[[], bool]


def viewer_needs_mjpython() -> bool:
    """True when this process cannot open a viewer window as it stands.

    macOS requires all UI work on the process's main thread, which the plain
    interpreter is using to run the simulation loop. The `mjpython` launcher
    shipped with MuJoCo reserves the main thread for the viewer and runs the
    script on a second one instead. This reads the same marker object that
    `mujoco.viewer.launch_passive` checks before refusing to open a window.
    """
    if sys.platform != "darwin":
        return False
    return getattr(mujoco.viewer, "_MJPYTHON", None) is None


def ensure_viewer_thread() -> None:
    """Re-run this exact command under mjpython if the viewer needs it.

    Replaces the current process on success, so nothing after the call runs.
    Entry points that open a viewer should call this before building a scene;
    library code should not, since restarting a caller's program would be a
    surprising thing for a function call to do.
    """
    if not viewer_needs_mjpython():
        return

    if os.environ.get(_RELAUNCH_ENV):
        raise RuntimeError(
            "re-ran under mjpython but the viewer still has no UI thread; "
            "invoke mjpython directly to see the underlying error"
        )

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        raise RuntimeError(
            "opening a viewer on macOS needs mjpython, which ships with the mujoco "
            "package but is not on PATH; activate the virtualenv, or run headless"
        )

    # execv discards buffered output, which is lost silently when stdout is a pipe.
    sys.stdout.flush()
    sys.stderr.flush()

    os.environ[_RELAUNCH_ENV] = "1"
    os.execv(mjpython, [mjpython, *relaunch_arguments()])


def relaunch_arguments() -> list[str]:
    """The arguments that re-run this process, as mjpython should receive them.

    Prefers `-m module` when the process was started that way, and otherwise
    passes the script path, which covers both console scripts and the demos.
    """
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return ["-m", spec.name, *sys.argv[1:]]
    return list(sys.argv)


@contextmanager
def paced_step(timestep: float):
    """Stretch the wrapped block out to one simulated timestep of wall-clock time.

    Without this the viewer would replay the run as fast as the machine can
    integrate it, which is far quicker than real time and unwatchable.
    """
    started_at = time.time()
    yield
    remaining = timestep - (time.time() - started_at)
    if remaining > 0:
        time.sleep(remaining)


def run_viewer_loop(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        step: StepFn,
        keep_going: ContinueFn | None = None,
) -> bool:
    """Drive `step` in the passive viewer at wall-clock speed.

    Runs until `keep_going` returns False, or forever if it is None, which is
    what the demos want: they stay open after the drone has settled so the final
    state can be inspected. Returns True if the window was closed first.
    """
    timestep = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while keep_going is None or keep_going():
            if not viewer.is_running():
                return True
            with paced_step(timestep):
                step(viewer)
                viewer.sync()
    return False


def draw_state_label(viewer: mujoco.viewer.Handle, drone: Drone, label: str) -> None:
    """Float `label` above the drone.

    MuJoCo attaches text to geometry rather than to a point, so this draws a
    fully transparent pinhead sphere purely to carry the label.
    """
    viewer.user_scn.ngeom = 1
    geom = viewer.user_scn.geoms[0]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    geom.pos[:] = [drone.get_x(), drone.get_y(), drone.get_z() + _LABEL_OFFSET_M]
    geom.size[:] = [0.01, 0.01, 0.01]
    geom.label = label
