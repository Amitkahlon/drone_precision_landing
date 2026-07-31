"""Shared plumbing for running a simulation inside MuJoCo's passive viewer."""

import time
from collections.abc import Callable
from contextlib import contextmanager

import mujoco
import mujoco.viewer
import numpy as np

from ..drone import Drone

# Height (m) above the drone at which the state label is drawn.
_LABEL_OFFSET_M = 0.4

StepFn = Callable[[mujoco.viewer.Handle], None]
ContinueFn = Callable[[], bool]


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
