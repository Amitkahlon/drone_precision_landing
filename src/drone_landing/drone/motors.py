import numpy as np

from ..enums import Motor


class MotorBank:
    """Named read/write access to the four motor thrusts in `data.ctrl`.

    Wraps the raw actuator array so callers say `motors.fl` instead of
    indexing `data.ctrl` and having to remember the rotor ordering.
    """

    def __init__(self, ctrl: np.ndarray):
        object.__setattr__(self, '_ctrl', ctrl)

    @property
    def fl(self) -> float: return float(self._ctrl[Motor.FL])

    @fl.setter
    def fl(self, thrust: float): self._ctrl[Motor.FL] = thrust

    @property
    def fr(self) -> float: return float(self._ctrl[Motor.FR])

    @fr.setter
    def fr(self, thrust: float): self._ctrl[Motor.FR] = thrust

    @property
    def br(self) -> float: return float(self._ctrl[Motor.BR])

    @br.setter
    def br(self, thrust: float): self._ctrl[Motor.BR] = thrust

    @property
    def bl(self) -> float: return float(self._ctrl[Motor.BL])

    @bl.setter
    def bl(self, thrust: float): self._ctrl[Motor.BL] = thrust

    def set_all(self, fl: float, fr: float, br: float, bl: float) -> None:
        self.fl = fl
        self.fr = fr
        self.br = br
        self.bl = bl
