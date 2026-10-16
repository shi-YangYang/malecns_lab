"""Drone interface. Every backend speaks normalised FlightCommands."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..motor.command import FlightCommand
from ..safety import Telemetry


class Drone(ABC):
    name = "drone"
    has_camera = False

    def connect(self) -> None:  # noqa: B027 - optional hook
        pass

    @abstractmethod
    def takeoff(self) -> None: ...

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def send(self, cmd: FlightCommand) -> None: ...

    @abstractmethod
    def telemetry(self) -> Telemetry: ...

    def frame(self) -> np.ndarray | None:
        """Latest camera frame as BGR uint8, or None if the drone has no camera."""
        return None

    def emergency_stop(self) -> None:
        """Cut motors immediately. Drone WILL fall. Default: land."""
        self.land()

    def close(self) -> None:  # noqa: B027 - optional hook
        pass
