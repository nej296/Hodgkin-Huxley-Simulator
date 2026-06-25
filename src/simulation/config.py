"""Simulation configuration objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical integration settings for a membrane simulation.

    Attributes:
        duration_ms: Total simulated time in milliseconds.
        dt_ms: Fixed integration time step in milliseconds.
        method: Explicit integration method. Only fourth-order Runge-Kutta is
            supported in the app to avoid ambiguous method choices.
        resting_voltage_mV: Voltage used to initialize gate steady states when
            an explicit initial state is not supplied.
    """

    duration_ms: float = 50.0
    dt_ms: float = 0.01
    method: str = "rk4"
    resting_voltage_mV: float = -65.0

    def __post_init__(self) -> None:
        """Validate simulation settings before running an experiment."""

        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive.")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive.")
        if self.dt_ms > self.duration_ms:
            raise ValueError("dt_ms must not exceed duration_ms.")
        if self.method != "rk4":
            raise ValueError("method must be 'rk4'.")
