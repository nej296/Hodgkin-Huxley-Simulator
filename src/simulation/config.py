"""Simulation configuration objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical integration settings for a membrane simulation.

    Attributes:
        duration_ms: Total simulated time in milliseconds.
        dt_ms: Fixed integration time step in milliseconds.
        method: Explicit integration method, either "rk4" or "euler".
    """

    duration_ms: float = 50.0
    dt_ms: float = 0.01
    method: str = "rk4"

    def __post_init__(self) -> None:
        """Validate simulation settings before running an experiment."""

        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive.")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive.")
        if self.dt_ms > self.duration_ms:
            raise ValueError("dt_ms must not exceed duration_ms.")
        if self.method not in {"rk4", "euler"}:
            raise ValueError("method must be either 'rk4' or 'euler'.")
