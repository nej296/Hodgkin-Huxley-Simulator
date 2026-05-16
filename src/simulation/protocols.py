"""Injected-current protocols for neuron simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class CurrentProtocol(Protocol):
    """Interface for objects that provide injected current over time."""

    def evaluate(self, time_ms: float | np.ndarray) -> float | np.ndarray:
        """Return injected current density in uA/cm^2 for one or more times."""


@dataclass(frozen=True)
class ConstantCurrent:
    """Constant injected current protocol."""

    amplitude: float = 0.0

    def evaluate(self, time_ms: float | np.ndarray) -> float | np.ndarray:
        """Return the same current density at every time point."""

        if np.isscalar(time_ms):
            return float(self.amplitude)
        return np.full_like(np.asarray(time_ms, dtype=float), self.amplitude, dtype=float)


@dataclass(frozen=True)
class StepCurrent:
    """Rectangular current step with optional baseline current.

    Attributes:
        amplitude: Step amplitude in uA/cm^2.
        start_ms: Time when the step turns on.
        end_ms: Time when the step turns off.
        baseline: Baseline current outside the step window in uA/cm^2.
    """

    amplitude: float
    start_ms: float
    end_ms: float
    baseline: float = 0.0

    def __post_init__(self) -> None:
        """Validate temporal ordering of the current step."""

        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms.")

    def evaluate(self, time_ms: float | np.ndarray) -> float | np.ndarray:
        """Return baseline plus step current for scalar or vector time input."""

        time_array = np.asarray(time_ms, dtype=float)
        in_window = (time_array >= self.start_ms) & (time_array <= self.end_ms)
        current = np.where(in_window, self.baseline + self.amplitude, self.baseline)
        if np.isscalar(time_ms):
            return float(current)
        return current.astype(float)
