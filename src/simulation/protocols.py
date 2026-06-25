"""Injected-current protocols for neuron simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class CurrentPulse:
    """One rectangular injected-current pulse."""

    amplitude: float
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        """Validate temporal ordering for one current pulse."""

        if self.end_ms <= self.start_ms:
            raise ValueError("pulse end_ms must be greater than start_ms.")


@dataclass(frozen=True)
class MultiPulseCurrent:
    """Baseline current plus any number of additive rectangular pulses."""

    baseline: float = 0.0
    pulses: tuple[CurrentPulse, ...] = field(default_factory=tuple)

    def evaluate(self, time_ms: float | np.ndarray) -> float | np.ndarray:
        """Return baseline plus all pulses active at each time point."""

        time_array = np.asarray(time_ms, dtype=float)
        current = np.full_like(time_array, self.baseline, dtype=float)
        for pulse in self.pulses:
            in_window = (time_array >= pulse.start_ms) & (time_array <= pulse.end_ms)
            current = np.where(in_window, current + pulse.amplitude, current)
        if np.isscalar(time_ms):
            return float(current)
        return current.astype(float)


@dataclass(frozen=True)
class ConductanceChange:
    """A maximum-conductance value that applies over a time interval."""

    time_ms: float
    end_ms: float
    value: float

    def __post_init__(self) -> None:
        """Validate one scheduled maximum-conductance interval."""

        if self.time_ms < 0:
            raise ValueError("conductance change time_ms must be non-negative.")
        if self.end_ms <= self.time_ms:
            raise ValueError("conductance change end_ms must be greater than time_ms.")
        if self.value < 0:
            raise ValueError("conductance values must be non-negative.")


@dataclass(frozen=True)
class ConductanceSchedule:
    """Piecewise-constant maximum conductance over simulation time."""

    base_value: float
    changes: tuple[ConductanceChange, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate and require non-negative maximum conductances."""

        if self.base_value < 0:
            raise ValueError("base conductance must be non-negative.")

    def evaluate(self, time_ms: float | np.ndarray) -> float | np.ndarray:
        """Return the maximum conductance active at each time point."""

        time_array = np.asarray(time_ms, dtype=float)
        conductance = np.full_like(time_array, self.base_value, dtype=float)
        for change in sorted(self.changes, key=lambda item: item.time_ms):
            in_window = (time_array >= change.time_ms) & (time_array <= change.end_ms)
            conductance = np.where(in_window, change.value, conductance)
        if np.isscalar(time_ms):
            return float(conductance)
        return conductance.astype(float)
