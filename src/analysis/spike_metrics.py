"""Spike detection and voltage-trace summary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.runner import SimulationResult


@dataclass(frozen=True)
class SpikeMetrics:
    """Summary measurements for an action-potential trace."""

    spike_count: int
    firing_rate_hz: float
    peak_voltage_mV: float
    trough_voltage_mV: float
    first_spike_time_ms: float | None
    threshold_crossings_ms: tuple[float, ...]


def detect_spikes(
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
    threshold_mV: float = 0.0,
    refractory_ms: float = 2.0,
) -> tuple[float, ...]:
    """Detect spikes from upward voltage threshold crossings.

    The refractory window prevents multiple crossings during a single action
    potential from being counted as separate spikes.
    """

    if time_ms.shape != voltage_mV.shape:
        raise ValueError("time_ms and voltage_mV must have the same shape.")

    crossings = np.where((voltage_mV[:-1] < threshold_mV) & (voltage_mV[1:] >= threshold_mV))[0]
    spike_times: list[float] = []
    last_spike_time = -np.inf

    for index in crossings:
        crossing_time = float(time_ms[index + 1])
        if crossing_time - last_spike_time >= refractory_ms:
            spike_times.append(crossing_time)
            last_spike_time = crossing_time

    return tuple(spike_times)


def summarize_voltage_trace(
    result: SimulationResult,
    threshold_mV: float = 0.0,
    refractory_ms: float = 2.0,
) -> SpikeMetrics:
    """Compute spike count, firing rate, and voltage extrema for a result."""

    spike_times = detect_spikes(
        time_ms=result.time_ms,
        voltage_mV=result.voltage_mV,
        threshold_mV=threshold_mV,
        refractory_ms=refractory_ms,
    )
    duration_seconds = (result.time_ms[-1] - result.time_ms[0]) / 1000.0
    firing_rate_hz = len(spike_times) / duration_seconds if duration_seconds > 0 else 0.0

    return SpikeMetrics(
        spike_count=len(spike_times),
        firing_rate_hz=float(firing_rate_hz),
        peak_voltage_mV=float(np.max(result.voltage_mV)),
        trough_voltage_mV=float(np.min(result.voltage_mV)),
        first_spike_time_ms=spike_times[0] if spike_times else None,
        threshold_crossings_ms=spike_times,
    )
