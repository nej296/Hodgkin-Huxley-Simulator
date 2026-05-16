"""Voltage-trace comparison utilities for validation workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TraceComparison:
    """Numerical comparison between a reference and candidate voltage trace."""

    rms_error_mV: float
    max_abs_error_mV: float
    peak_voltage_error_mV: float
    compared_points: int


def compare_voltage_traces(
    reference_time_ms: np.ndarray,
    reference_voltage_mV: np.ndarray,
    candidate_time_ms: np.ndarray,
    candidate_voltage_mV: np.ndarray,
) -> TraceComparison:
    """Compare two voltage traces on the reference time grid.

    This is intended for future validation against NEURON or other simulator
    outputs. Candidate voltage is linearly interpolated onto the reference time
    grid before error metrics are computed.
    """

    if reference_time_ms.shape != reference_voltage_mV.shape:
        raise ValueError("reference time and voltage arrays must have the same shape.")
    if candidate_time_ms.shape != candidate_voltage_mV.shape:
        raise ValueError("candidate time and voltage arrays must have the same shape.")
    if len(reference_time_ms) == 0 or len(candidate_time_ms) == 0:
        raise ValueError("trace arrays must not be empty.")

    candidate_interp = np.interp(reference_time_ms, candidate_time_ms, candidate_voltage_mV)
    error = candidate_interp - reference_voltage_mV

    return TraceComparison(
        rms_error_mV=float(np.sqrt(np.mean(error**2))),
        max_abs_error_mV=float(np.max(np.abs(error))),
        peak_voltage_error_mV=float(np.max(candidate_voltage_mV) - np.max(reference_voltage_mV)),
        compared_points=int(len(reference_time_ms)),
    )
