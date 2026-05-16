"""Tests for the core Hodgkin-Huxley simulator."""

from __future__ import annotations

import numpy as np

from src.analysis.spike_metrics import summarize_voltage_trace
from src.models.hodgkin_huxley import HodgkinHuxleyNeuron
from src.simulation.config import SimulationConfig
from src.simulation.protocols import StepCurrent
from src.simulation.runner import simulate


def test_rate_functions_are_finite_at_singular_voltages() -> None:
    """Alpha-rate singularities should be numerically well behaved."""

    neuron = HodgkinHuxleyNeuron()

    assert np.isfinite(neuron.alpha_m(-40.0))
    assert np.isfinite(neuron.alpha_n(-55.0))


def test_current_step_evokes_action_potential() -> None:
    """A 10 uA/cm^2 current step should produce spikes in the default model."""

    neuron = HodgkinHuxleyNeuron()
    config = SimulationConfig(duration_ms=50.0, dt_ms=0.025, method="rk4")
    current = StepCurrent(amplitude=10.0, start_ms=5.0, end_ms=35.0)

    result = simulate(neuron=neuron, config=config, current_protocol=current)
    metrics = summarize_voltage_trace(result)

    assert metrics.spike_count >= 1
    assert metrics.peak_voltage_mV > 20.0
    assert np.all((result.m >= 0.0) & (result.m <= 1.0))
    assert np.all((result.h >= 0.0) & (result.h <= 1.0))
    assert np.all((result.n >= 0.0) & (result.n <= 1.0))


def test_step_current_protocol_window() -> None:
    """StepCurrent should return baseline outside and amplitude inside the window."""

    protocol = StepCurrent(amplitude=4.0, start_ms=1.0, end_ms=3.0, baseline=0.5)
    samples = protocol.evaluate(np.array([0.0, 1.0, 2.0, 3.1]))

    np.testing.assert_allclose(samples, np.array([0.5, 4.5, 4.5, 0.5]))
