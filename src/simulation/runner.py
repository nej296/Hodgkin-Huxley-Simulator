"""Numerical integration loop for Hodgkin-Huxley simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyState
from src.simulation.config import SimulationConfig
from src.simulation.protocols import ConductanceSchedule, ConstantCurrent, CurrentProtocol


@dataclass(frozen=True)
class SimulationResult:
    """Time-series output from a single-compartment simulation."""

    time_ms: np.ndarray
    voltage_mV: np.ndarray
    m: np.ndarray
    h: np.ndarray
    n: np.ndarray
    injected_current_uA_cm2: np.ndarray
    g_na_max_mS_cm2: np.ndarray
    g_k_max_mS_cm2: np.ndarray
    sodium_conductance_mS_cm2: np.ndarray
    potassium_conductance_mS_cm2: np.ndarray
    sodium_current_uA_cm2: np.ndarray
    potassium_current_uA_cm2: np.ndarray
    leak_current_uA_cm2: np.ndarray
    net_ionic_current_uA_cm2: np.ndarray

    @property
    def state_matrix(self) -> np.ndarray:
        """Return state variables as an array with columns [V, m, h, n]."""

        return np.column_stack((self.voltage_mV, self.m, self.h, self.n))


def _clip_gate_probabilities(state_vector: np.ndarray) -> np.ndarray:
    """Keep gating variables in their physical probability range."""

    clipped = state_vector.copy()
    clipped[1:] = np.clip(clipped[1:], 0.0, 1.0)
    return clipped


def _rk4_step(
    neuron: HodgkinHuxleyNeuron,
    state_vector: np.ndarray,
    injected_current: float,
    dt_ms: float,
    g_na: float,
    g_k: float,
) -> np.ndarray:
    """Advance the state with fourth-order Runge-Kutta integration.

    The injected current is held constant over the step. This is appropriate for
    piecewise-constant current protocols and keeps the integration loop simple.
    """

    k1 = neuron.derivatives_array(state_vector, injected_current, g_na=g_na, g_k=g_k)
    k2 = neuron.derivatives_array(
        state_vector + 0.5 * dt_ms * k1,
        injected_current,
        g_na=g_na,
        g_k=g_k,
    )
    k3 = neuron.derivatives_array(
        state_vector + 0.5 * dt_ms * k2,
        injected_current,
        g_na=g_na,
        g_k=g_k,
    )
    k4 = neuron.derivatives_array(
        state_vector + dt_ms * k3,
        injected_current,
        g_na=g_na,
        g_k=g_k,
    )
    return state_vector + (dt_ms / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    neuron: HodgkinHuxleyNeuron,
    config: SimulationConfig,
    current_protocol: CurrentProtocol | None = None,
    initial_state: HodgkinHuxleyState | None = None,
    g_na_schedule: ConductanceSchedule | None = None,
    g_k_schedule: ConductanceSchedule | None = None,
) -> SimulationResult:
    """Run a fixed-step Hodgkin-Huxley simulation.

    Args:
        neuron: Conductance-based neuron model.
        config: Duration, step size, and numerical method.
        current_protocol: Injected-current protocol. Defaults to zero current.
        initial_state: Optional initial voltage and gate values. When omitted,
            gates are initialized at steady state for config.resting_voltage_mV.
        g_na_schedule: Optional time-varying maximum sodium conductance.
        g_k_schedule: Optional time-varying maximum potassium conductance.
    """

    protocol = current_protocol or ConstantCurrent(0.0)
    state = initial_state or HodgkinHuxleyState.from_voltage(
        neuron,
        voltage=config.resting_voltage_mV,
    )

    time_ms = np.arange(0.0, config.duration_ms + config.dt_ms, config.dt_ms)
    injected_current = np.asarray(protocol.evaluate(time_ms), dtype=float)
    g_na_protocol = g_na_schedule or ConductanceSchedule(neuron.parameters.g_na)
    g_k_protocol = g_k_schedule or ConductanceSchedule(neuron.parameters.g_k)
    g_na_max = np.asarray(g_na_protocol.evaluate(time_ms), dtype=float)
    g_k_max = np.asarray(g_k_protocol.evaluate(time_ms), dtype=float)
    voltage = np.empty_like(time_ms)
    m = np.empty_like(time_ms)
    h = np.empty_like(time_ms)
    n = np.empty_like(time_ms)

    state_vector = np.array([state.voltage, state.m, state.h, state.n], dtype=float)
    voltage[0], m[0], h[0], n[0] = state_vector

    for index in range(1, len(time_ms)):
        state_vector = _rk4_step(
            neuron=neuron,
            state_vector=state_vector,
            injected_current=float(injected_current[index - 1]),
            dt_ms=config.dt_ms,
            g_na=float(g_na_max[index - 1]),
            g_k=float(g_k_max[index - 1]),
        )
        state_vector = _clip_gate_probabilities(state_vector)
        voltage[index], m[index], h[index], n[index] = state_vector

    sodium_conductance = g_na_max * (m**3) * h
    potassium_conductance = g_k_max * (n**4)
    sodium_current = sodium_conductance * (voltage - neuron.parameters.e_na)
    potassium_current = potassium_conductance * (voltage - neuron.parameters.e_k)
    leak_current = neuron.parameters.g_l * (voltage - neuron.parameters.e_l)
    net_ionic_current = sodium_current + potassium_current + leak_current

    return SimulationResult(
        time_ms=time_ms,
        voltage_mV=voltage,
        m=m,
        h=h,
        n=n,
        injected_current_uA_cm2=injected_current,
        g_na_max_mS_cm2=g_na_max,
        g_k_max_mS_cm2=g_k_max,
        sodium_conductance_mS_cm2=sodium_conductance,
        potassium_conductance_mS_cm2=potassium_conductance,
        sodium_current_uA_cm2=sodium_current,
        potassium_current_uA_cm2=potassium_current,
        leak_current_uA_cm2=leak_current,
        net_ionic_current_uA_cm2=net_ionic_current,
    )
