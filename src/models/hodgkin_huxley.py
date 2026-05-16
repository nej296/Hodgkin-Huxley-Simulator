"""Single-compartment Hodgkin-Huxley membrane model.

The implementation follows the classic squid giant axon equations from
Hodgkin and Huxley (1952). Voltages are represented in millivolts, time in
milliseconds, conductances in mS/cm^2, capacitance in uF/cm^2, and current
density in uA/cm^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HodgkinHuxleyParameters:
    """Electrical parameters for a single isopotential HH compartment.

    Attributes:
        membrane_capacitance: Membrane capacitance density in uF/cm^2.
        g_na: Maximum sodium conductance density in mS/cm^2.
        g_k: Maximum potassium conductance density in mS/cm^2.
        g_l: Leak conductance density in mS/cm^2.
        e_na: Sodium reversal potential in mV.
        e_k: Potassium reversal potential in mV.
        e_l: Leak reversal potential in mV.
    """

    membrane_capacitance: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.387

    def __post_init__(self) -> None:
        """Validate that passive and maximum conductance parameters are physical."""

        if self.membrane_capacitance <= 0:
            raise ValueError("membrane_capacitance must be positive.")
        if self.g_na < 0 or self.g_k < 0 or self.g_l < 0:
            raise ValueError("Conductances must be non-negative.")


@dataclass(frozen=True)
class HodgkinHuxleyState:
    """State variables for the membrane voltage and channel gates.

    Attributes:
        voltage: Membrane potential in mV.
        m: Sodium activation gate probability.
        h: Sodium inactivation gate probability.
        n: Potassium activation gate probability.
    """

    voltage: float
    m: float
    h: float
    n: float

    @classmethod
    def from_voltage(
        cls,
        neuron: "HodgkinHuxleyNeuron",
        voltage: float = -65.0,
    ) -> "HodgkinHuxleyState":
        """Initialize gates at their steady-state values for a voltage."""

        m_inf, h_inf, n_inf = neuron.steady_state_gates(voltage)
        return cls(voltage=voltage, m=m_inf, h=h_inf, n=n_inf)


@dataclass(frozen=True)
class HodgkinHuxleyDerivative:
    """Time derivatives of HH state variables.

    Attributes:
        d_voltage: Membrane voltage derivative in mV/ms.
        d_m: Sodium activation gate derivative in 1/ms.
        d_h: Sodium inactivation gate derivative in 1/ms.
        d_n: Potassium activation gate derivative in 1/ms.
    """

    d_voltage: float
    d_m: float
    d_h: float
    d_n: float

    def as_array(self) -> np.ndarray:
        """Return derivatives as a NumPy vector ordered as [V, m, h, n]."""

        return np.array([self.d_voltage, self.d_m, self.d_h, self.d_n], dtype=float)


class HodgkinHuxleyNeuron:
    """Classic conductance-based Hodgkin-Huxley neuron model."""

    def __init__(self, parameters: HodgkinHuxleyParameters | None = None) -> None:
        """Create a neuron with default or user-supplied electrical parameters."""

        self.parameters = parameters or HodgkinHuxleyParameters()

    @staticmethod
    def _vtrap(x: float | np.ndarray, y: float) -> float | np.ndarray:
        """Evaluate x / (1 - exp(-x / y)) safely near x = 0.

        The HH alpha-rate equations contain removable singularities at specific
        voltages. This helper replaces the singular expression with its Taylor
        expansion when the numerator is very small.
        """

        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 0:
            x_value = float(x_array)
            if abs(x_value / y) < 1e-7:
                return y * (1.0 + x_value / (2.0 * y))
            return x_value / (1.0 - np.exp(-x_value / y))

        value = np.empty_like(x_array, dtype=float)
        near_zero = np.abs(x_array / y) < 1e-7
        value[near_zero] = y * (1.0 + x_array[near_zero] / (2.0 * y))
        value[~near_zero] = x_array[~near_zero] / (
            1.0 - np.exp(-x_array[~near_zero] / y)
        )
        return value

    def alpha_m(self, voltage: float | np.ndarray) -> float | np.ndarray:
        """Sodium activation opening rate alpha_m(V) in 1/ms."""

        return 0.1 * self._vtrap(np.asarray(voltage) + 40.0, 10.0)

    def beta_m(self, voltage: float | np.ndarray) -> float | np.ndarray:
        """Sodium activation closing rate beta_m(V) in 1/ms."""

        return 4.0 * np.exp(-(np.asarray(voltage) + 65.0) / 18.0)

    def alpha_h(self, voltage: float | np.ndarray) -> float | np.ndarray:
        """Sodium inactivation recovery rate alpha_h(V) in 1/ms."""

        return 0.07 * np.exp(-(np.asarray(voltage) + 65.0) / 20.0)

    def beta_h(self, voltage: float | np.ndarray) -> float | np.ndarray:
        """Sodium inactivation closing rate beta_h(V) in 1/ms."""

        return 1.0 / (1.0 + np.exp(-(np.asarray(voltage) + 35.0) / 10.0))

    def alpha_n(self, voltage: float | np.ndarray) -> float | np.ndarray:
        """Potassium activation opening rate alpha_n(V) in 1/ms."""

        return 0.01 * self._vtrap(np.asarray(voltage) + 55.0, 10.0)

    def beta_n(self, voltage: float | np.ndarray) -> float | np.ndarray:
        """Potassium activation closing rate beta_n(V) in 1/ms."""

        return 0.125 * np.exp(-(np.asarray(voltage) + 65.0) / 80.0)

    def steady_state_gates(self, voltage: float) -> tuple[float, float, float]:
        """Return steady-state gate probabilities m_inf, h_inf, and n_inf."""

        alpha_m = self.alpha_m(voltage)
        beta_m = self.beta_m(voltage)
        alpha_h = self.alpha_h(voltage)
        beta_h = self.beta_h(voltage)
        alpha_n = self.alpha_n(voltage)
        beta_n = self.beta_n(voltage)

        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        n_inf = alpha_n / (alpha_n + beta_n)
        return float(m_inf), float(h_inf), float(n_inf)

    def ionic_currents(
        self,
        voltage: float,
        m: float,
        h: float,
        n: float,
    ) -> dict[str, float]:
        """Compute sodium, potassium, and leak current densities.

        Current follows the electrophysiology sign convention used in the
        original HH equations: positive ionic current is outward and
        hyperpolarizing. The simulator uses I_injected - I_ionic for dV/dt.
        """

        p = self.parameters
        i_na = p.g_na * (m**3) * h * (voltage - p.e_na)
        i_k = p.g_k * (n**4) * (voltage - p.e_k)
        i_l = p.g_l * (voltage - p.e_l)
        return {"sodium": i_na, "potassium": i_k, "leak": i_l}

    def derivatives(
        self,
        state: HodgkinHuxleyState,
        injected_current: float,
    ) -> HodgkinHuxleyDerivative:
        """Compute HH state derivatives for one time point.

        Args:
            state: Current voltage and gate values.
            injected_current: Applied current density in uA/cm^2. Positive
                injected current depolarizes the membrane.
        """

        currents = self.ionic_currents(state.voltage, state.m, state.h, state.n)
        total_ionic_current = currents["sodium"] + currents["potassium"] + currents["leak"]
        d_voltage = (
            injected_current - total_ionic_current
        ) / self.parameters.membrane_capacitance

        alpha_m = self.alpha_m(state.voltage)
        beta_m = self.beta_m(state.voltage)
        alpha_h = self.alpha_h(state.voltage)
        beta_h = self.beta_h(state.voltage)
        alpha_n = self.alpha_n(state.voltage)
        beta_n = self.beta_n(state.voltage)

        d_m = alpha_m * (1.0 - state.m) - beta_m * state.m
        d_h = alpha_h * (1.0 - state.h) - beta_h * state.h
        d_n = alpha_n * (1.0 - state.n) - beta_n * state.n

        return HodgkinHuxleyDerivative(
            d_voltage=float(d_voltage),
            d_m=float(d_m),
            d_h=float(d_h),
            d_n=float(d_n),
        )

    def derivatives_array(self, state_vector: np.ndarray, injected_current: float) -> np.ndarray:
        """Compute derivatives for an array state ordered as [V, m, h, n]."""

        state = HodgkinHuxleyState(
            voltage=float(state_vector[0]),
            m=float(state_vector[1]),
            h=float(state_vector[2]),
            n=float(state_vector[3]),
        )
        return self.derivatives(state, injected_current).as_array()
