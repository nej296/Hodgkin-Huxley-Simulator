"""Multi-compartment cable model built on the Hodgkin-Huxley membrane.

This extends the single-compartment HH model to a spatially-extended neuron by
discretising the cable equation into connected compartments (the standard
method-of-lines / compartmental approach). Every compartment carries the *same*
HH channel kinetics as the point model -- the membrane maths is reused verbatim
from :class:`~src.models.hodgkin_huxley.HodgkinHuxleyNeuron`; this module only
adds geometry (surface area) and axial coupling between neighbouring
compartments.

Cable equation (per compartment i, all membrane terms as densities)::

    c_m dV_i/dt = -i_ion(V_i, m_i, h_i, n_i)
                  + I_inj,i / A_i
                  + (1 / A_i) * sum_j g_axial(i,j) (V_j - V_i)

Units follow the rest of the project: mV, ms, uF/cm^2, mS/cm^2, uA/cm^2, with
lengths/radii in cm and axial resistivity R_a in ohm*cm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyParameters
from src.morphology.swc import Morphology

# Convert micrometres (SWC units) to centimetres (simulation units).
UM_TO_CM = 1e-4


@dataclass(frozen=True)
class Compartment:
    """One cylindrical compartment of a cable.

    Attributes:
        length_cm: Compartment length in cm.
        radius_cm: Compartment radius in cm.
        parent: Index of the parent compartment, or -1 for a root.
        type: SWC structure-type code (soma/axon/dendrite), for reference.
    """

    length_cm: float
    radius_cm: float
    parent: int
    type: int = 0


@dataclass(frozen=True)
class CableResult:
    """Result of a cable simulation.

    Attributes:
        time_ms: Time vector, shape (T,).
        voltage_mV: Membrane potential, shape (T, N) for N compartments.
        m, h, n: Gating variables, shape (T, N), or None when not recorded
            (skipped automatically for very large morphologies to save memory).
    """

    time_ms: np.ndarray
    voltage_mV: np.ndarray
    m: np.ndarray | None = None
    h: np.ndarray | None = None
    n: np.ndarray | None = None

    def compartment_trace(self, index: int) -> np.ndarray:
        """Return the voltage trace of a single compartment."""

        return self.voltage_mV[:, index]

    @property
    def has_gates(self) -> bool:
        """Whether per-compartment gating variables were recorded."""

        return self.m is not None and self.h is not None and self.n is not None


class CableNeuron:
    """A branched cable of HH compartments coupled by axial resistance."""

    def __init__(
        self,
        compartments: list[Compartment],
        hh_parameters: HodgkinHuxleyParameters | None = None,
        axial_resistivity_ohm_cm: float = 100.0,
    ) -> None:
        """Build the cable and precompute geometry and coupling conductances."""

        if not compartments:
            raise ValueError("A cable needs at least one compartment.")
        if axial_resistivity_ohm_cm <= 0:
            raise ValueError("axial_resistivity_ohm_cm must be positive.")

        self.compartments = compartments
        self.neuron = HodgkinHuxleyNeuron(hh_parameters)
        self.axial_resistivity_ohm_cm = axial_resistivity_ohm_cm
        self.n = len(compartments)

        # Surface area of each compartment (lateral cylinder area), in cm^2.
        self.area_cm2 = np.array(
            [2.0 * math.pi * c.radius_cm * c.length_cm for c in compartments],
            dtype=float,
        )
        if np.any(self.area_cm2 <= 0):
            raise ValueError("Every compartment must have positive length and radius.")

        # Axial coupling edges between each compartment and its parent.
        edge_i: list[int] = []
        edge_p: list[int] = []
        edge_g_mS: list[float] = []
        for index, comp in enumerate(compartments):
            parent = comp.parent
            if parent < 0:
                continue
            if not 0 <= parent < self.n:
                raise ValueError(
                    f"Compartment {index} has out-of-range parent {parent}."
                )
            edge_i.append(index)
            edge_p.append(parent)
            edge_g_mS.append(self._axial_conductance_mS(comp, compartments[parent]))

        self.edge_i = np.array(edge_i, dtype=int)
        self.edge_p = np.array(edge_p, dtype=int)
        self.edge_g_mS = np.array(edge_g_mS, dtype=float)

    def _axial_conductance_mS(self, a: Compartment, b: Compartment) -> float:
        """Axial conductance (mS) between the centres of two compartments.

        Series resistance of two half-cylinders: R = R_a * (L/2) / (pi r^2) for
        each compartment. Conductance is the reciprocal, scaled to mS so that
        g[mS] * dV[mV] yields axial current in uA.
        """

        r_a = self.axial_resistivity_ohm_cm
        cross_a = math.pi * a.radius_cm**2
        cross_b = math.pi * b.radius_cm**2
        resistance_ohm = r_a * (a.length_cm / 2.0) / cross_a + r_a * (b.length_cm / 2.0) / cross_b
        return 1.0e3 / resistance_ohm

    @classmethod
    def from_morphology(
        cls,
        morphology: Morphology,
        hh_parameters: HodgkinHuxleyParameters | None = None,
        axial_resistivity_ohm_cm: float = 100.0,
    ) -> "CableNeuron":
        """Build a cable from a parsed SWC :class:`Morphology`.

        One compartment is created per SWC sample. A sample's geometry is taken
        from the segment joining it to its parent (length = inter-sample
        distance, radius = the sample's radius). Root samples, which have no
        parent segment, are given a short stub length of twice their radius.
        """

        nodes = morphology.nodes
        id_to_index = {node.id: index for index, node in enumerate(nodes)}
        compartments: list[Compartment] = []
        for node in nodes:
            radius_cm = node.radius * UM_TO_CM
            if node.parent == -1:
                length_cm = 2.0 * radius_cm  # nominal soma/root stub
                parent_index = -1
            else:
                parent = morphology.node_by_id[node.parent]
                distance_um = math.dist(
                    (node.x, node.y, node.z), (parent.x, parent.y, parent.z)
                )
                length_cm = max(distance_um * UM_TO_CM, radius_cm)
                parent_index = id_to_index[node.parent]
            compartments.append(
                Compartment(
                    length_cm=length_cm,
                    radius_cm=radius_cm,
                    parent=parent_index,
                    type=node.type,
                )
            )
        return cls(
            compartments,
            hh_parameters=hh_parameters,
            axial_resistivity_ohm_cm=axial_resistivity_ohm_cm,
        )

    @classmethod
    def straight_cable(
        cls,
        n_compartments: int,
        length_um: float,
        radius_um: float,
        hh_parameters: HodgkinHuxleyParameters | None = None,
        axial_resistivity_ohm_cm: float = 100.0,
    ) -> "CableNeuron":
        """Build an unbranched cable of identical compartments in a line."""

        if n_compartments < 1:
            raise ValueError("n_compartments must be >= 1.")
        length_cm = length_um * UM_TO_CM
        radius_cm = radius_um * UM_TO_CM
        compartments = [
            Compartment(
                length_cm=length_cm,
                radius_cm=radius_cm,
                parent=index - 1,  # -1 for the first compartment
                type=0,
            )
            for index in range(n_compartments)
        ]
        return cls(
            compartments,
            hh_parameters=hh_parameters,
            axial_resistivity_ohm_cm=axial_resistivity_ohm_cm,
        )

    # ------------------------------------------------------------------ solve

    def max_stable_dt_ms(self) -> float:
        """A conservative RK4-stable time step (ms) for the current geometry.

        Explicit integration of the cable is stability-limited by the stiffest
        axial coupling (coupling rate grows as compartments shrink). This returns
        a step small enough to stay stable, capped at 0.01 ms for normal cables.
        """

        c_m = self.neuron.parameters.membrane_capacitance
        if self.edge_i.size == 0:
            return 0.01
        # Per-compartment coupling rate (1/ms): sum of g/(area*c_m) over its edges.
        coupling_rate = np.zeros(self.n, dtype=float)
        np.add.at(coupling_rate, self.edge_i, self.edge_g_mS / self.area_cm2[self.edge_i] / c_m)
        np.add.at(coupling_rate, self.edge_p, self.edge_g_mS / self.area_cm2[self.edge_p] / c_m)
        lam = 2.0 * float(np.max(coupling_rate)) + 5.0  # + rough membrane term
        return max(2e-4, min(0.01, 2.0 / lam))

    def _axial_density(self, voltage: np.ndarray) -> np.ndarray:
        """Return axial current density (uA/cm^2) into each compartment."""

        density = np.zeros(self.n, dtype=float)
        if self.edge_i.size == 0:
            return density
        # Axial current from parent p into child i: g (mS) * dV (mV) = current (uA).
        flux_uA = self.edge_g_mS * (voltage[self.edge_p] - voltage[self.edge_i])
        np.add.at(density, self.edge_i, flux_uA / self.area_cm2[self.edge_i])
        np.add.at(density, self.edge_p, -flux_uA / self.area_cm2[self.edge_p])
        return density

    def _derivatives(self, state: np.ndarray, injected_density: np.ndarray) -> np.ndarray:
        """Compute d/dt of the stacked [V, m, h, n] state for all compartments."""

        n = self.n
        voltage = state[0:n]
        m = state[n : 2 * n]
        h = state[2 * n : 3 * n]
        gate_n = state[3 * n : 4 * n]

        neuron = self.neuron
        params = neuron.parameters
        # Ionic current densities reuse the exact HH channel definitions.
        i_na = params.g_na * (m**3) * h * (voltage - params.e_na)
        i_k = params.g_k * (gate_n**4) * (voltage - params.e_k)
        i_l = params.g_l * (voltage - params.e_l)
        i_ion = i_na + i_k + i_l

        d_voltage = (
            injected_density - i_ion + self._axial_density(voltage)
        ) / params.membrane_capacitance

        alpha_m = neuron.alpha_m(voltage)
        beta_m = neuron.beta_m(voltage)
        alpha_h = neuron.alpha_h(voltage)
        beta_h = neuron.beta_h(voltage)
        alpha_n = neuron.alpha_n(voltage)
        beta_n = neuron.beta_n(voltage)

        d_m = alpha_m * (1.0 - m) - beta_m * m
        d_h = alpha_h * (1.0 - h) - beta_h * h
        d_n = alpha_n * (1.0 - gate_n) - beta_n * gate_n

        return np.concatenate([d_voltage, d_m, d_h, d_n])

    def simulate(
        self,
        duration_ms: float,
        dt_ms: float = 0.01,
        resting_voltage_mV: float = -65.0,
        stim_current_nA: float = 0.0,
        stim_compartment: int = 0,
        stim_start_ms: float = 1.0,
        stim_end_ms: float = 2.0,
        store_gates: bool = True,
        max_gate_samples: int = 6_000_000,
    ) -> CableResult:
        """Integrate the cable with RK4 under a rectangular point stimulus.

        Args:
            duration_ms: Total simulated time.
            dt_ms: Fixed integration step.
            resting_voltage_mV: Initial voltage; gates start at steady state.
            stim_current_nA: Absolute injected current at ``stim_compartment``.
            stim_compartment: Index of the stimulated compartment.
            stim_start_ms, stim_end_ms: Stimulus on/off times.
            store_gates: Record per-compartment m/h/n so any compartment can be
                inspected in the standard graphs. Skipped automatically when the
                morphology is large enough that storage would exceed
                ``max_gate_samples`` (N * timesteps).
        """

        if not 0 <= stim_compartment < self.n:
            raise ValueError("stim_compartment is out of range.")
        n = self.n
        steps = int(round(duration_ms / dt_ms))
        time_ms = np.arange(steps + 1, dtype=float) * dt_ms

        m_inf, h_inf, n_inf = self.neuron.steady_state_gates(resting_voltage_mV)
        state = np.concatenate(
            [
                np.full(n, resting_voltage_mV, dtype=float),
                np.full(n, m_inf, dtype=float),
                np.full(n, h_inf, dtype=float),
                np.full(n, n_inf, dtype=float),
            ]
        )

        voltage_out = np.empty((steps + 1, n), dtype=float)
        voltage_out[0] = state[0:n]

        record_gates = store_gates and n * (steps + 1) <= max_gate_samples
        if record_gates:
            m_out = np.empty((steps + 1, n), dtype=np.float32)
            h_out = np.empty((steps + 1, n), dtype=np.float32)
            n_out = np.empty((steps + 1, n), dtype=np.float32)
            m_out[0], h_out[0], n_out[0] = state[n : 2 * n], state[2 * n : 3 * n], state[3 * n :]

        # Pre-build the injected-current density applied to the stim compartment.
        stim_density = (stim_current_nA * 1e-3) / self.area_cm2[stim_compartment]

        def injected_density(t_ms: float) -> np.ndarray:
            density = np.zeros(n, dtype=float)
            if stim_start_ms <= t_ms <= stim_end_ms:
                density[stim_compartment] = stim_density
            return density

        for step in range(steps):
            t = time_ms[step]
            inj = injected_density(t)
            inj_mid = injected_density(t + 0.5 * dt_ms)
            inj_next = injected_density(t + dt_ms)

            k1 = self._derivatives(state, inj)
            k2 = self._derivatives(state + 0.5 * dt_ms * k1, inj_mid)
            k3 = self._derivatives(state + 0.5 * dt_ms * k2, inj_mid)
            k4 = self._derivatives(state + dt_ms * k3, inj_next)
            state = state + (dt_ms / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            voltage_out[step + 1] = state[0:n]
            if record_gates:
                m_out[step + 1] = state[n : 2 * n]
                h_out[step + 1] = state[2 * n : 3 * n]
                n_out[step + 1] = state[3 * n :]

        if record_gates:
            return CableResult(
                time_ms=time_ms, voltage_mV=voltage_out, m=m_out, h=h_out, n=n_out
            )
        return CableResult(time_ms=time_ms, voltage_mV=voltage_out)
