"""Tests for SWC parsing and the multi-compartment HH cable model."""

from __future__ import annotations

import numpy as np
import pytest

from src.morphology.swc import Morphology, parse_swc
from src.simulation.cable import CableNeuron, Compartment


SAMPLE_SWC = """\
# a tiny three-point cable
1 1 0.0 0.0 0.0 1.0 -1
2 2 10.0 0.0 0.0 1.0 1
3 2 20.0 0.0 0.0 1.0 2
"""


def test_parse_swc_reads_all_samples() -> None:
    morphology = parse_swc(SAMPLE_SWC)
    assert isinstance(morphology, Morphology)
    assert len(morphology.nodes) == 3
    assert morphology.roots()[0].id == 1
    assert morphology.node_by_id[2].parent == 1


def test_parse_swc_ignores_comments_and_blank_lines() -> None:
    text = "# header\n\n1 1 0 0 0 1 -1\n\n# trailing comment\n"
    morphology = parse_swc(text)
    assert len(morphology.nodes) == 1


def test_parse_swc_rejects_missing_parent() -> None:
    with pytest.raises(ValueError):
        parse_swc("1 1 0 0 0 1 -1\n2 2 1 0 0 1 99\n")


def test_cable_from_morphology_has_expected_compartments() -> None:
    cable = CableNeuron.from_morphology(parse_swc(SAMPLE_SWC))
    assert cable.n == 3
    # Two parent links -> two axial coupling edges.
    assert cable.edge_i.size == 2


def test_axial_conductance_is_symmetric() -> None:
    a = Compartment(length_cm=0.01, radius_cm=1e-4, parent=-1)
    b = Compartment(length_cm=0.01, radius_cm=1e-4, parent=0)
    cable = CableNeuron([a, b])
    # Reversing the argument order must not change the coupling conductance.
    g_ab = cable._axial_conductance_mS(a, b)
    g_ba = cable._axial_conductance_mS(b, a)
    assert g_ab == pytest.approx(g_ba)


def test_isolated_compartment_stays_near_rest() -> None:
    # With no stimulus, a single compartment initialised at rest should not drift
    # far from the HH resting potential (it reduces to the point model).
    cable = CableNeuron.straight_cable(n_compartments=1, length_um=50.0, radius_um=1.0)
    result = cable.simulate(duration_ms=5.0, dt_ms=0.01, stim_current_nA=0.0)
    assert np.all(np.isfinite(result.voltage_mV))
    assert np.allclose(result.voltage_mV, -65.0, atol=1.0)


def test_action_potential_propagates_down_cable() -> None:
    cable = CableNeuron.straight_cable(n_compartments=60, length_um=50.0, radius_um=1.0)
    result = cable.simulate(
        duration_ms=20.0,
        dt_ms=0.01,
        stim_current_nA=0.3,
        stim_compartment=0,
        stim_start_ms=1.0,
        stim_end_ms=2.0,
    )
    assert np.all(np.isfinite(result.voltage_mV))

    def first_crossing_ms(index: int) -> float | None:
        trace = result.voltage_mV[:, index]
        crossings = np.where((trace[:-1] < 0.0) & (trace[1:] >= 0.0))[0]
        return float(result.time_ms[crossings[0]]) if crossings.size else None

    near = first_crossing_ms(0)
    far = first_crossing_ms(cable.n - 1)
    # Both ends spike, and the far end spikes later -> the AP propagated.
    assert near is not None
    assert far is not None
    assert far > near


def test_simulate_records_gates_by_default() -> None:
    cable = CableNeuron.straight_cable(n_compartments=10, length_um=50.0, radius_um=1.0)
    result = cable.simulate(duration_ms=3.0, dt_ms=0.01, stim_current_nA=0.0)
    assert result.has_gates
    assert result.m.shape == result.voltage_mV.shape
    assert result.h.shape == result.voltage_mV.shape
    assert result.n.shape == result.voltage_mV.shape


def test_store_gates_can_be_disabled() -> None:
    cable = CableNeuron.straight_cable(n_compartments=10, length_um=50.0, radius_um=1.0)
    result = cable.simulate(duration_ms=3.0, dt_ms=0.01, store_gates=False)
    assert not result.has_gates
    assert result.m is None


def test_max_stable_dt_is_positive_and_capped() -> None:
    cable = CableNeuron.straight_cable(n_compartments=40, length_um=50.0, radius_um=1.0)
    dt = cable.max_stable_dt_ms()
    assert 0 < dt <= 0.01


def test_compartment_reconstructs_valid_simulation_result() -> None:
    # The GUI turns one cable compartment into a SimulationResult for the standard
    # graphs; this checks that reconstruction produces finite, consistent traces.
    from src.analysis.spike_metrics import summarize_voltage_trace
    from src.models.hodgkin_huxley import HodgkinHuxleyParameters
    from src.simulation.runner import SimulationResult

    cable = CableNeuron.straight_cable(n_compartments=30, length_um=50.0, radius_um=1.0)
    result = cable.simulate(
        duration_ms=15.0, dt_ms=cable.max_stable_dt_ms(),
        stim_current_nA=0.3, stim_compartment=0, stim_start_ms=1.0, stim_end_ms=2.0,
    )
    idx = 0
    params = HodgkinHuxleyParameters()
    t = result.time_ms
    v = result.voltage_mV[:, idx].astype(float)
    m = result.m[:, idx].astype(float)
    h = result.h[:, idx].astype(float)
    gn = result.n[:, idx].astype(float)
    g_na = params.g_na * (m**3) * h
    g_k = params.g_k * (gn**4)
    sim = SimulationResult(
        time_ms=t, voltage_mV=v, m=m, h=h, n=gn,
        injected_current_uA_cm2=np.zeros_like(t),
        g_na_max_mS_cm2=np.full_like(t, params.g_na),
        g_k_max_mS_cm2=np.full_like(t, params.g_k),
        sodium_conductance_mS_cm2=g_na,
        potassium_conductance_mS_cm2=g_k,
        sodium_current_uA_cm2=g_na * (v - params.e_na),
        potassium_current_uA_cm2=g_k * (v - params.e_k),
        leak_current_uA_cm2=params.g_l * (v - params.e_l),
        net_ionic_current_uA_cm2=np.zeros_like(t),
    )
    metrics = summarize_voltage_trace(sim)
    assert np.all(np.isfinite(sim.voltage_mV))
    assert metrics.spike_count >= 1  # the stimulated end fires
