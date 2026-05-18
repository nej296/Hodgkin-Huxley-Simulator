"""Run NEURON Hodgkin-Huxley sweeps using density-current dclamp."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

from neuron import h
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuron_validation.validation_config import (  # noqa: E402
    CURRENT_DURATION_MS,
    CURRENT_END_MS,
    CURRENT_LEVELS_UA_CM2,
    CURRENT_START_MS,
    DT_MS,
    EXAMPLE_CURRENTS_UA_CM2,
    INITIAL_VOLTAGE_MV,
    REFRACTORY_MS,
    SPIKE_THRESHOLD_MV,
    TSTOP_MS,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "validation" / "neuron_hh"
TRACE_DIR = OUTPUT_DIR / "traces"
INDIVIDUAL_PLOT_DIR = OUTPUT_DIR / "individual_voltage_plots"


def safe_current_label(current_uA_cm2: float) -> str:
    """Return a filename-safe current label."""

    return f"{current_uA_cm2:g}".replace(".", "p")


def build_single_compartment_hh_cell():
    """Create one NEURON soma with standard Hodgkin-Huxley channels."""

    soma = h.Section(name="soma")
    soma.L = 20.0
    soma.diam = 20.0
    soma.nseg = 1
    soma.cm = 1.0

    soma.insert("hh")
    soma.insert("dclamp")

    soma.ena = 50.0
    soma.ek = -77.0

    for seg in soma:
        seg.hh.gnabar = 0.120
        seg.hh.gkbar = 0.036
        seg.hh.gl = 0.0003
        seg.hh.el = -54.387
        seg.dclamp.delay = CURRENT_START_MS
        seg.dclamp.dur = CURRENT_DURATION_MS

    return soma


def detect_spikes(time_ms: np.ndarray, voltage_mV: np.ndarray) -> list[float]:
    """Detect upward threshold crossings during the stimulus window."""

    crossings = np.where(
        (voltage_mV[:-1] < SPIKE_THRESHOLD_MV)
        & (voltage_mV[1:] >= SPIKE_THRESHOLD_MV)
    )[0]
    spike_times: list[float] = []
    last_spike_time = -np.inf

    for index in crossings:
        spike_time = float(time_ms[index + 1])
        in_stimulus = CURRENT_START_MS <= spike_time <= CURRENT_END_MS
        outside_refractory = spike_time - last_spike_time >= REFRACTORY_MS
        if in_stimulus and outside_refractory:
            spike_times.append(spike_time)
            last_spike_time = spike_time

    return spike_times


def run_one_current(soma, current_uA_cm2: float):
    """Run one NEURON simulation with dclamp current in uA/cm2."""

    for seg in soma:
        seg.dclamp.amp = float(current_uA_cm2)

    time_vec = h.Vector().record(h._ref_t)
    voltage_vec = h.Vector().record(soma(0.5)._ref_v)
    current_vec = h.Vector().record(soma(0.5).dclamp._ref_iinj)

    h.finitialize(INITIAL_VOLTAGE_MV)
    h.continuerun(TSTOP_MS)

    time_ms = np.array(time_vec)
    voltage_mV = np.array(voltage_vec)
    current_trace_uA_cm2 = np.array(current_vec)
    spike_times = detect_spikes(time_ms, voltage_mV)
    first_latency = spike_times[0] - CURRENT_START_MS if spike_times else np.nan
    firing_rate = len(spike_times) / (CURRENT_DURATION_MS / 1000.0)

    metrics = {
        "current_uA_cm2": float(current_uA_cm2),
        "spike_count": int(len(spike_times)),
        "firing_rate_hz": float(firing_rate),
        "peak_voltage_mV": float(np.max(voltage_mV)),
        "trough_voltage_mV": float(np.min(voltage_mV)),
        "first_spike_latency_ms": float(first_latency),
        "max_recorded_current_uA_cm2": float(np.max(current_trace_uA_cm2)),
    }
    return time_ms, voltage_mV, current_trace_uA_cm2, metrics


def save_trace_csv(
    current_uA_cm2: float,
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
    current_trace_uA_cm2: np.ndarray,
) -> Path:
    """Save one NEURON voltage/current trace to CSV."""

    path = TRACE_DIR / f"neuron_trace_{safe_current_label(current_uA_cm2)}_uA_cm2.csv"
    data = np.column_stack((time_ms, voltage_mV, current_trace_uA_cm2))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="time_ms,voltage_mV,injected_current_uA_cm2",
        comments="",
    )
    return path


def save_summary_csv(summary_rows: list[dict[str, float]]) -> Path:
    """Save NEURON F-I and voltage summary metrics."""

    path = OUTPUT_DIR / "neuron_hh_fi_summary.csv"
    fieldnames = [
        "current_uA_cm2",
        "spike_count",
        "firing_rate_hz",
        "peak_voltage_mV",
        "trough_voltage_mV",
        "first_spike_latency_ms",
        "max_recorded_current_uA_cm2",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return path


def plot_individual_voltage_trace(
    current_uA_cm2: float,
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
) -> Path:
    """Save one readable voltage figure for a selected current level."""

    path = (
        INDIVIDUAL_PLOT_DIR
        / f"neuron_voltage_{safe_current_label(current_uA_cm2)}_uA_cm2.png"
    )
    plt.figure(figsize=(9, 4.8))
    plt.plot(time_ms, voltage_mV, color="#1f5f99", linewidth=1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"NEURON HH response at {current_uA_cm2:g} uA/cm2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    """Run all validation current levels through NEURON with dclamp."""

    h.load_file("stdrun.hoc")
    h.dt = DT_MS
    h.tstop = TSTOP_MS
    h.CVode().active(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    soma = build_single_compartment_hh_cell()
    summary_rows: list[dict[str, float]] = []

    for current in CURRENT_LEVELS_UA_CM2:
        time_ms, voltage_mV, current_trace, metrics = run_one_current(soma, current)
        save_trace_csv(current, time_ms, voltage_mV, current_trace)
        summary_rows.append(metrics)

        if current in EXAMPLE_CURRENTS_UA_CM2:
            plot_individual_voltage_trace(current, time_ms, voltage_mV)

        print(
            f"{current:>6g} uA/cm2 | "
            f"spikes={metrics['spike_count']:>2} | "
            f"rate={metrics['firing_rate_hz']:>7.2f} Hz | "
            f"peak={metrics['peak_voltage_mV']:>7.2f} mV"
        )

    summary_path = save_summary_csv(summary_rows)
    print()
    print("Finished NEURON HH validation sweep.")
    print(f"Summary CSV: {summary_path}")
    print(f"Trace CSV folder: {TRACE_DIR}")
    print(f"Individual voltage plot folder: {INDIVIDUAL_PLOT_DIR}")


if __name__ == "__main__":
    main()
