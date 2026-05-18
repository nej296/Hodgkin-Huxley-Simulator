"""Run matched Python Hodgkin-Huxley sweeps for NEURON validation."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

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
from src.analysis.spike_metrics import detect_spikes  # noqa: E402
from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyState  # noqa: E402
from src.simulation.config import SimulationConfig  # noqa: E402
from src.simulation.protocols import StepCurrent  # noqa: E402
from src.simulation.runner import SimulationResult, simulate  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output" / "validation" / "python_hh"
TRACE_DIR = OUTPUT_DIR / "traces"
INDIVIDUAL_PLOT_DIR = OUTPUT_DIR / "individual_voltage_plots"


def safe_current_label(current_uA_cm2: float) -> str:
    """Return a filename-safe current label."""

    return f"{current_uA_cm2:g}".replace(".", "p")


def summarize_result(current_uA_cm2: float, result: SimulationResult) -> dict[str, float]:
    """Compute validation metrics using the shared spike-detection settings."""

    spike_times = detect_spikes(
        time_ms=result.time_ms,
        voltage_mV=result.voltage_mV,
        threshold_mV=SPIKE_THRESHOLD_MV,
        refractory_ms=REFRACTORY_MS,
    )
    stimulus_spikes = tuple(
        spike_time
        for spike_time in spike_times
        if CURRENT_START_MS <= spike_time <= CURRENT_END_MS
    )

    first_latency = (
        stimulus_spikes[0] - CURRENT_START_MS if stimulus_spikes else np.nan
    )
    firing_rate = len(stimulus_spikes) / (CURRENT_DURATION_MS / 1000.0)

    return {
        "current_uA_cm2": float(current_uA_cm2),
        "spike_count": int(len(stimulus_spikes)),
        "firing_rate_hz": float(firing_rate),
        "peak_voltage_mV": float(np.max(result.voltage_mV)),
        "trough_voltage_mV": float(np.min(result.voltage_mV)),
        "first_spike_latency_ms": float(first_latency),
        "max_recorded_current_uA_cm2": float(np.max(result.injected_current_uA_cm2)),
    }


def save_trace_csv(current_uA_cm2: float, result: SimulationResult) -> Path:
    """Save one Python HH trace to CSV."""

    path = TRACE_DIR / f"python_trace_{safe_current_label(current_uA_cm2)}_uA_cm2.csv"
    data = np.column_stack(
        (
            result.time_ms,
            result.voltage_mV,
            result.injected_current_uA_cm2,
        )
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="time_ms,voltage_mV,injected_current_uA_cm2",
        comments="",
    )
    return path


def save_summary_csv(summary_rows: list[dict[str, float]]) -> Path:
    """Save Python HH summary metrics for each current level."""

    path = OUTPUT_DIR / "python_hh_fi_summary.csv"
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


def plot_individual_voltage_trace(current_uA_cm2: float, result: SimulationResult) -> Path:
    """Save one readable voltage figure for a selected current level."""

    path = (
        INDIVIDUAL_PLOT_DIR
        / f"python_voltage_{safe_current_label(current_uA_cm2)}_uA_cm2.png"
    )
    plt.figure(figsize=(9, 4.8))
    plt.plot(result.time_ms, result.voltage_mV, color="#1f5f99", linewidth=1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Python HH response at {current_uA_cm2:g} uA/cm2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    """Run all validation current levels using the Python HH simulator."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    neuron = HodgkinHuxleyNeuron()
    config = SimulationConfig(duration_ms=TSTOP_MS, dt_ms=DT_MS, method="rk4")
    initial_state = HodgkinHuxleyState.from_voltage(neuron, INITIAL_VOLTAGE_MV)
    summary_rows: list[dict[str, float]] = []

    for current in CURRENT_LEVELS_UA_CM2:
        protocol = StepCurrent(
            amplitude=current,
            start_ms=CURRENT_START_MS,
            end_ms=CURRENT_END_MS,
        )
        result = simulate(
            neuron=neuron,
            config=config,
            current_protocol=protocol,
            initial_state=initial_state,
        )
        save_trace_csv(current, result)
        metrics = summarize_result(current, result)
        summary_rows.append(metrics)

        if current in EXAMPLE_CURRENTS_UA_CM2:
            plot_individual_voltage_trace(current, result)

        print(
            f"{current:>6g} uA/cm2 | "
            f"spikes={metrics['spike_count']:>2} | "
            f"rate={metrics['firing_rate_hz']:>7.2f} Hz | "
            f"peak={metrics['peak_voltage_mV']:>7.2f} mV"
        )

    summary_path = save_summary_csv(summary_rows)
    print()
    print("Finished Python HH validation sweep.")
    print(f"Summary CSV: {summary_path}")
    print(f"Trace CSV folder: {TRACE_DIR}")
    print(f"Individual voltage plot folder: {INDIVIDUAL_PLOT_DIR}")


if __name__ == "__main__":
    main()
