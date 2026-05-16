"""Parameter sweeps for sodium and potassium conductance effects."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.spike_metrics import summarize_voltage_trace
from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyParameters
from src.simulation.config import SimulationConfig
from src.simulation.protocols import StepCurrent
from src.simulation.runner import SimulationResult, simulate
from src.visualization.plotting import plot_conductance_sweep


def run_sweep(
    conductance_name: str,
    values: list[float],
    baseline: HodgkinHuxleyParameters,
    config: SimulationConfig,
    current: StepCurrent,
) -> dict[str, SimulationResult]:
    """Run a one-parameter conductance sweep and return labeled traces."""

    traces: dict[str, SimulationResult] = {}
    for value in values:
        parameters = replace(baseline, **{conductance_name: value})
        neuron = HodgkinHuxleyNeuron(parameters)
        label = f"{conductance_name}={value:g}"
        traces[label] = simulate(neuron=neuron, config=config, current_protocol=current)
    return traces


def write_summary(
    traces: dict[str, SimulationResult],
    conductance_name: str,
    output_path: Path,
) -> Path:
    """Export sweep-level spike and voltage metrics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "condition",
                "conductance",
                "spike_count",
                "firing_rate_hz",
                "peak_voltage_mV",
                "trough_voltage_mV",
                "first_spike_time_ms",
            ]
        )
        for label, result in traces.items():
            metrics = summarize_voltage_trace(result)
            writer.writerow(
                [
                    label,
                    conductance_name,
                    metrics.spike_count,
                    metrics.firing_rate_hz,
                    metrics.peak_voltage_mV,
                    metrics.trough_voltage_mV,
                    metrics.first_spike_time_ms,
                ]
            )
    return output_path


def main() -> None:
    """Compare how g_Na and g_K reshape action-potential traces."""

    output_dir = PROJECT_ROOT / "output" / "conductance_sweeps"
    baseline = HodgkinHuxleyParameters()
    config = SimulationConfig(duration_ms=60.0, dt_ms=0.01, method="rk4")
    current = StepCurrent(amplitude=10.0, start_ms=5.0, end_ms=45.0)

    sodium_traces = run_sweep(
        conductance_name="g_na",
        values=[80.0, 100.0, 120.0, 140.0, 160.0],
        baseline=baseline,
        config=config,
        current=current,
    )
    potassium_traces = run_sweep(
        conductance_name="g_k",
        values=[20.0, 30.0, 36.0, 45.0, 55.0],
        baseline=baseline,
        config=config,
        current=current,
    )

    plot_conductance_sweep(
        sodium_traces,
        output_dir / "sodium_conductance_sweep.png",
        title="Sodium conductance sweep",
    )
    plot_conductance_sweep(
        potassium_traces,
        output_dir / "potassium_conductance_sweep.png",
        title="Potassium conductance sweep",
    )
    sodium_summary = write_summary(sodium_traces, "g_na", output_dir / "sodium_summary.csv")
    potassium_summary = write_summary(potassium_traces, "g_k", output_dir / "potassium_summary.csv")

    print(f"Saved sodium sweep summary: {sodium_summary}")
    print(f"Saved potassium sweep summary: {potassium_summary}")


if __name__ == "__main__":
    main()
