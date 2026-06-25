"""Command-line entry point for a basic Hodgkin-Huxley simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.spike_metrics import summarize_voltage_trace
from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyParameters
from src.simulation.config import SimulationConfig
from src.simulation.protocols import StepCurrent
from src.simulation.runner import simulate
from src.utils.export import export_simulation_csv
from src.visualization.plotting import plot_voltage_trace


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for reproducible simulation runs."""

    parser = argparse.ArgumentParser(description="Run a Hodgkin-Huxley neuron simulation.")
    parser.add_argument("--duration-ms", type=float, default=50.0)
    parser.add_argument("--dt-ms", type=float, default=0.01)
    parser.add_argument("--current", type=float, default=10.0, help="Step current in uA/cm^2.")
    parser.add_argument("--start-ms", type=float, default=5.0)
    parser.add_argument("--end-ms", type=float, default=35.0)
    parser.add_argument("--g-na", type=float, default=120.0, help="Maximum Na conductance.")
    parser.add_argument("--g-k", type=float, default=36.0, help="Maximum K conductance.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    return parser


def main() -> None:
    """Run a configurable HH simulation and save trace outputs."""

    args = build_parser().parse_args()
    parameters = HodgkinHuxleyParameters(g_na=args.g_na, g_k=args.g_k)
    neuron = HodgkinHuxleyNeuron(parameters)
    config = SimulationConfig(duration_ms=args.duration_ms, dt_ms=args.dt_ms, method="rk4")
    current = StepCurrent(amplitude=args.current, start_ms=args.start_ms, end_ms=args.end_ms)

    result = simulate(neuron=neuron, config=config, current_protocol=current)
    metrics = summarize_voltage_trace(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_simulation_csv(result, args.output_dir / "simulation.csv")
    plot_path = args.output_dir / "simulation.png"
    plot_voltage_trace(result, plot_path, show=args.show)

    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Spikes: {metrics.spike_count}")
    print(f"Firing rate: {metrics.firing_rate_hz:.2f} Hz")
    print(f"Peak voltage: {metrics.peak_voltage_mV:.2f} mV")


if __name__ == "__main__":
    main()
