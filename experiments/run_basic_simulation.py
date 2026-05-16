"""Run a reproducible single-compartment Hodgkin-Huxley simulation."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.spike_metrics import summarize_voltage_trace
from src.models.hodgkin_huxley import HodgkinHuxleyNeuron
from src.simulation.config import SimulationConfig
from src.simulation.protocols import StepCurrent
from src.simulation.runner import simulate
from src.utils.export import export_simulation_csv
from src.visualization.plotting import plot_voltage_trace


def main() -> None:
    """Generate a voltage trace, CSV export, and publication-style plot."""

    neuron = HodgkinHuxleyNeuron()
    config = SimulationConfig(duration_ms=50.0, dt_ms=0.01, method="rk4")
    current = StepCurrent(amplitude=10.0, start_ms=5.0, end_ms=35.0)

    result = simulate(neuron=neuron, config=config, current_protocol=current)
    metrics = summarize_voltage_trace(result)

    output_dir = PROJECT_ROOT / "output"
    csv_path = export_simulation_csv(result, output_dir / "basic_simulation.csv")
    plot_voltage_trace(result, output_dir / "basic_simulation.png")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {output_dir / 'basic_simulation.png'}")
    print(
        "Spike summary: "
        f"{metrics.spike_count} spikes, "
        f"{metrics.firing_rate_hz:.1f} Hz, "
        f"peak {metrics.peak_voltage_mV:.2f} mV"
    )


if __name__ == "__main__":
    main()
