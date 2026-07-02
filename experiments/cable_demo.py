"""Demonstrate action-potential propagation along an HH cable.

Builds an unbranched cable, injects current at one end, and shows the spike
propagating down the cable as a space-time voltage map plus a few single-
compartment traces. If an SWC file path is given on the command line, the cable
is built from that morphology instead.

Usage::

    python experiments/cable_demo.py [path/to/morphology.swc]
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.morphology.swc import load_swc
from src.simulation.cable import CableNeuron


def _conduction_velocity(result, cable: CableNeuron) -> float | None:
    """Estimate conduction velocity (m/s) from spike times along a line cable."""

    spike_threshold_mV = 0.0
    distances_cm = np.cumsum([c.length_cm for c in cable.compartments])
    spike_times: list[tuple[float, float]] = []
    for index in range(cable.n):
        trace = result.voltage_mV[:, index]
        crossings = np.where((trace[:-1] < spike_threshold_mV) & (trace[1:] >= spike_threshold_mV))[0]
        if crossings.size:
            spike_times.append((distances_cm[index], result.time_ms[crossings[0]]))
    if len(spike_times) < 2:
        return None
    distances = np.array([d for d, _ in spike_times])
    times = np.array([t for _, t in spike_times])
    slope_cm_per_ms = np.polyfit(times, distances, 1)[0]  # cm/ms
    return slope_cm_per_ms * 10.0  # cm/ms -> m/s


def main() -> None:
    """Run the cable demo and save a figure to output/."""

    if len(sys.argv) > 1:
        morphology = load_swc(sys.argv[1])
        cable = CableNeuron.from_morphology(morphology)
        title = f"Cable from {Path(sys.argv[1]).name} ({cable.n} compartments)"
    else:
        cable = CableNeuron.straight_cable(
            n_compartments=80, length_um=50.0, radius_um=1.0
        )
        title = f"Straight HH cable ({cable.n} compartments)"

    result = cable.simulate(
        duration_ms=22.0,
        dt_ms=0.01,
        stim_current_nA=0.3,
        stim_compartment=0,
        stim_start_ms=1.0,
        stim_end_ms=2.0,
    )

    velocity = _conduction_velocity(result, cable)
    print(title)
    print(f"  compartments: {cable.n}")
    print(f"  peak voltage: {result.voltage_mV.max():.1f} mV")
    if velocity is not None:
        print(f"  estimated conduction velocity: {velocity:.2f} m/s")
    else:
        print("  (no propagating spike detected)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_map, ax_traces) = plt.subplots(2, 1, figsize=(8, 7))
    mesh = ax_map.imshow(
        result.voltage_mV.T,
        aspect="auto",
        origin="lower",
        extent=[result.time_ms[0], result.time_ms[-1], 0, cable.n],
        cmap="viridis",
    )
    ax_map.set_xlabel("time (ms)")
    ax_map.set_ylabel("compartment")
    ax_map.set_title(title)
    fig.colorbar(mesh, ax=ax_map, label="V (mV)")

    for index in np.linspace(0, cable.n - 1, 5, dtype=int):
        ax_traces.plot(result.time_ms, result.voltage_mV[:, index], label=f"comp {index}")
    ax_traces.set_xlabel("time (ms)")
    ax_traces.set_ylabel("V (mV)")
    ax_traces.legend(fontsize=8, ncol=5)
    fig.tight_layout()

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "cable_propagation.png"
    fig.savefig(out_path, dpi=130)
    print(f"  saved figure: {out_path}")


if __name__ == "__main__":
    main()
