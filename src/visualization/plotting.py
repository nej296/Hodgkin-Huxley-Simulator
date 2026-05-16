"""Matplotlib plotting utilities for Hodgkin-Huxley simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt

from src.simulation.runner import SimulationResult


def plot_voltage_trace(
    result: SimulationResult,
    output_path: str | Path | None = None,
    title: str = "Hodgkin-Huxley membrane response",
    show: bool = False,
):
    """Plot membrane voltage and injected current over time.

    Args:
        result: Simulation output to visualize.
        output_path: Optional image path. Parent directories are created.
        title: Figure title.
        show: Whether to display the plot interactively.
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(result.time_ms, result.voltage_mV, color="#1f5f99", linewidth=1.8)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        result.time_ms,
        result.injected_current_uA_cm2,
        color="#7a3b12",
        linewidth=1.5,
    )
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Current (uA/cm^2)")
    axes[1].grid(True, alpha=0.25)

    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_conductance_sweep(
    traces: Mapping[str, SimulationResult],
    output_path: str | Path | None = None,
    title: str = "Conductance sweep voltage traces",
    show: bool = False,
):
    """Plot voltage traces from a conductance sweep on shared axes."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for label, result in traces.items():
        axis.plot(result.time_ms, result.voltage_mV, linewidth=1.3, label=label)

    axis.set_title(title)
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Voltage (mV)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axis
