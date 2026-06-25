"""Data export helpers for simulation outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from src.simulation.runner import SimulationResult


def export_simulation_csv(result: SimulationResult, output_path: str | Path) -> Path:
    """Export a simulation result to a CSV file.

    Columns are explicit about units so exported traces remain interpretable in
    downstream analysis notebooks, spreadsheets, or validation scripts.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "time_ms",
                "voltage_mV",
                "m",
                "h",
                "n",
                "injected_current_uA_cm2",
                "g_na_max_mS_cm2",
                "g_k_max_mS_cm2",
                "sodium_conductance_mS_cm2",
                "potassium_conductance_mS_cm2",
            ]
        )
        writer.writerows(
            zip(
                result.time_ms,
                result.voltage_mV,
                result.m,
                result.h,
                result.n,
                result.injected_current_uA_cm2,
                result.g_na_max_mS_cm2,
                result.g_k_max_mS_cm2,
                result.sodium_conductance_mS_cm2,
                result.potassium_conductance_mS_cm2,
            )
        )

    return path
