"""Quantitative error analysis for Python HH vs NEURON HH validation outputs."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuron_validation.validation_config import CURRENT_LEVELS_UA_CM2  # noqa: E402


VALIDATION_DIR = PROJECT_ROOT / "output" / "validation"
PYTHON_DIR = VALIDATION_DIR / "python_hh"
NEURON_DIR = VALIDATION_DIR / "neuron_hh"
COMPARISON_DIR = VALIDATION_DIR / "comparison"


def safe_current_label(current_uA_cm2: float) -> str:
    """Return the filename-safe current label used by the sweep exporters."""

    return f"{current_uA_cm2:g}".replace(".", "p")


def load_trace(path: Path) -> dict[str, np.ndarray]:
    """Load a trace CSV with columns time, voltage, and injected current."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "time_ms": np.asarray(data["time_ms"], dtype=float),
        "voltage_mV": np.asarray(data["voltage_mV"], dtype=float),
        "injected_current_uA_cm2": np.asarray(
            data["injected_current_uA_cm2"],
            dtype=float,
        ),
    }


def load_summary(path: Path) -> dict[float, dict[str, float]]:
    """Load a summary CSV and index rows by current level."""

    rows: dict[float, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            current = float(row["current_uA_cm2"])
            rows[current] = {
                key: float(value) if value.lower() != "nan" else np.nan
                for key, value in row.items()
            }
    return rows


def compute_voltage_errors(
    python_trace: dict[str, np.ndarray],
    neuron_trace: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute pointwise voltage and injected-current error metrics."""

    python_time = python_trace["time_ms"]
    neuron_time = neuron_trace["time_ms"]
    python_voltage = python_trace["voltage_mV"]
    neuron_voltage = neuron_trace["voltage_mV"]

    if python_time.shape != neuron_time.shape or not np.allclose(
        python_time,
        neuron_time,
        atol=1e-12,
    ):
        neuron_voltage = np.interp(python_time, neuron_time, neuron_voltage)

    voltage_error = python_voltage - neuron_voltage
    current_error = (
        python_trace["injected_current_uA_cm2"]
        - np.interp(
            python_time,
            neuron_time,
            neuron_trace["injected_current_uA_cm2"],
        )
    )

    return {
        "voltage_rmse_mV": float(np.sqrt(np.mean(voltage_error**2))),
        "voltage_mae_mV": float(np.mean(np.abs(voltage_error))),
        "voltage_max_abs_error_mV": float(np.max(np.abs(voltage_error))),
        "voltage_mean_error_mV": float(np.mean(voltage_error)),
        "injected_current_max_abs_error_uA_cm2": float(np.max(np.abs(current_error))),
        "compared_points": int(len(python_time)),
    }


def finite_difference(a: float, b: float) -> float:
    """Return a - b, preserving NaN when either side is unavailable."""

    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)


def compare_current_level(
    current: float,
    python_summary: dict[str, float],
    neuron_summary: dict[str, float],
    voltage_errors: dict[str, float],
) -> dict[str, float]:
    """Combine trace and summary errors for one current level."""

    return {
        "current_uA_cm2": current,
        "python_spike_count": int(python_summary["spike_count"]),
        "neuron_spike_count": int(neuron_summary["spike_count"]),
        "spike_count_difference": int(
            python_summary["spike_count"] - neuron_summary["spike_count"]
        ),
        "python_firing_rate_hz": python_summary["firing_rate_hz"],
        "neuron_firing_rate_hz": neuron_summary["firing_rate_hz"],
        "firing_rate_difference_hz": (
            python_summary["firing_rate_hz"] - neuron_summary["firing_rate_hz"]
        ),
        "python_peak_voltage_mV": python_summary["peak_voltage_mV"],
        "neuron_peak_voltage_mV": neuron_summary["peak_voltage_mV"],
        "peak_voltage_difference_mV": (
            python_summary["peak_voltage_mV"] - neuron_summary["peak_voltage_mV"]
        ),
        "python_trough_voltage_mV": python_summary["trough_voltage_mV"],
        "neuron_trough_voltage_mV": neuron_summary["trough_voltage_mV"],
        "trough_voltage_difference_mV": (
            python_summary["trough_voltage_mV"] - neuron_summary["trough_voltage_mV"]
        ),
        "python_first_spike_latency_ms": python_summary["first_spike_latency_ms"],
        "neuron_first_spike_latency_ms": neuron_summary["first_spike_latency_ms"],
        "first_spike_latency_difference_ms": finite_difference(
            python_summary["first_spike_latency_ms"],
            neuron_summary["first_spike_latency_ms"],
        ),
        **voltage_errors,
    }


def save_rows_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    """Save a list of metric rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def compute_aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Summarize validation errors across all current levels."""

    voltage_rmse_values = np.array([row["voltage_rmse_mV"] for row in rows])
    voltage_mae_values = np.array([row["voltage_mae_mV"] for row in rows])
    max_abs_values = np.array([row["voltage_max_abs_error_mV"] for row in rows])
    firing_errors = np.array([row["firing_rate_difference_hz"] for row in rows])
    spike_count_errors = np.array([row["spike_count_difference"] for row in rows])
    peak_errors = np.array([row["peak_voltage_difference_mV"] for row in rows])
    trough_errors = np.array([row["trough_voltage_difference_mV"] for row in rows])
    latency_errors = np.array(
        [row["first_spike_latency_difference_ms"] for row in rows],
        dtype=float,
    )

    python_threshold = next(
        (
            row["current_uA_cm2"]
            for row in rows
            if row["python_spike_count"] > 0
        ),
        np.nan,
    )
    neuron_threshold = next(
        (
            row["current_uA_cm2"]
            for row in rows
            if row["neuron_spike_count"] > 0
        ),
        np.nan,
    )

    return {
        "mean_voltage_rmse_mV": float(np.mean(voltage_rmse_values)),
        "median_voltage_rmse_mV": float(np.median(voltage_rmse_values)),
        "max_voltage_rmse_mV": float(np.max(voltage_rmse_values)),
        "mean_voltage_mae_mV": float(np.mean(voltage_mae_values)),
        "max_voltage_max_abs_error_mV": float(np.max(max_abs_values)),
        "firing_rate_rmse_hz": float(np.sqrt(np.mean(firing_errors**2))),
        "max_abs_firing_rate_error_hz": float(np.max(np.abs(firing_errors))),
        "max_abs_spike_count_difference": int(np.max(np.abs(spike_count_errors))),
        "mean_abs_peak_voltage_error_mV": float(np.mean(np.abs(peak_errors))),
        "max_abs_peak_voltage_error_mV": float(np.max(np.abs(peak_errors))),
        "mean_abs_trough_voltage_error_mV": float(np.mean(np.abs(trough_errors))),
        "max_abs_trough_voltage_error_mV": float(np.max(np.abs(trough_errors))),
        "mean_abs_first_spike_latency_error_ms": float(
            np.nanmean(np.abs(latency_errors))
        ),
        "max_abs_first_spike_latency_error_ms": float(
            np.nanmax(np.abs(latency_errors))
        ),
        "python_threshold_current_uA_cm2": float(python_threshold),
        "neuron_threshold_current_uA_cm2": float(neuron_threshold),
        "threshold_current_difference_uA_cm2": finite_difference(
            float(python_threshold),
            float(neuron_threshold),
        ),
    }


def save_aggregate_metrics(path: Path, metrics: dict[str, float]) -> Path:
    """Save aggregate metrics as a two-column CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    return path


def plot_error_summary(rows: list[dict[str, float]]) -> Path:
    """Plot voltage RMSE and firing-rate differences across current levels."""

    currents = [row["current_uA_cm2"] for row in rows]
    voltage_rmse = [row["voltage_rmse_mV"] for row in rows]
    firing_error = [row["firing_rate_difference_hz"] for row in rows]

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 7), sharex=True)
    axes[0].plot(currents, voltage_rmse, marker="o")
    axes[0].set_ylabel("Voltage RMSE (mV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].plot(currents, firing_error, marker="o")
    axes[1].set_xlabel("Injected current density (uA/cm2)")
    axes[1].set_ylabel("Firing rate error (Hz)")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    path = COMPARISON_DIR / "python_vs_neuron_error_summary.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main() -> None:
    """Run quantitative comparison on existing validation CSV outputs."""

    python_summary = load_summary(PYTHON_DIR / "python_hh_fi_summary.csv")
    neuron_summary = load_summary(NEURON_DIR / "neuron_hh_fi_summary.csv")
    rows: list[dict[str, float]] = []

    for current in CURRENT_LEVELS_UA_CM2:
        label = safe_current_label(current)
        python_trace = load_trace(PYTHON_DIR / "traces" / f"python_trace_{label}_uA_cm2.csv")
        neuron_trace = load_trace(NEURON_DIR / "traces" / f"neuron_trace_{label}_uA_cm2.csv")
        voltage_errors = compute_voltage_errors(python_trace, neuron_trace)
        rows.append(
            compare_current_level(
                current=current,
                python_summary=python_summary[current],
                neuron_summary=neuron_summary[current],
                voltage_errors=voltage_errors,
            )
        )

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    per_current_path = save_rows_csv(
        COMPARISON_DIR / "python_vs_neuron_per_current_errors.csv",
        rows,
    )
    aggregate_metrics = compute_aggregate_metrics(rows)
    aggregate_path = save_aggregate_metrics(
        COMPARISON_DIR / "python_vs_neuron_aggregate_errors.csv",
        aggregate_metrics,
    )
    plot_path = plot_error_summary(rows)

    print("Finished Python-vs-NEURON quantitative error analysis.")
    print(f"Per-current error table: {per_current_path}")
    print(f"Aggregate error table: {aggregate_path}")
    print(f"Error summary plot: {plot_path}")
    print()
    print("Key aggregate results:")
    for key, value in aggregate_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
