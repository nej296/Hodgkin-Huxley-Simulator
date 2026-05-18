"""Summary-level error analysis for Python HH and NEURON FI summaries.

This script compares already-generated summary CSV files row-by-row by injected
current level. It is intentionally separate from full voltage-trace comparison
because summary metrics are easier to inspect and report in the paper.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PYTHON_SUMMARY = (
    PROJECT_ROOT / "output" / "validation" / "python_hh" / "python_hh_fi_summary.csv"
)
NEURON_SUMMARY = (
    PROJECT_ROOT / "output" / "validation" / "neuron_hh" / "neuron_hh_fi_summary.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "validation" / "summary_comparison"

METRICS = [
    "spike_count",
    "firing_rate_hz",
    "peak_voltage_mV",
    "trough_voltage_mV",
    "first_spike_latency_ms",
]


def parse_float(value: str) -> float:
    """Parse a numeric CSV field, preserving NaN values."""

    value = value.strip()
    if value.lower() == "nan" or value == "":
        return np.nan
    return float(value)


def load_summary(path: Path) -> dict[float, dict[str, float]]:
    """Load a simulator summary CSV and index rows by current level."""

    rows: dict[float, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            current = parse_float(row["current_uA_cm2"])
            rows[current] = {key: parse_float(value) for key, value in row.items()}
    return rows


def valid_array(values: list[float]) -> np.ndarray:
    """Return numeric values with NaNs removed."""

    array = np.asarray(values, dtype=float)
    return array[~np.isnan(array)]


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize paired differences with deterministic error metrics."""

    array = valid_array(values)
    if len(array) == 0:
        return {
            "n": np.nan,
            "mean_difference": np.nan,
            "sd_difference": np.nan,
            "mean_absolute_difference": np.nan,
            "median_absolute_difference": np.nan,
            "rmse": np.nan,
            "max_absolute_difference": np.nan,
            "min_difference": np.nan,
            "max_difference": np.nan,
        }

    return {
        "n": int(len(array)),
        "mean_difference": float(np.mean(array)),
        "sd_difference": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "mean_absolute_difference": float(np.mean(np.abs(array))),
        "median_absolute_difference": float(np.median(np.abs(array))),
        "rmse": float(math.sqrt(np.mean(array**2))),
        "max_absolute_difference": float(np.max(np.abs(array))),
        "min_difference": float(np.min(array)),
        "max_difference": float(np.max(array)),
    }


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    """Compute Pearson correlation after dropping paired NaN values."""

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    mask = ~np.isnan(x_array) & ~np.isnan(y_array)
    if np.sum(mask) < 2:
        return np.nan
    return float(np.corrcoef(x_array[mask], y_array[mask])[0, 1])


def save_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    """Save rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    """Compare Python and NEURON summary metrics by current level."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    python_rows = load_summary(PYTHON_SUMMARY)
    neuron_rows = load_summary(NEURON_SUMMARY)
    currents = sorted(set(python_rows) & set(neuron_rows))

    per_current_rows: list[dict[str, float]] = []
    for current in currents:
        row: dict[str, float] = {"current_uA_cm2": current}
        for metric in METRICS:
            python_value = python_rows[current][metric]
            neuron_value = neuron_rows[current][metric]
            difference = (
                python_value - neuron_value
                if not (np.isnan(python_value) or np.isnan(neuron_value))
                else np.nan
            )
            row[f"python_{metric}"] = python_value
            row[f"neuron_{metric}"] = neuron_value
            row[f"{metric}_difference_python_minus_neuron"] = difference
            row[f"{metric}_absolute_difference"] = (
                abs(difference) if not np.isnan(difference) else np.nan
            )
        per_current_rows.append(row)

    aggregate_rows: list[dict[str, float]] = []
    for metric in METRICS:
        differences = [
            row[f"{metric}_difference_python_minus_neuron"]
            for row in per_current_rows
        ]
        python_values = [row[f"python_{metric}"] for row in per_current_rows]
        neuron_values = [row[f"neuron_{metric}"] for row in per_current_rows]
        summary = summarize(differences)
        aggregate_rows.append(
            {
                "metric": metric,
                "n": summary["n"],
                "mean_difference_python_minus_neuron": summary["mean_difference"],
                "sd_difference": summary["sd_difference"],
                "mean_absolute_difference": summary["mean_absolute_difference"],
                "median_absolute_difference": summary["median_absolute_difference"],
                "rmse": summary["rmse"],
                "max_absolute_difference": summary["max_absolute_difference"],
                "min_difference": summary["min_difference"],
                "max_difference": summary["max_difference"],
                "pearson_correlation": pearson_correlation(
                    python_values,
                    neuron_values,
                ),
            }
        )

    per_current_path = save_csv(
        OUTPUT_DIR / "per_current_summary_differences.csv",
        per_current_rows,
    )
    aggregate_path = save_csv(
        OUTPUT_DIR / "aggregate_summary_statistics.csv",
        aggregate_rows,
    )

    print("Finished summary-level error analysis.")
    print(f"Per-current differences: {per_current_path}")
    print(f"Aggregate statistics: {aggregate_path}")


if __name__ == "__main__":
    main()
