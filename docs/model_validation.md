# Validation Framework

This project includes a small trace-comparison layer in `src/analysis/comparison.py`.
It is designed for future validation against NEURON simulator outputs or other
trusted computational neuroscience tools.

The expected workflow is:

1. Run this Python simulator with a fixed current protocol and parameter set.
2. Export the membrane voltage trace as CSV.
3. Run the same protocol and parameters in NEURON.
4. Load both traces and compare them with `compare_voltage_traces`.

The comparison function currently reports:

- root-mean-square voltage error in mV
- maximum absolute voltage error in mV
- peak voltage error in mV
- number of compared samples

Future validation work should add tolerances for specific protocols, comparison
plots, and regression tests that run against committed reference traces.
