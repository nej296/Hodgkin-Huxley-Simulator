# Building and Validating a Single-Compartment Hodgkin-Huxley Simulator in Python

An interactive Python desktop simulator for exploring Hodgkin-Huxley membrane
dynamics and validating the model against NEURON.

## Quick Start: Open the Desktop App

These steps run the same Python/Tkinter simulator used in this repository.

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
2. Download this repository from GitHub or clone it with git.
3. Open PowerShell in the `Hodgkin-Huxley-Simulator` folder.
4. Run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

After the first setup, reopen the app from the same folder with:

```powershell
.\.venv\Scripts\activate
python app.py
```

## Abstract

The simulator implements voltage-dependent sodium, potassium, and leak currents
using the classical Hodgkin-Huxley conductance-based framework. Users can adjust
ionic conductances, reversal potentials, membrane capacitance, current-injection
settings, simulation duration, time step, and numerical integration method. The
main scientific question was whether a simple Python implementation could
reproduce the firing behavior of an equivalent single-compartment
Hodgkin-Huxley model in NEURON.

Validation was performed across injected current amplitudes from 0 to
200 uA/cm2. The Python simulator and NEURON reference model matched exactly in
spike count, firing rate, threshold current, and the overall frequency-current
relationship. Both models began firing at 3 uA/cm2, reached a maximum firing
rate of 133.33 Hz at 60 uA/cm2, and showed reduced repetitive firing at higher
current amplitudes consistent with depolarization block. Voltage-level
differences were small: the mean absolute peak-voltage difference was
approximately 0.171 mV, the mean absolute trough-voltage difference was
approximately 0.044 mV, and the mean absolute first-spike latency difference was
approximately 0.003 ms.

## Scientific Question

The validation experiment asked:

> Can a custom Python implementation of the single-compartment Hodgkin-Huxley
> model reproduce the voltage dynamics and current-dependent firing behavior of
> an equivalent NEURON implementation?

The primary test was the frequency-current relationship: firing rate was
measured while injected current density was increased across a defined range.
This provides a direct way to test whether the simulator reproduces threshold
behavior, repetitive firing, and high-current depolarization block.

## Model Summary

The simulator represents the neuron as a single isopotential compartment. The
membrane equation is:

```text
C_m dV/dt = I_inj - I_Na - I_K - I_L
```

The ionic currents are:

```text
I_Na = g_Na m^3 h (V - E_Na)
I_K  = g_K n^4 (V - E_K)
I_L  = g_L (V - E_L)
```

The gating variables follow first-order voltage-dependent kinetics:

```text
dm/dt = alpha_m(V)(1 - m) - beta_m(V)m
dh/dt = alpha_h(V)(1 - h) - beta_h(V)h
dn/dt = alpha_n(V)(1 - n) - beta_n(V)n
```

The simulator uses fixed-step fourth-order Runge-Kutta integration by default.
Forward Euler integration is also available for comparison and teaching.

### Default Parameters

| Parameter | Default | Units | Description |
| --- | ---: | --- | --- |
| `C_m` | 1.0 | uF/cm2 | membrane capacitance density |
| `g_Na` | 120.0 | mS/cm2 | maximum sodium conductance |
| `g_K` | 36.0 | mS/cm2 | maximum potassium conductance |
| `g_L` | 0.3 | mS/cm2 | leak conductance |
| `E_Na` | 50.0 | mV | sodium reversal potential |
| `E_K` | -77.0 | mV | potassium reversal potential |
| `E_L` | -54.387 | mV | leak reversal potential |

## Interactive Simulator

Launch the desktop interface with:

```bash
python app.py
```

The interface provides controls for:

- simulation duration and time step
- initial membrane voltage
- numerical integration method (`rk4` or `euler`)
- current-step amplitude, start time, end time, and baseline
- sodium, potassium, and leak conductances
- sodium, potassium, and leak reversal potentials
- membrane capacitance

The interface displays:

- membrane voltage over time
- injected current over time
- sodium and potassium reversal-potential reference lines
- spike count, firing rate, peak voltage, trough voltage, and first-spike time

CSV export and plot saving are supported from the graphical interface.

## Validation Against NEURON

NEURON was used as an independent reference simulator. The NEURON validation
model used a single compartment with NEURON's built-in `hh` mechanism, matching
the Python simulator's Hodgkin-Huxley conductances, reversal potentials,
capacitance, initial voltage, duration, time step, and current-injection timing.

### Density-Current Clamp

A unit mismatch was identified during validation. The Python simulator applies
current as current density in `uA/cm2`, while NEURON's standard `IClamp` applies
point current in `nA`. Using `IClamp` would require converting current density
into total current through a geometry-dependent area calculation.

To avoid that conversion, this project uses a local NMODL mechanism,
`density_clamp.mod`, which defines a density-current clamp called `dclamp`.
This mechanism allows current to be specified directly in `uA/cm2`, matching the
Python simulator. The `dclamp` mechanism only controls the injected-current
protocol; the membrane dynamics in NEURON are still generated by NEURON's
standard Hodgkin-Huxley `hh` mechanism.

## Validation Protocol

Both simulators were run with:

| Setting | Value |
| --- | ---: |
| Current amplitudes | 0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 200 uA/cm2 |
| Simulation duration | 100 ms |
| Time step | 0.01 ms |
| Initial voltage | -65 mV |
| Current start | 5 ms |
| Current end | 80 ms |
| Spike threshold | 0 mV |
| Refractory window | 2 ms |

For each current amplitude, voltage traces were exported and summarized using
the same spike-detection criteria. Summary metrics included spike count, firing
rate, peak voltage, trough voltage, first-spike latency, and the resulting
frequency-current curve.

## Key Validation Results

| Metric | Result |
| --- | --- |
| Threshold current | 3 uA/cm2 in both simulators |
| Maximum firing rate | 133.33 Hz at 60 uA/cm2 in both simulators |
| Spike count agreement | Exact match across all tested current amplitudes |
| Firing-rate agreement | Exact match across all tested current amplitudes |
| Mean absolute peak-voltage difference | approximately 0.171 mV |
| Mean absolute trough-voltage difference | approximately 0.044 mV |
| Mean absolute first-spike latency difference | approximately 0.003 ms |

The remaining differences were voltage-level differences rather than differences
in firing behavior. These small differences are consistent with expected
implementation and solver differences between a custom Python RK4 simulator and
NEURON's internal Hodgkin-Huxley implementation.

## Installation

Clone the repository:

```bash
git clone https://github.com/nej296/Hodgkin-Huxley-Simulator.git
cd Hodgkin-Huxley-Simulator
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

NEURON is required only for the NEURON validation workflow.

## Usage

Run the graphical simulator:

```bash
python app.py
```

Run a default command-line simulation:

```bash
python main.py
```

Run a custom current-step simulation:

```bash
python main.py --current 12 --start-ms 5 --end-ms 40 --g-na 140 --g-k 30
```

Generated command-line outputs are written to:

```text
output/simulation.csv
output/simulation.png
```

The `output/` directory is intentionally ignored by git except for
`output/.gitkeep`.

## Reproducing the Validation Workflow

Compile the NEURON density-current clamp mechanism:

```bash
nrnivmodl neuron_validation
```

Run the Python validation sweep:

```bash
python neuron_validation/run_python_validation_sweep.py
```

Run the NEURON validation sweep:

```bash
python neuron_validation/run_neuron_dclamp_sweep.py
```

Compare the generated outputs:

```bash
python neuron_validation/compare_python_neuron_outputs.py
python neuron_validation/compare_summary_statistics.py
```

Generated validation traces, plots, and summary CSV files are written under:

```text
output/validation/
```

## R Plotting Scripts

The `rstudio/` directory contains R scripts for plotting validation summaries
and frequency-current curves:

- `validation_summary_plots.R`
- `neuron_fi_curve_0_to_60.R`
- `python_fi_curve_full_range.R`
- `neuron_fi_curve_full_range.R`

These scripts are intended for generating publication-style figures from the
exported CSV summaries.

## Repository Structure

```text
.
|-- app.py                         # interactive desktop simulator
|-- main.py                        # command-line simulation entry point
|-- src/
|   |-- models/                    # Hodgkin-Huxley model and parameters
|   |-- simulation/                # configuration, protocols, integration loop
|   |-- analysis/                  # spike metrics and validation comparisons
|   |-- visualization/             # plotting utilities
|   `-- utils/                     # CSV export helpers
|-- experiments/                   # reproducible simulation scripts
|-- neuron_validation/             # NEURON validation workflow and dclamp NMODL
|-- rstudio/                       # R scripts for validation plots
|-- tests/                         # automated tests
|-- docs/                          # validation and scientific notes
|-- data/                          # input/reference data placeholder
|-- output/                        # generated outputs, ignored by git
|-- requirements.txt
`-- pyproject.toml
```

## Testing

Run the test suite:

```bash
python -m pytest
```

The current tests verify:

- finite Hodgkin-Huxley rate functions at removable singularities
- action-potential generation under a standard current step
- current-step protocol behavior
- gating variables remaining in the physical probability range

## Limitations

This simulator is deterministic and single-compartment. It assumes spatially
uniform membrane voltage and does not include dendrites, axons, cable dynamics,
synaptic inputs, stochastic ion channels, or network interactions. Validation
was performed against an equivalent NEURON implementation rather than against
experimental electrophysiology recordings. Therefore, the validation shows that
the Python simulator accurately reproduces the matched Hodgkin-Huxley model, not
the full biological complexity of real neurons.

## Future Work

Potential extensions include:

- multicompartment morphology
- synaptic conductances
- additional current-clamp protocols
- improved compatibility between point-current and density-current workflows
- side-by-side Python and NEURON trace overlays
- gating-variable visualization
- experimental electrophysiology comparison
- improved educational user-interface design

## References

Hodgkin, A. L., and Huxley, A. F. (1952). A quantitative description of
membrane current and its application to conduction and excitation in nerve.
Journal of Physiology, 117(4), 500-544.

Hines, M. L., and Carnevale, N. T. (1997). The NEURON simulation environment.
Neural Computation, 9(6), 1179-1209.

Bianchi, D., Marasco, A., Limongiello, A., Marchetti, C., Marie, H., Tirozzi,
B., and Migliore, M. (2012). On the mechanisms underlying the depolarization
block in the spiking dynamics of CA1 pyramidal neurons. Journal of
Computational Neuroscience, 33(2), 207-225.
