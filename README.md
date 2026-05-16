# Hodgkin-Huxley Simulator

A Python implementation of the classic single-compartment Hodgkin-Huxley neuron
model for studying how sodium and potassium conductances shape action-potential
generation, firing frequency, threshold behavior, and membrane voltage dynamics.

This repository is structured as a computational neuroscience research project:
the core model is separated from simulation protocols, plotting, analysis, data
export, and reproducible experiment scripts.

## Biological Background

The Hodgkin-Huxley model describes action-potential generation as the interaction
between membrane capacitance and voltage-gated ionic conductances. In this
formulation, the membrane voltage changes when injected current and ionic
currents are not balanced. Sodium channels drive rapid depolarization, potassium
channels drive repolarization and after-hyperpolarization, and leak current
pulls the membrane toward a passive reversal potential.

The main conductance parameters are:

- `g_na`: maximum sodium conductance density
- `g_k`: maximum potassium conductance density
- `g_l`: leak conductance density

Changing `g_na` and `g_k` changes spike amplitude, spike width, excitability,
firing frequency, and recovery dynamics.

## Model Equations

The membrane voltage equation is:

```text
C_m dV/dt = I_injected - I_Na - I_K - I_L
```

The ionic currents are:

```text
I_Na = g_Na m^3 h (V - E_Na)
I_K  = g_K n^4 (V - E_K)
I_L  = g_L (V - E_L)
```

The gating variables follow first-order kinetics:

```text
dm/dt = alpha_m(V) (1 - m) - beta_m(V) m
dh/dt = alpha_h(V) (1 - h) - beta_h(V) h
dn/dt = alpha_n(V) (1 - n) - beta_n(V) n
```

Variables:

- `V`: membrane voltage in mV
- `m`: sodium activation gate
- `h`: sodium inactivation gate
- `n`: potassium activation gate
- `C_m`: membrane capacitance density in uF/cm^2
- `I_injected`: applied current density in uA/cm^2

The default parameterization follows the original squid giant axon model:

| Parameter | Default | Units |
| --- | ---: | --- |
| `C_m` | 1.0 | uF/cm^2 |
| `g_na` | 120.0 | mS/cm^2 |
| `g_k` | 36.0 | mS/cm^2 |
| `g_l` | 0.3 | mS/cm^2 |
| `E_Na` | 50.0 | mV |
| `E_K` | -77.0 | mV |
| `E_L` | -54.387 | mV |

## Numerical Methods

The simulator uses fixed-step explicit integration. The default method is
fourth-order Runge-Kutta (`rk4`), with forward Euler (`euler`) also available
for teaching and comparison. Current protocols are evaluated on the simulation
time grid and held constant over each integration step.

The gating-rate equations contain removable singularities at specific voltages.
The implementation evaluates those terms with a stable helper so simulations
remain finite at the singular points.

## Repository Structure

```text
.
|-- src/
|   |-- models/          # Hodgkin-Huxley model and parameters
|   |-- simulation/      # configuration, protocols, integration loop
|   |-- visualization/   # matplotlib plotting utilities
|   |-- analysis/        # spike metrics and validation comparisons
|   `-- utils/           # CSV export and support utilities
|-- experiments/         # reproducible experiment scripts
|-- notebooks/           # exploratory analysis notebooks
|-- docs/                # scientific and validation notes
|-- tests/               # automated tests
|-- data/                # input/reference data
|-- output/              # generated CSV and plot files, ignored by git
|-- app.py               # interactive desktop simulator
|-- main.py              # configurable simulation entry point
`-- requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Usage

Launch the interactive membrane dynamics simulator:

```bash
python app.py
```

The desktop app lets you adjust sodium, potassium, and leak conductances;
reversal potentials; injected-current timing and amplitude; time step; and
integration method. Press **Run Simulation** to update the membrane voltage,
and injected-current plots. Use **Export CSV** or **Save Plot** to save the
active simulation. The voltage plot includes dashed reference lines for the
sodium and potassium reversal potentials (`E_Na` and `E_K`).

Run the default current-step simulation:

```bash
python main.py
```

This writes:

- `output/simulation.csv`
- `output/simulation.png`

Run the reproducible basic experiment:

```bash
python experiments/run_basic_simulation.py
```

Run sodium and potassium conductance sweeps:

```bash
python experiments/conductance_sweep.py
```

Customize a simulation from the command line:

```bash
python main.py --current 12 --start-ms 5 --end-ms 40 --g-na 140 --g-k 30
```

## Testing

```bash
pytest
```

The current tests verify that:

- HH rate functions remain finite at singular voltages
- a standard depolarizing current step evokes an action potential
- current-step protocols return the expected injected current values

## Current Features

- Standard Hodgkin-Huxley equations
- Single-compartment membrane voltage simulation
- Sodium, potassium, and leak currents
- Gating variables `m`, `h`, and `n`
- Adjustable sodium and potassium conductances
- Adjustable step-current injection
- Interactive desktop simulator for membrane dynamics
- RK4 and Euler integration
- Voltage and injected-current plotting
- CSV export of voltage, gates, and current
- Spike-count and firing-rate analysis
- Conductance sweep experiments
- Trace comparison utilities for future validation against NEURON

## Roadmap

Planned extensions are organized around future research use:

- NEURON comparison experiments using matched protocols and reference traces
- multicompartment cable models
- morphology loading from SWC files
- synaptic conductances and current-based synapses
- stochastic ion-channel variants
- parameter sweeps with structured result directories
- network simulations
- GPU-accelerated integration backends
- notebook-based analysis reports for conductance-dependent firing behavior

## References

Hodgkin, A. L., and Huxley, A. F. (1952). A quantitative description of
membrane current and its application to conduction and excitation in nerve.
Journal of Physiology, 117(4), 500-544.

Dayan, P., and Abbott, L. F. (2001). Theoretical Neuroscience: Computational
and Mathematical Modeling of Neural Systems. MIT Press.

Koch, C. (1999). Biophysics of Computation: Information Processing in Single
Neurons. Oxford University Press.
